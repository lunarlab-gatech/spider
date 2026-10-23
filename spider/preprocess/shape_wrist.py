"""Reshape the free-motion wrist path of an IK reference for an easy arm IK job.

An arm attached to the wrist should have an easy job. Arm-agnostic: the costs use only the human body estimate and the scene.

The hand stays as IK placed it in every grip span (>= 2 tracked finger contacts, padded), so the
grasp does not change; only the motion between grip spans moves. All costs are soft: zero inside
a comfortable zone and squared outside it.

1. reach: wrist within reach_frac of the arm length from the shoulder, averaged over a spread
   of shoulder offsets and arm lengths;
2. forearm: a short thin capsule behind the wrist, along the hand back axis, stays clear of the
   table, the object and the fixtures, and that axis stays within a cone around the direction
   to the shoulder;
3. orientation: total swing rotation (rotation about the grasp axis is free) and the forearm
   roll accumulated beyond a band;
4. smoothness: acceleration, jerk, and speed / angular speed beyond comfortable limits;
5. clearance under the wrist from the table and the object.

The body estimate comes from --body, an npz with `shoulder` (N, 3) and `heading` (N, 3, 3) on
the export clock and the scalar `arm_len_m` (the xr-to-trajectory-pipeline writes it with
`xr-spider-body`). Row k is taken as reference frame k: IK drops up to two export frames, 40 ms,
well inside the 1 s smoothing of the estimate. The spread is drawn here, seeded.

Off by default in the pipeline (ARM_SHAPE=1). Writes trajectory_kinematic.npz in place and keeps
the IK output as trajectory_kinematic_unshaped.npz, plus trajectory_kinematic_shape.json (costs
before and after, largest shift, weights).

usage: python spider/preprocess/shape_wrist.py --run-dir RUN --body body_estimate.npz
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

HAND_BODY = "right_hand_base"
BACK_AXIS = (0.0, 1.0, 0.0)  # hand frame: fingers along -y, so +y points to the forearm


@dataclass
class Weights:
    """Soft cost weights and comfortable zones; the defaults are the 2026-10-22 rec07 settings."""

    reach_frac: float = 0.85
    reach_min_frac: float = 0.3
    w_reach: float = 300.0
    forearm_len: float = 0.10
    forearm_radius: float = 0.035
    forearm_cone_deg: float = 45.0
    w_forearm: float = 300.0
    w_cone: float = 30.0
    w_swing: float = 0.5  # per rad of swing rotation
    roll_band: float = 1.0  # rad of forearm roll allowed before it costs
    w_roll: float = 10.0
    w_acc: float = 2e-3
    w_jerk: float = 2e-7
    v_comf: float = 0.5  # m/s
    w_comf: float = 1.5  # rad/s
    w_speed: float = 10.0
    wrist_clear: float = 0.08  # m above the table, away from grips
    obj_clear: float = 0.05  # m from the object surface, away from grips
    ramp_s: float = 0.4  # clearances ramp to zero over this time before / after a grip
    w_clear: float = 1000.0
    w_track: float = 1.0
    pin_pad_s: float = 0.1
    spread_n: int = 64  # shoulder samples
    spread_offset_sd: float = 0.03  # m, per axis, in the heading frame
    spread_len_log_sd: float = 0.06  # arm length spread
    grip_min_fingers: int = 2


def euler_chain_to_matrix(a: np.ndarray) -> np.ndarray:
    """Hinge chain rot_x, rot_y, rot_z of the floating hand: R = Rx Ry Rz (intrinsic XYZ)."""
    return Rotation.from_euler("XYZ", a).as_matrix()


def matrix_to_euler_chain(m: np.ndarray, near: np.ndarray) -> np.ndarray:
    """Hinge angles for rotations m, on the branch and 2 pi wrap nearest to `near` per frame.

    XYZ angles have two solutions, (a, b, c) and (a + pi, pi - b, c + pi); the reference can sit
    on either, and MJWP tracks the joint angles, so a branch or wrap change would spin the hand.
    """
    a = Rotation.from_matrix(m).as_euler("XYZ")
    alt = a + np.array([np.pi, 0.0, np.pi])
    alt[:, 1] = np.pi - a[:, 1]

    def wrap(x):
        return near + (x - near + np.pi) % (2 * np.pi) - np.pi

    a, alt = wrap(a), wrap(alt)
    pick = np.abs(alt - near).sum(1) < np.abs(a - near).sum(1)
    return np.where(pick[:, None], alt, a)


def free_mask(
    contact: np.ndarray, dt: float, pad_s: float, min_fingers: int
) -> np.ndarray:
    """True on frames the shaping may move: outside every grip (>= min_fingers contacts), padded."""
    touch = contact.sum(1) >= min_fingers
    pad = int(round(pad_s / dt))
    pinned = np.zeros_like(touch)
    for i in np.flatnonzero(touch):
        pinned[max(0, i - pad) : i + pad + 1] = True
    return ~pinned


def grip_distance_s(free: np.ndarray, dt: float) -> np.ndarray:
    """Time to the nearest pinned frame, s (inf when nothing is pinned)."""
    idx = np.flatnonzero(~free)
    if idx.size == 0:
        return np.full(len(free), np.inf)
    k = np.arange(len(free))
    return np.abs(k[:, None] - idx[None]).min(1) * dt


# ------------------------------------------------------------------ torch helpers


def quat_mul(a, b):
    """Hamilton product of wxyz quaternions."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        -1,
    )


def quat_conj(q):
    """Conjugate of a wxyz quaternion."""
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q.dtype)


def quat_exp(v):
    """Unit quaternion of a rotation vector."""
    th = torch.sqrt((v * v).sum(-1, keepdim=True) + 1e-12)
    return torch.cat([torch.cos(th / 2), torch.sin(th / 2) * v / th], -1)


def quat_log(q):
    """Rotation vector of a unit quaternion (shortest arc)."""
    q = q * torch.sign(q[..., :1] + 1e-12)  # shortest arc
    vn = torch.sqrt((q[..., 1:] ** 2).sum(-1, keepdim=True) + 1e-12)
    return 2 * torch.atan2(vn, q[..., :1]) * q[..., 1:] / vn


def quat_rotate(q, v):
    """Rotate vectors v by quaternions q."""
    qv = torch.cat([torch.zeros_like(v[..., :1]), v], -1)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[..., 1:]


def hinge2(x):
    """Squared hinge: zero for x <= 0, x**2 above."""
    return torch.relu(x) ** 2


def cyl_dist(p, c, axis, radius, half):
    """Distance from points p (..., 3) to a capped cylinder (center c, unit axis, radius, half height)."""
    d = p - c
    h = (d * axis).sum(-1)
    r = torch.sqrt(((d - h[..., None] * axis) ** 2).sum(-1) + 1e-12)
    out_r = torch.relu(r - radius)
    out_h = torch.relu(h.abs() - half)
    inside = torch.minimum(radius - r, half - h.abs())
    return torch.where(
        (out_r > 0) | (out_h > 0), torch.sqrt(out_r**2 + out_h**2 + 1e-12), -inside
    )


def box_dist(p, lo, hi):
    """Distance from points to an axis-aligned box (negative inside)."""
    q = torch.maximum(lo - p, p - hi)
    outside = torch.sqrt((torch.relu(q) ** 2).sum(-1) + 1e-12)
    return torch.where((q > 0).any(-1), outside, q.max(-1).values)


# ------------------------------------------------------------------ scene geometry


def object_cylinder(model: mujoco.MjModel) -> tuple[float, float, float]:
    """Radius, half height and z centre of the object collision parts in the object frame."""
    bid = model.body("right_object").id
    pts = []
    for g in range(model.ngeom):
        if (
            model.geom_bodyid[g] != bid
            or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
            or model.geom_group[g] != 3
        ):
            continue
        mid = model.geom_dataid[g]
        v = model.mesh_vert[
            model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]
        ]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[g])
        pts.append(v @ rot.reshape(3, 3).T + model.geom_pos[g])
    p = np.concatenate(pts)
    return (
        float(np.linalg.norm(p[:, :2], axis=1).max()),
        float(np.ptp(p[:, 2]) / 2),
        float((p[:, 2].max() + p[:, 2].min()) / 2),
    )


def fixture_boxes(model: mujoco.MjModel) -> list[tuple[np.ndarray, np.ndarray]]:
    """World AABBs of the static fixture collision parts (bodies named fixture_*)."""
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    boxes = []
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        if (
            not model.body(b).name.startswith("fixture_")
            or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
        ):
            continue
        if model.geom_group[g] != 3:
            continue
        mid = model.geom_dataid[g]
        v = model.mesh_vert[
            model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]
        ]
        w = v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        boxes.append((w.min(0), w.max(0)))
    return boxes


# ------------------------------------------------------------------ main


def shape(
    qpos: np.ndarray,
    contact: np.ndarray,
    dt: float,
    body: dict,
    model: mujoco.MjModel,
    w: Weights,
    iters: int = 400,
) -> tuple[np.ndarray, dict]:
    """Return the reshaped qpos and a cost report. Only the 6 floating-hand joints change."""
    torch.set_default_dtype(torch.float64)
    n = len(qpos)
    free = free_mask(contact, dt, w.pin_pad_s, w.grip_min_fingers)
    if not free.any():
        return qpos.copy(), {"free_frames": 0}
    gdist = torch.as_tensor(grip_distance_s(free, dt))
    ramp = torch.clamp(gdist / w.ramp_s, 0, 1)
    p_ref = torch.as_tensor(qpos[:, :3])
    rot_ref = euler_chain_to_matrix(qpos[:, 3:6])
    q_ref = torch.as_tensor(Rotation.from_matrix(rot_ref).as_quat()[:, [3, 0, 1, 2]])
    sh = torch.as_tensor(body["shoulder"])
    hd = torch.as_tensor(body["heading"])
    rng = np.random.default_rng(0)
    offs = torch.as_tensor(rng.normal(0, w.spread_offset_sd, (w.spread_n, 3)))
    lens = torch.as_tensor(
        float(body["arm_len_m"])
        * np.exp(rng.normal(0, w.spread_len_log_sd, w.spread_n))
    )
    shoulders = sh[:, None] + torch.einsum("nij,kj->nki", hd, offs)  # (N, K, 3)
    # grasp axis: object +z in the hand frame, mean over contact frames
    obj_q = qpos[:, -4:]  # wxyz
    rot_obj = Rotation.from_quat(obj_q[:, [1, 2, 3, 0]]).as_matrix()
    touch = ~free
    ga = np.einsum("nji,nj->ni", rot_ref[touch], rot_obj[touch][:, :, 2]).mean(0)
    ga = torch.as_tensor(ga / np.linalg.norm(ga))
    back = torch.as_tensor(BACK_AXIS)
    rad, half, zc = object_cylinder(model)
    obj_c = torch.as_tensor(
        qpos[:, -7:-4] + np.einsum("nij,j->ni", rot_obj, [0, 0, zc])
    )
    obj_ax = torch.as_tensor(rot_obj[:, :, 2])
    boxes = [
        (torch.as_tensor(lo), torch.as_tensor(hi)) for lo, hi in fixture_boxes(model)
    ]
    fm = torch.as_tensor(free)
    # pinned frames hold the reference height; the table clearance ramps from it
    z_floor = (
        torch.minimum(torch.as_tensor(w.wrist_clear), p_ref[:, 2]) * (1 - ramp)
        + w.wrist_clear * ramp
    )
    dp = torch.zeros(n, 3, requires_grad=True)
    dth = torch.zeros(n, 3, requires_grad=True)
    cos_cone = float(np.cos(np.radians(w.forearm_cone_deg)))
    seg = np.split(np.arange(n), np.flatnonzero(np.diff(free.astype(int))) + 1)
    segs = [torch.as_tensor(s) for s in seg if free[s[0]]]

    def terms():
        p = p_ref + dp * fm[:, None]
        q = quat_mul(quat_exp(dth * fm[:, None]), q_ref)
        v = (p[1:] - p[:-1]) / dt
        a = (v[1:] - v[:-1]) / dt
        j = (a[1:] - a[:-1]) / dt
        om = quat_log(quat_mul(q[1:], quat_conj(q[:-1]))) / dt  # (N-1, 3) world
        al = (om[1:] - om[:-1]) / dt
        bw = quat_rotate(q, back.expand(n, 3))  # back axis, world
        gw = quat_rotate(q[:-1], ga.expand(n - 1, 3))
        om_sw = om - (om * gw).sum(-1, keepdim=True) * gw
        f = fm.double()
        fv = fm[1:] | fm[:-1]
        # 1. reach over the spread
        dist = torch.linalg.norm(p[:, None] - shoulders, dim=-1)
        c_reach = (
            hinge2(dist - w.reach_frac * lens) + hinge2(w.reach_min_frac * lens - dist)
        ).mean(1)
        # 2. forearm: a capsule behind the wrist along the hand back axis (every arm has a wrist
        #    housing there), and a cone that keeps that axis pointing toward the shoulder
        u = sh - p
        u = u / torch.linalg.norm(u, dim=-1, keepdim=True)
        ls = torch.linspace(0.0, w.forearm_len, 5)[1:]
        caps = p[:, None] + bw[:, None] * ls[None, :, None]  # (N, 4, 3)
        need = w.forearm_radius + 0.01
        c_fore = hinge2(need - caps[..., 2]).sum(1)
        c_fore = c_fore + hinge2(
            need - cyl_dist(caps, obj_c[:, None], obj_ax[:, None], rad, half)
        ).sum(1)
        for lo, hi in boxes:
            c_fore = c_fore + hinge2(need - box_dist(caps, lo, hi)).sum(1)
        c_fore = c_fore * ramp
        c_cone = hinge2(cos_cone - (bw * u).sum(-1))
        # 3. orientation: swing rotation (L1 over time) and forearm roll beyond the band, per free span
        c_swing = torch.sqrt((om_sw**2).sum(-1) + 1e-6) * dt
        c_roll = torch.zeros(())
        roll_rate = (om * quat_rotate(q[:-1], back.expand(n - 1, 3))).sum(-1) * dt
        for s in segs:
            s = s[s < n - 1]
            if len(s):
                c_roll = (
                    c_roll
                    + hinge2(torch.cumsum(roll_rate[s], 0).abs() - w.roll_band).sum()
                )
        # 4. smoothness
        c_acc = (a**2).sum(-1).sum() + (al**2).sum(-1).sum()
        c_jerk = (j**2).sum(-1).sum()
        c_speed = hinge2(torch.linalg.norm(v, dim=-1) - w.v_comf) + hinge2(
            torch.sqrt((om**2).sum(-1) + 1e-12) - w.w_comf
        )
        # 5. clearance under the wrist (table) and from the object, ramped near grips
        c_clear = (
            hinge2(z_floor - p[:, 2])
            + hinge2(w.obj_clear - cyl_dist(p, obj_c, obj_ax, rad, half)) * ramp
        )
        c_track = (dp**2).sum(-1) + 0.1 * (dth**2).sum(-1)
        return {
            "reach": w.w_reach * (c_reach * f).sum(),
            "forearm": w.w_forearm * (c_fore * f).sum() + w.w_cone * (c_cone * f).sum(),
            "swing": w.w_swing * (c_swing * fv).sum(),
            "roll": w.w_roll * c_roll,
            "smooth": w.w_acc * c_acc
            + w.w_jerk * c_jerk
            + w.w_speed * (c_speed * fv).sum(),
            "clear": w.w_clear * (c_clear * f).sum(),
            "track": w.w_track * (c_track * f).sum(),
        }

    def total():
        return sum(terms().values())

    with torch.no_grad():
        before = {k: float(v) for k, v in terms().items()}
    opt = torch.optim.LBFGS(
        [dp, dth],
        lr=1.0,
        max_iter=iters,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=50,
    )

    def closure():
        opt.zero_grad()
        c = total()
        c.backward()
        return c

    opt.step(closure)
    with torch.no_grad():
        after = {k: float(v) for k, v in terms().items()}
        p = (p_ref + dp * fm[:, None]).numpy()
        q = quat_mul(quat_exp(dth * fm[:, None]), q_ref).numpy()
    out = qpos.copy()
    out[:, :3] = p
    out[:, 3:6] = matrix_to_euler_chain(
        Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix(), qpos[:, 3:6]
    )
    out[~free] = qpos[~free]
    step = np.abs(np.diff(out[:, 3:6], axis=0)).max()
    ref_step = np.abs(np.diff(qpos[:, 3:6], axis=0)).max()
    if step > max(0.5, 2 * ref_step):
        raise RuntimeError(
            f"hinge angles jump {step:.3f} rad in one frame (reference {ref_step:.3f})"
        )
    report = {
        "free_frames": int(free.sum()),
        "cost_before": before,
        "cost_after": after,
        "max_shift_m": float(np.linalg.norm(p - qpos[:, :3], axis=1).max()),
        "weights": asdict(w),
    }
    return out, report


def main():
    """Shape trajectory_kinematic.npz in place from the command line."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--run-dir",
        required=True,
        help="directory with trajectory_kinematic.npz; scene.xml one level up",
    )
    ap.add_argument("--body", required=True)
    ap.add_argument(
        "--weights", default=None, help="JSON object of Weights fields to override"
    )
    args = ap.parse_args()
    run = Path(args.run_dir)
    src = run / "trajectory_kinematic.npz"
    keep = run / "trajectory_kinematic_unshaped.npz"
    if keep.exists():
        raise FileExistsError(
            f"{keep} exists: the reference is already shaped; rerun IK first"
        )
    d = dict(np.load(src))
    model = mujoco.MjModel.from_xml_path(str(run.parent / "scene.xml"))
    hand = model.body(HAND_BODY)
    obj = model.joint("right_object_joint")
    if (
        hand.jntnum != 6
        or model.jnt_qposadr[hand.jntadr] != 0
        or model.jnt_qposadr[obj.id] != model.nq - 7
    ):
        raise ValueError(
            f"shape_wrist needs the right hand's 6 floating joints first and the right object's "
            f"free joint last in qpos; {run.parent / 'scene.xml'} differs"
        )
    body = dict(np.load(args.body))
    n = len(d["qpos"])
    if len(body["shoulder"]) < n:
        raise ValueError(
            f"body estimate has {len(body['shoulder'])} frames, the reference {n}"
        )
    body = {
        "shoulder": body["shoulder"][:n],
        "heading": body["heading"][:n],
        "arm_len_m": body["arm_len_m"],
    }
    w = Weights(**(json.loads(args.weights) if args.weights else {}))
    dt = 1.0 / float(d["frequency"])
    qpos, report = shape(d["qpos"], d["contact"], dt, body, model, w)
    qvel = d["qvel"].copy()
    for i in range(1, len(qpos)):
        mujoco.mj_differentiatePos(model, qvel[i], dt, qpos[i - 1], qpos[i])
    d["qpos"], d["qvel"] = qpos, qvel
    src.rename(keep)
    np.savez(src, **d)
    (run / "trajectory_kinematic_shape.json").write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("free_frames", "cost_before", "cost_after", "max_shift_m")
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
