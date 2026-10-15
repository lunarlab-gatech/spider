"""Run IK for the given hand type and mode.

This is a simplified version of IK based on mink.
No act scene or open-hand support. When detect_contact.py wrote per-span
object-surface contact points and normals, the fingertip site of a finger in
contact tracks the point moved out along the normal by the radius of the
fingertip collision geom, so the geom end touches the surface with its tip.
The target blends from the human fingertip to that point over the first
contact_ramp_s of each span and back over the last (see contact_targets and
contact_blend_weights), and the output carries the contact points for the MJWP
contact reward. With object_clearance set, the hand
collision geoms keep that distance from the object and fixture collision geoms:
a hard limit stops the hand moving in, and PenetrationTask pushes out geoms the
moving object stepped into, so the reference does not ask MJWP to hold fingers
inside an object.

Input: mujoco scene xml file + hand keypoints + object trajectories.
Output: npz file which contains qpos for key frames in xml.

Strategy:
1. Frame 0: set wrist targets first, integrate many steps, then add finger
   tip targets and integrate more steps.
2. Subsequent frames: solve IK with all targets, integrating sim_dt steps
   for each ref_dt interval.
3. Object qpos is set directly (no retargeting).

Author: Chaoyi Pan
Date: 2026-03-07
"""

import json
import os
from pathlib import Path

import loguru
import mink
import mujoco
import numpy as np
import tyro
from loop_rate_limiters import RateLimiter

from spider import ROOT
from spider.io import get_processed_data_dir
from spider.mujoco_utils import get_viewer

# MANO 21-joint skeleton bones (parent -> child) for reference visualization
_MANO_SKELETON_BONES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]
_MANO_FINGERTIP_INDICES = [4, 8, 12, 16, 20]


def _build_ref_skeleton(server, mano_assets_root, qpos_ref, ref_idx, embodiment_type):
    """Render the reference MANO skeleton in viser using fingertip + wrist targets.

    Returns a callable update_fn(t) -> None that updates joint/bone positions
    for frame t. We don't actually need MANO FK here because qpos_ref already
    has wrist + 5 fingertips per frame; we draw a simple wrist+fingertip
    skeleton plus straight bones (wrist->each fingertip).
    """
    if "right" in embodiment_type or embodiment_type == "bimanual":
        wrist_idx = ref_idx["right_palm"]
        tip_idxs = [
            ref_idx[f"right_{n}"]
            for n in ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
        ]
    elif "left" in embodiment_type:
        wrist_idx = ref_idx["left_palm"]
        tip_idxs = [
            ref_idx[f"left_{n}"]
            for n in ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
        ]
    else:
        return None

    wrist_h = server.scene.add_icosphere(
        "ref_hand/wrist",
        radius=0.012,
        color=(80, 130, 220),
        position=tuple(qpos_ref[0, wrist_idx, :3]),
    )
    tip_handles = [
        server.scene.add_icosphere(
            f"ref_hand/tip_{i}",
            radius=0.008,
            color=(220, 80, 80),
            position=tuple(qpos_ref[0, tidx, :3]),
        )
        for i, tidx in enumerate(tip_idxs)
    ]
    bone_handles = []
    for tidx in tip_idxs:
        h = server.scene.add_spline_catmull_rom(
            f"ref_hand/bone_w_to_{tidx}",
            positions=np.stack([qpos_ref[0, wrist_idx, :3], qpos_ref[0, tidx, :3]]),
            color=(255, 200, 50),
            line_width=2.5,
        )
        bone_handles.append((h, tidx))

    def update(t):
        wrist_pos = qpos_ref[t, wrist_idx, :3]
        wrist_h.position = tuple(wrist_pos)
        for k, tidx in enumerate(tip_idxs):
            tip_handles[k].position = tuple(qpos_ref[t, tidx, :3])
        for h, tidx in bone_handles:
            h.positions = np.stack([wrist_pos, qpos_ref[t, tidx, :3]])

    return update


# Cost of 1 m of hand-object penetration, against finger_pos_cost (10 by default)
# per meter of fingertip error: a finger 1 mm inside an object costs as much as
# 1 cm of fingertip error.
PENETRATION_COST = 100.0


class PenetrationTask(mink.Task):
    """Soft push-out for hand geoms that end up inside an object.

    The object follows its reference and can step into the hand between frames,
    which mink's CollisionAvoidanceLimit cannot undo: it only bounds approach.
    For each pair closer than `clearance`, the task error is dist - clearance and
    its Jacobian the rate of change of dist, so the solver trades the push-out
    against the fingertip targets instead of failing.
    """

    def __init__(self, limit, clearance, cost=PENETRATION_COST):
        super().__init__(cost=np.full(len(limit.geom_id_pairs), cost), gain=1.0, lm_damping=1e-3)
        self.limit, self.clearance = limit, clearance

    def _rows(self, configuration):
        from mink.limits.collision_avoidance_limit import compute_contact_normal_jacobian

        lim, model, data = self.limit, configuration.model, configuration.data
        error = np.zeros(len(lim.geom_id_pairs))
        jac = np.zeros((len(lim.geom_id_pairs), model.nv))
        for i, (g1, g2) in enumerate(lim.geom_id_pairs):
            dist = mujoco.mj_geomDistance(model, data, g1, g2, self.clearance + 0.01, lim._fromto)
            if dist >= self.clearance:
                continue
            row = compute_contact_normal_jacobian(
                model, data, g1, g2, lim._fromto, lim._normal, lim._jac1, lim._jac2
            )
            error[i] = dist - self.clearance
            jac[i] = row if dist >= 0 else -row
        return error, jac

    def compute_error(self, configuration):
        """Task error from _rows."""
        return self._rows(configuration)[0]

    def compute_jacobian(self, configuration):
        """Task Jacobian from _rows."""
        return self._rows(configuration)[1]


def fingertip_geom(model, site_name):
    """The one hand collision geom on the body of a fingertip site.

    Every SPIDER hand puts a single collision_hand_* geom on the fingertip-site
    body; ik_fast offsets the contact target by its radius.
    """
    body = model.site_bodyid[model.site(site_name).id]
    geoms = [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == body and model.geom(g).name.startswith("collision_hand_")
    ]
    if len(geoms) != 1:
        raise ValueError(
            f"{site_name}: expected one collision_hand_* geom on body "
            f"{model.body(body).name}, found {len(geoms)}"
        )
    return geoms[0]


# How far a fingertip site may sit from the center of its tip geom's end cap.
# The Inspire and MetaHand sites sit within 0.01 mm of it; the XHand index site
# sits 5 mm off, which would leave its contact target 5 mm wrong.
TIP_SITE_TOLERANCE_M = 1e-3


def check_tip_site(model, site_name):
    """Radius of the fingertip geom; raise if the site is not at its tip end.

    The contact target puts the site at contact_point + radius * normal, which
    puts the geom surface on the contact point only when the site is the
    center of the geom's end cap: the center of a sphere, or one end of a
    capsule's axis.
    """
    site = model.site(site_name).id
    geom = fingertip_geom(model, site_name)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
    # Site and geom share a body, so their body-frame poses compare directly.
    local = rotation.reshape(3, 3).T @ (model.site_pos[site] - model.geom_pos[geom])
    kind = model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        offset = np.linalg.norm(local)
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        half = model.geom_size[geom, 1]
        offset = min(np.linalg.norm(local - [0, 0, z]) for z in (-half, half))
    else:
        raise ValueError(
            f"{site_name}: tip geom {model.geom(geom).name} is "
            f"{mujoco.mjtGeom(kind).name}, expected a sphere or capsule"
        )
    if offset > TIP_SITE_TOLERANCE_M:
        raise ValueError(
            f"site {site_name} is {offset * 1000:.2f} mm from the tip end of geom "
            f"{model.geom(geom).name}; tolerance {TIP_SITE_TOLERANCE_M * 1000:.1f} mm"
        )
    return float(model.geom_size[geom, 0])


def contact_blend_weights(contact: np.ndarray, ramp_frames: int) -> np.ndarray:
    """Per-frame weight (N, K) of the contact target against the human fingertip.

    Inside a contact span the weight rises in equal steps from 0 on the frame
    before the span to 1 on its frame ramp_frames, and falls the same way to 0
    on the frame after it, so a span of 2 * ramp_frames frames or fewer peaks
    below 1. Outside spans it is 0.
    """
    weights = np.zeros(contact.shape)
    for k in range(contact.shape[1]):
        edges = np.flatnonzero(np.diff(np.r_[0, contact[:, k] > 0, 0]))
        for start, end in zip(edges[::2], edges[1::2], strict=True):
            j = np.arange(end - start)
            ramp = np.minimum(j + 1, end - start - j) / (ramp_frames + 1)
            weights[start:end, k] = np.minimum(ramp, 1.0)
    return weights


def contact_targets(
    contact: np.ndarray,
    local_points: np.ndarray,
    local_normals: np.ndarray,
    radius: float,
    object_pose: np.ndarray,
) -> np.ndarray:
    """World-frame fingertip-site targets (N, 3) of one finger in contact.

    local_points and local_normals (N, 3) hold the finger's object-surface span
    point and outward normal in the frame of the object it touches, and
    object_pose (N, 7) that object's pose as [x, y, z, qw, qx, qy, qz]. The
    target is the point moved out along the normal by the tip geom radius, so
    the geom surface touches the point. Targets move with the object; frames
    without contact get the object origin, which the masked reward ignores.
    """
    targets = np.zeros(local_points.shape)
    rotation = np.zeros(9)
    for t in range(len(object_pose)):
        mujoco.mju_quat2Mat(rotation, object_pose[t, 3:])
        local = local_points[t] + radius * local_normals[t]
        world = rotation.reshape(3, 3) @ local + object_pose[t, :3]
        targets[t] = world if contact[t] > 0 else object_pose[t, :3]
    return targets


def main(
    dataset_dir: str = f"{ROOT}/../example_datasets",
    dataset_name: str = "oakink",
    robot_type: str = "xhand",
    embodiment_type: str = "bimanual",
    task: str = "pick_spoon_bowl",
    show_viewer: bool = True,
    show_viser_viewer: bool = False,
    save_video: bool = True,
    save_viser: bool = False,
    mano_assets_root: str | None = None,
    data_id: int = 0,
    start_idx: int = 0,
    end_idx: int = -1,
    sim_dt: float = 0.005,
    ref_dt: float | None = None,
    wrist_pos_cost: float = 0.3,
    wrist_ori_cost: float = 3.0,
    finger_pos_cost: float = 10.0,
    posture_cost: float = 1e-2,
    wrist_init_steps: int = 200,
    finger_init_steps: int = 300,
    average_frame_size: int = 3,
    z_offset: float = 0.0,
    object_clearance: float | None = None,
    contact_ramp_s: float = 0.1,
):
    # Resolve directories
    dataset_dir = os.path.abspath(dataset_dir)
    processed_dir_robot = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type=robot_type,
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    processed_dir_mano = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type="mano",
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    os.makedirs(processed_dir_robot, exist_ok=True)
    model_path = f"{processed_dir_robot}/../scene.xml"

    # Resolve ref_dt: prefer the per-task value the dataset processor wrote
    # to task_info.json (mocap rate varies per dataset — Arctic 30 Hz,
    # OakInk2 50 Hz, etc.) so IK plays back at the recorded rate. Fall back
    # to 50 Hz / 0.02 s only if no task_info is present.
    task_info_path = Path(processed_dir_mano).parent / "task_info.json"
    if ref_dt is None:
        if task_info_path.exists():
            with open(task_info_path) as f:
                ref_dt = float(json.load(f).get("ref_dt", 0.02))
            loguru.logger.info(
                f"ref_dt={ref_dt:.4f}s ({1.0/ref_dt:.1f}Hz) loaded from {task_info_path}"
            )
        else:
            ref_dt = 0.02
            loguru.logger.warning(
                f"No task_info.json at {task_info_path}; defaulting ref_dt=0.02s"
            )

    # Load reference keypoints
    file_path = f"{processed_dir_mano}/trajectory_keypoints.npz"
    loaded_data = np.load(file_path)
    qpos_finger_right = loaded_data["qpos_finger_right"][start_idx:end_idx]
    qpos_finger_left = loaded_data["qpos_finger_left"][start_idx:end_idx]
    qpos_wrist_right = loaded_data["qpos_wrist_right"][start_idx:end_idx]
    qpos_wrist_left = loaded_data["qpos_wrist_left"][start_idx:end_idx]
    qpos_obj_right = loaded_data["qpos_obj_right"][start_idx:end_idx]
    qpos_obj_left = loaded_data["qpos_obj_left"][start_idx:end_idx]

    # Build reference array: (H, num_sites, 7) where 7 = [x, y, z, qw, qx, qy, qz]
    qpos_ref = np.concatenate(
        [
            qpos_wrist_right[:, None],
            qpos_finger_right,
            qpos_wrist_left[:, None],
            qpos_finger_left,
            qpos_obj_right[:, None],
            qpos_obj_left[:, None],
        ],
        axis=1,
    )
    qpos_ref[:, :, 2] += z_offset

    num_frames = qpos_finger_right.shape[0]

    # Reference index mapping:
    # 0: right_palm, 1-5: right fingers, 6: left_palm, 7-11: left fingers,
    # 12: right_object, 13: left_object
    ref_idx = {}
    ref_idx["right_palm"] = 0
    for i, name in enumerate(
        ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    ):
        ref_idx[f"right_{name}"] = i + 1
    ref_idx["left_palm"] = 6
    for i, name in enumerate(
        ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    ):
        ref_idx[f"left_{name}"] = i + 7
    ref_idx["right_object"] = 12
    ref_idx["left_object"] = 13

    # -- Viser setup (optional) --
    viser_state = {"enabled": False}
    if show_viser_viewer or save_viser:
        try:
            from spider.viewers.viser_viewer import (
                _STATE as _VISER_STATE,
                build_and_log_scene,
                init_viser,
                log_frame,
            )

            init_viser(app_name=f"spider-ik-{task}")
            spec, viser_model, viewer_body_entity_and_ids = build_and_log_scene(
                Path(model_path)
            )
            del spec, viser_model
            viser_state.update(
                {
                    "enabled": True,
                    "log_frame": log_frame,
                    "body_ids": viewer_body_entity_and_ids,
                    "state": _VISER_STATE,
                }
            )
            loguru.logger.info(
                "Viser scene initialized (will record IK rollout for inspection)."
            )
        except Exception as e:
            loguru.logger.warning(f"Failed to init viser: {e}")

    # Optional reference MANO skeleton (only if MANO is loaded correctly)
    ref_skeleton = None
    if viser_state["enabled"] and mano_assets_root is not None:
        ref_skeleton = _build_ref_skeleton(
            viser_state["state"].server,
            mano_assets_root,
            qpos_ref,
            ref_idx,
            embodiment_type,
        )

    # Object DOF count
    if embodiment_type == "bimanual":
        nq_obj = 14
    elif embodiment_type in ["right", "left"]:
        nq_obj = 7
    else:
        nq_obj = 0

    # Build site name lists
    finger_names = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    if robot_type in ["allegro", "metahand"]:
        finger_names = finger_names[:4]

    wrist_sites = []
    finger_sites = []
    if embodiment_type in ["right", "bimanual"]:
        wrist_sites.append("right_palm")
        finger_sites.extend([f"right_{f}" for f in finger_names])
    if embodiment_type in ["left", "bimanual"]:
        wrist_sites.append("left_palm")
        finger_sites.extend([f"left_{f}" for f in finger_names])

    # Load model
    model = mujoco.MjModel.from_xml_path(model_path)
    # The IK model is a separate copy: the collision limit needs
    # contype/conaffinity set, which the rollout model must not have (SPIDER
    # scenes enable contact only through explicit pairs).
    ik_model = mujoco.MjModel.from_xml_path(model_path)

    # -- Contact targets, one column per tracked fingertip site --
    # Per-frame contact flags (N, K) of the kept spans from detect_contact.py,
    # the world fingertip-site targets on the object surface (N, K, 3), and the
    # blend weight of those targets against the human fingertip (N, K).
    contact = targets = weights = None
    sides = [s.split("_")[0] for s in finger_sites]
    if all(f"contact_normal_{s}" in loaded_data for s in set(sides)):
        # The ramp spreads the step between the human fingertip and the
        # contact target over several frames. On recordings 1, 7 and 8 of
        # 2026-10-09 that step is 17 mm at p50 and 35 mm at p90 at span ends,
        # and the human fingertip moves 5.9 mm per 50 Hz frame at p90; 0.1 s
        # (5 frames, weight steps of 1/6) keeps the p90 target step at 5.9 mm.
        ramp_frames = max(1, round(contact_ramp_s / ref_dt))
        columns, target_columns = [], []
        for site_name, side in zip(finger_sites, sides, strict=True):
            slot = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"].index(
                site_name[len(side) + 1 :]
            )
            obj = str(loaded_data[f"contact_object_{side}"])
            column = loaded_data[f"contact_{side}"][start_idx:end_idx, slot]
            columns.append(column)
            target_columns.append(
                contact_targets(
                    column,
                    loaded_data[f"contact_point_{side}"][start_idx:end_idx, slot],
                    loaded_data[f"contact_normal_{side}"][start_idx:end_idx, slot],
                    check_tip_site(model, site_name),
                    qpos_ref[:, ref_idx[obj]],
                )
            )
        contact = np.stack(columns, axis=1)
        targets = np.stack(target_columns, axis=1)
        weights = contact_blend_weights(contact, ramp_frames)
        loguru.logger.info(
            f"Contact targets blend in over {ramp_frames} frames for {finger_sites}"
        )

    # Create mink configuration.
    configuration = mink.Configuration(ik_model)
    data = configuration.data
    limits = [mink.ConfigurationLimit(ik_model)]
    constraints = None
    if object_clearance is not None:

        def geom_names(prefix):
            return [
                g
                for g in range(ik_model.ngeom)
                if (mujoco.mj_id2name(ik_model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(prefix)
            ]

        hand_geoms = geom_names("collision_hand_")
        object_geoms = [
            g
            for g in geom_names("right_object_") + geom_names("left_object_") + geom_names("fixture_")
            if ik_model.geom_group[g] == 3 or "collision" in ik_model.geom(g).name
        ]
        ik_model.geom_contype[hand_geoms + object_geoms] = 1
        ik_model.geom_conaffinity[hand_geoms + object_geoms] = 1
        limits.append(
            mink.CollisionAvoidanceLimit(
                ik_model,
                [(hand_geoms, object_geoms)],
                minimum_distance_from_collisions=object_clearance,
                collision_detection_distance=object_clearance + 0.02,
            )
        )
        # Objects are set from the reference every substep; without this the
        # solver would move an object instead of the hand to clear a collision.
        object_dofs = [
            ik_model.jnt_dofadr[j] + k
            for j in range(ik_model.njnt)
            if ik_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            for k in range(6)
        ]
        if object_dofs:
            constraints = [mink.DofFreezingTask(ik_model, object_dofs)]
        loguru.logger.info(
            f"IK keeps {len(hand_geoms)} hand geoms {object_clearance} m from "
            f"{len(object_geoms)} object and fixture geoms"
        )

    extra_tasks = []
    if object_clearance is not None:
        extra_tasks = [PenetrationTask(limits[-1], object_clearance)]

    qpos_list, unlimited_steps = [], []

    def solve(tasks):
        try:
            return mink.solve_ik(
                configuration, [*tasks, *extra_tasks], sim_dt, solver, damping=1e-5,
                limits=limits, constraints=constraints,
            )
        except mink.exceptions.NoSolutionFound:
            # A finger wedged between object parts gets push-out bounds from
            # opposite sides that no velocity meets; for this step only
            # PenetrationTask moves it out.
            unlimited_steps.append(len(qpos_list))
            return mink.solve_ik(
                configuration, [*tasks, *extra_tasks], sim_dt, solver, damping=1e-5,
                limits=limits[:1], constraints=constraints,
            )

    # Create IK tasks
    # Cost priority: finger_pos > wrist_pos > wrist_ori
    wrist_tasks = [
        mink.FrameTask(
            frame_name=s,
            frame_type="site",
            position_cost=wrist_pos_cost,
            orientation_cost=wrist_ori_cost,
            lm_damping=1.0,
        )
        for s in wrist_sites
    ]

    finger_tasks = [
        mink.FrameTask(
            frame_name=s,
            frame_type="site",
            position_cost=finger_pos_cost,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        for s in finger_sites
    ]

    posture_task = mink.PostureTask(ik_model, cost=posture_cost)
    posture_task.set_target(configuration.q.copy())

    tasks_wrist = [posture_task, *wrist_tasks]
    tasks_all = [posture_task, *wrist_tasks, *finger_tasks]

    solver = "daqp"
    n_substeps = max(1, int(round(ref_dt / sim_dt)))

    # -- Helpers --
    def set_object_qpos(t):
        if embodiment_type == "bimanual":
            data.qpos[-14:-7] = qpos_ref[t, ref_idx["right_object"]]
            data.qpos[-7:] = qpos_ref[t, ref_idx["left_object"]]
        elif embodiment_type == "right":
            data.qpos[-7:] = qpos_ref[t, ref_idx["right_object"]]
        elif embodiment_type == "left":
            data.qpos[-7:] = qpos_ref[t, ref_idx["left_object"]]

    def set_wrist_targets(t):
        for wrist_task, site_name in zip(wrist_tasks, wrist_sites, strict=True):
            pos = qpos_ref[t, ref_idx[site_name], :3]
            quat_wxyz = qpos_ref[t, ref_idx[site_name], 3:]
            wrist_task.set_target(mink.SE3(wxyz_xyz=np.concatenate([quat_wxyz, pos])))

    def set_finger_targets(t):
        # A finger in contact moves from the human fingertip, whose position
        # includes the human finger's thickness, to its contact target.
        for k, (finger_task, site_name) in enumerate(zip(finger_tasks, finger_sites, strict=True)):
            pos = qpos_ref[t, ref_idx[site_name], :3]
            if weights is not None:
                pos = (1.0 - weights[t, k]) * pos + weights[t, k] * targets[t, k]
            finger_task.set_target(
                mink.SE3(wxyz_xyz=np.array([1.0, 0.0, 0.0, 0.0, *pos]))
            )

    # -- Phase 1: Initialize wrist (frame 0) --
    loguru.logger.info("Phase 1: Initializing wrist position...")
    set_object_qpos(0)
    configuration.update()
    set_wrist_targets(0)
    for _ in range(wrist_init_steps):
        vel = solve(tasks_wrist)
        configuration.integrate_inplace(vel, sim_dt)
        set_object_qpos(0)

    # -- Phase 2: Add finger tips (frame 0) --
    loguru.logger.info("Phase 2: Initializing finger positions...")
    configuration.update()
    set_finger_targets(0)
    for _ in range(finger_init_steps):
        vel = solve(tasks_all)
        configuration.integrate_inplace(vel, sim_dt)
        set_object_qpos(0)

    # -- Main IK loop --
    loguru.logger.info(f"Running IK for {num_frames} frames...")
    images = []

    if save_video:
        import imageio

        model.vis.global_.offwidth = 720
        model.vis.global_.offheight = 480
        renderer = mujoco.Renderer(model, height=480, width=720)

    # Render and view through the scene model; data belongs to ik_model.
    view_data = mujoco.MjData(model)

    def sync_view():
        view_data.qpos[:] = configuration.q
        mujoco.mj_forward(model, view_data)

    run_viewer = get_viewer(show_viewer, model, view_data)
    rate_limiter = RateLimiter(1 / ref_dt)

    with run_viewer() as gui:
        for t in range(num_frames):
            set_wrist_targets(t)
            set_finger_targets(t)
            set_object_qpos(t)
            configuration.update()

            for _ in range(n_substeps):
                vel = solve(tasks_all)
                configuration.integrate_inplace(vel, sim_dt)
                set_object_qpos(t)

            qpos_list.append(configuration.q.copy())

            if save_video:
                sync_view()
                renderer.update_scene(data=view_data, camera="front")
                images.append(renderer.render())

            if viser_state["enabled"]:
                sync_view()
                viser_state["log_frame"](
                    view_data,
                    sim_time=t * ref_dt,
                    viewer_body_entity_and_ids=viser_state["body_ids"],
                    show_ui=True,
                    playback_fps=1.0 / ref_dt,
                )
                if ref_skeleton is not None:
                    ref_skeleton(t)

            if show_viewer:
                sync_view()
                gui.sync()
                rate_limiter.sleep()

            if t % 100 == 0:
                loguru.logger.info(f"  Frame {t}/{num_frames}")

    qpos_list = np.array(qpos_list)
    if unlimited_steps:
        loguru.logger.warning(
            f"collision limit infeasible on {len(unlimited_steps)} IK steps in "
            f"{len(set(unlimited_steps))} frames; those steps ran without it"
        )

    # -- Post-processing: moving average filter --
    def moving_average_filter(signal_data, window_size=5):
        return np.convolve(
            signal_data, np.ones(window_size) / window_size, mode="valid"
        )

    def filter_columns(values):
        flat = values.reshape(len(values), -1)
        out = np.zeros((len(flat) - average_frame_size + 1, flat.shape[1]))
        for i in range(flat.shape[1]):
            out[:, i] = moving_average_filter(flat[:, i], average_frame_size)
        return out.reshape(len(out), *values.shape[1:])

    qpos_list = filter_columns(qpos_list)

    contact_out = {}
    if contact is not None:
        contact_out = {
            "contact": filter_columns(contact)[1:],
            "contact_pos": filter_columns(targets)[1:],
        }
        site_ids = [model.site(s).id for s in finger_sites]
        robot_task_info = Path(processed_dir_robot).parent / "task_info.json"
        info = json.loads(robot_task_info.read_text()) if robot_task_info.exists() else {}
        info["contact_site_ids"] = site_ids
        robot_task_info.write_text(json.dumps(info, indent=2))
        loguru.logger.info(
            f"Contact targets for {finger_sites}; site ids {site_ids} in {robot_task_info}"
        )

    # -- Compute qvel via finite differences --
    n_filtered = qpos_list.shape[0]
    qvel_list = np.zeros((n_filtered - 1, model.nv))
    for i in range(1, n_filtered):
        mujoco.mj_differentiatePos(
            model, qvel_list[i - 1], ref_dt, qpos_list[i - 1], qpos_list[i]
        )
    qpos_list = qpos_list[1:]
    assert qpos_list.shape[0] == qvel_list.shape[0]

    # -- Forward rollout for validation --
    mj_data_rollout = mujoco.MjData(model)
    n_rollout_substeps = max(1, int(round(ref_dt / sim_dt)))
    model.opt.timestep = ref_dt / n_rollout_substeps
    mj_data_rollout.qpos[:] = qpos_list[0]
    mj_data_rollout.qvel[:] = qvel_list[0]
    mj_data_rollout.ctrl[:] = qpos_list[0][: model.nq - nq_obj]
    for _ in range(n_rollout_substeps):
        mujoco.mj_step(model, mj_data_rollout)
    n_final = qpos_list.shape[0]
    qpos_rollout = np.zeros((n_final, model.nq))
    qpos_rollout[0] = qpos_list[0]
    for i in range(1, n_final):
        mj_data_rollout.ctrl[:] = qpos_list[i][: model.nq - nq_obj]
        noise = np.random.randn(model.nu) * 0.2
        noise[:6] *= 0.0
        noise[22:28] *= 0.0
        mj_data_rollout.ctrl[:] += noise
        for _ in range(n_rollout_substeps):
            mujoco.mj_step(model, mj_data_rollout)
        qpos_rollout[i] = mj_data_rollout.qpos.copy()

    # -- Save outputs --
    file_dir = processed_dir_robot
    if save_video:
        import imageio

        video_path = f"{file_dir}/visualization_ik.mp4"
        imageio.mimsave(video_path, images, fps=int(1 / ref_dt))
        loguru.logger.info(f"Saved video to {video_path}")

    out_npz = f"{file_dir}/trajectory_kinematic.npz"
    np.savez(
        out_npz, qpos=qpos_list, qvel=qvel_list, frequency=1 / ref_dt, **contact_out
    )
    loguru.logger.info(f"Saved {out_npz}")

    out_npz = f"{file_dir}/trajectory_ikrollout.npz"
    np.savez(out_npz, qpos=qpos_rollout)
    loguru.logger.info(f"Saved {out_npz}")

    # -- Save viser export for offline inspection --
    if save_viser and viser_state["enabled"]:
        viser_path = f"{file_dir}/visualization_ik.viser"
        try:
            server = viser_state["state"].server
            data_bytes = server.get_scene_serializer().serialize()
            Path(viser_path).write_bytes(data_bytes)
            loguru.logger.info(f"Saved viser scene to {viser_path}")
        except Exception as e:
            loguru.logger.warning(f"Failed to save viser scene: {e}")

    if show_viser_viewer and viser_state["enabled"]:
        loguru.logger.info(
            "Viser viewer running. Use timeline slider in browser. Ctrl-C to exit."
        )
        import time

        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    tyro.cli(main)
