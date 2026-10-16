# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic optimization variant.

Select best trajectory on final iteration, record full simulation state,
and retry if tracking error is too large.

Author: Chaoyi Pan
Date: 2025-04-16
"""

from __future__ import annotations

import loguru
import numpy as np
import torch
import warp as wp

from spider.config import Config
from spider.math import quat_sub
from spider.simulators.mjwp import copy_from_torch
from spider.optimizers.sampling import (
    _compute_weights_compiled,
    _compute_weights_impl,
    sample_ctrls,
)

_MAX_ATTEMPTS = 4


def _snapshot_env_state(env) -> dict[str, torch.Tensor]:
    """Clone all core state fields for all N samples into a dict of tensors."""
    snap: dict[str, torch.Tensor] = {}

    core_fields = [
        "qpos",
        "qvel",
        "qacc",
        "time",
        "ctrl",
        "act",
        "act_dot",
        "qacc_warmstart",
        "qfrc_applied",
        "xfrc_applied",
        "energy",
        "mocap_pos",
        "mocap_quat",
        "xpos",
        "xquat",
        "xmat",
        "xipos",
        "ximat",
        "geom_xpos",
        "geom_xmat",
        "site_xpos",
        "site_xmat",
        "cacc",
        "cfrc_int",
        "cfrc_ext",
        "sensordata",
        "actuator_length",
        "actuator_velocity",
        "actuator_force",
        "ten_length",
        "ten_velocity",
    ]
    for name in core_fields:
        if hasattr(env.data_wp, name):
            snap[name] = wp.to_torch(getattr(env.data_wp, name)).clone()

    contact_fields = [
        "dist",
        "pos",
        "frame",
        "includemargin",
        "friction",
        "solref",
        "solreffriction",
        "solimp",
        "dim",
        "geom",
        "efc_address",
        "worldid",
    ]
    for name in contact_fields:
        if hasattr(env.data_wp.contact, name):
            snap[f"contact.{name}"] = wp.to_torch(
                getattr(env.data_wp.contact, name)
            ).clone()

    efc_fields = [
        "type",
        "id",
        "J",
        "pos",
        "margin",
        "D",
        "vel",
        "aref",
        "frictionloss",
        "force",
    ]
    for name in efc_fields:
        if hasattr(env.data_wp.efc, name):
            snap[f"efc.{name}"] = wp.to_torch(getattr(env.data_wp.efc, name)).clone()

    return snap


def _restore_best_state_to_env(env, snapshot: dict[str, torch.Tensor], best_idx: int):
    """Restore best sample's full state from snapshot to ALL worlds of env.data_wp."""
    nw = env.num_worlds
    for key, tensor in snapshot.items():
        best_val = tensor[best_idx : best_idx + 1]  # (1, ...)
        ndim = best_val.dim()
        reps = [nw] + [1] * (ndim - 1)
        expanded = best_val.repeat(*reps)  # (nw, ...)

        parts = key.split(".")
        if len(parts) == 1:
            copy_from_torch(getattr(env.data_wp, parts[0]), expanded)
        elif len(parts) == 2:
            parent = getattr(env.data_wp, parts[0])
            copy_from_torch(getattr(parent, parts[1]), expanded)


def check_tracking_error(
    config: Config,
    recorded_qpos: torch.Tensor,
    ref_qpos: torch.Tensor,
) -> tuple[bool, dict]:
    """Check if object tracking error exceeds thresholds within ctrl_steps.

    Args:
        config: Config with embodiment_type, nq_obj, and threshold fields.
        recorded_qpos: (ctrl_steps, nq) best trajectory qpos.
        ref_qpos: (ctrl_steps, nq) reference qpos.

    Returns:
        exceeded: True if max error exceeds threshold.
        error_info: dict with max_pos_error, max_rot_error.
    """
    if config.embodiment_type == "bimanual":
        if config.nq_obj == 12:
            r_pos = recorded_qpos[:, -12:-9] - ref_qpos[:, -12:-9]
            r_rot = recorded_qpos[:, -9:-6] - ref_qpos[:, -9:-6]
            l_pos = recorded_qpos[:, -6:-3] - ref_qpos[:, -6:-3]
            l_rot = recorded_qpos[:, -3:] - ref_qpos[:, -3:]

            r_pos_err = torch.norm(r_pos, p=2, dim=1)
            r_rot_err = torch.norm(r_rot, p=2, dim=1)
            l_pos_err = torch.norm(l_pos, p=2, dim=1)
            l_rot_err = torch.norm(l_rot, p=2, dim=1)

            if torch.all(ref_qpos[:, -12:-9].abs() < 1e-4):
                r_pos_err = r_pos_err * 0.0
                r_rot_err = r_rot_err * 0.0
            if torch.all(ref_qpos[:, -6:-3].abs() < 1e-4):
                l_pos_err = l_pos_err * 0.0
                l_rot_err = l_rot_err * 0.0

            max_pos = max(r_pos_err.max().item(), l_pos_err.max().item())
            max_rot = max(r_rot_err.max().item(), l_rot_err.max().item())
        else:
            l_pos_err = torch.norm(
                recorded_qpos[:, -14:-11] - ref_qpos[:, -14:-11], p=2, dim=1
            )
            l_quat_err = torch.norm(
                quat_sub(recorded_qpos[:, -11:-7], ref_qpos[:, -11:-7]), p=2, dim=1
            )
            r_pos_err = torch.norm(
                recorded_qpos[:, -7:-4] - ref_qpos[:, -7:-4], p=2, dim=1
            )
            r_quat_err = torch.norm(
                quat_sub(recorded_qpos[:, -4:], ref_qpos[:, -4:]), p=2, dim=1
            )

            if torch.all(ref_qpos[:, -7:-4].abs() < 1e-4):
                r_pos_err = r_pos_err * 0.0
                r_quat_err = r_quat_err * 0.0
            if torch.all(ref_qpos[:, -14:-11].abs() < 1e-4):
                l_pos_err = l_pos_err * 0.0
                l_quat_err = l_quat_err * 0.0

            max_pos = max(r_pos_err.max().item(), l_pos_err.max().item())
            max_rot = max(r_quat_err.max().item(), l_quat_err.max().item())

    elif config.embodiment_type in ["right", "left"]:
        if config.nq_obj == 6:
            pos_err = torch.norm(
                recorded_qpos[:, -6:-3] - ref_qpos[:, -6:-3], p=2, dim=1
            )
            rot_err = torch.norm(recorded_qpos[:, -3:] - ref_qpos[:, -3:], p=2, dim=1)
        else:
            pos_err = torch.norm(
                recorded_qpos[:, -7:-4] - ref_qpos[:, -7:-4], p=2, dim=1
            )
            rot_err = torch.norm(
                quat_sub(recorded_qpos[:, -4:], ref_qpos[:, -4:]), p=2, dim=1
            )
        max_pos = pos_err.max().item()
        max_rot = rot_err.max().item()

    elif config.embodiment_type in ["humanoid", "humanoid_object"]:
        pos_err = torch.norm(recorded_qpos[:, :3] - ref_qpos[:, :3], p=2, dim=1)
        rot_err = torch.norm(
            quat_sub(recorded_qpos[:, 3:7], ref_qpos[:, 3:7]), p=2, dim=1
        )
        max_pos = pos_err.max().item()
        max_rot = rot_err.max().item()
    else:
        raise ValueError(f"Invalid embodiment_type: {config.embodiment_type}")

    exceeded = (
        max_pos > config.object_pos_threshold or max_rot > config.object_rot_threshold
    )
    return exceeded, {"max_pos_error": max_pos, "max_rot_error": max_rot}


def make_rollout_fn_fast(  # noqa: D103
    step_env,
    save_state,
    load_state,
    get_reward,
    get_terminal_reward,
    get_terminate,
    get_trace,
    save_env_params,
    load_env_params,
    copy_sample_state,
    get_qpos,
    get_qvel,
):
    def rollout(
        config: Config,
        env,
        ctrls: torch.Tensor,
        ref_slice: tuple[torch.Tensor, ...],
        env_param: dict,
        record_states: bool = False,
    ) -> torch.Tensor:
        """Rollout with optional state recording for deterministic replay.

        When record_states=True, records qpos/qvel at each step within ctrl_steps
        and takes a full env state snapshot at t == ctrl_steps - 1.
        """
        init_state = save_state(env)
        init_env_param = save_env_params(config, env)
        env = load_env_params(config, env, env_param)

        n_samples, n_horizon = ctrls.shape[:2]
        trace_list = []
        cum_rew = torch.zeros(n_samples, device=config.device)
        info_list = []

        if record_states:
            recorded_qpos = torch.zeros(
                (n_samples, config.ctrl_steps, config.nq), device=config.device
            )
            recorded_qvel = torch.zeros(
                (n_samples, config.ctrl_steps, config.nv), device=config.device
            )
            state_snapshot = None

        for t in range(n_horizon):
            step_env(config, env, ctrls[:, t])

            ref = [r[t] for r in ref_slice]
            rew, info = (
                get_reward(config, env, ref)
                if t < n_horizon - 1
                else get_terminal_reward(config, env, ref)
            )
            cum_rew += rew

            trace = get_trace(config, env)
            trace_list.append(trace)
            info_list.append(info)

            terminate = get_terminate(config, env, ref)
            if (
                config.terminate_resample
                and t < n_horizon - 1
                and terminate.any()
                and (not terminate.all())
            ):
                bad_indices = torch.nonzero(terminate).squeeze(-1)
                good_indices = torch.nonzero(~terminate).squeeze(-1)
                if good_indices.shape[0] > bad_indices.shape[0]:
                    good_indices = good_indices[: bad_indices.shape[0]]
                elif good_indices.shape[0] < bad_indices.shape[0]:
                    random_idx = torch.randint(
                        0, good_indices.shape[0], (bad_indices.shape[0],)
                    )
                    good_indices = good_indices[random_idx]

                copy_sample_state(config, env, good_indices, bad_indices)
                ctrls[bad_indices, :t] = ctrls[good_indices, :t]
                cum_rew[bad_indices] = cum_rew[good_indices]

            if record_states and t < config.ctrl_steps:
                recorded_qpos[:, t] = get_qpos(config, env)
                recorded_qvel[:, t] = get_qvel(config, env)
                if t == config.ctrl_steps - 1:
                    state_snapshot = _snapshot_env_state(env)

        info_combined = {
            k: torch.stack([info[k] for info in info_list], axis=0)
            for k in info_list[0]
        }
        mean_info = {k: v.mean(axis=0) for k, v in info_combined.items()}
        mean_rew = cum_rew / n_horizon

        env = load_state(env, init_state)
        env = load_env_params(config, env, init_env_param)

        trace_list = torch.stack(trace_list, dim=1)
        info = {
            "trace": trace_list,
            **mean_info,
        }

        if record_states:
            info["recorded_qpos"] = recorded_qpos
            info["recorded_qvel"] = recorded_qvel
            info["state_snapshot"] = state_snapshot

        return ctrls, mean_rew, terminate, info

    return rollout


def make_optimize_once_fn_fast(rollout):  # noqa: D103
    def optimize_once(
        config: Config,
        env,
        ctrls: torch.Tensor,
        ref_slice: tuple[torch.Tensor, ...],
        env_params: list[dict] | None = None,
        sample_params: dict | None = None,
        select_best: bool = False,
    ) -> torch.Tensor:
        """Single optimization step.

        When select_best=True, returns the single best trajectory instead of
        the softmax-weighted mean, along with recorded states.
        """
        if env_params is None:
            env_params = [{}]
        ctrls_samples = sample_ctrls(config, ctrls, sample_params)

        min_rew = torch.full((config.num_samples,), float("inf"), device=config.device)
        for env_param in env_params:
            ctrls_samples, rews, terminate, rollout_info = rollout(
                config,
                env,
                ctrls_samples,
                ref_slice,
                env_param,
                record_states=select_best,
            )
            min_rew = torch.minimum(min_rew, rews)
        rews = min_rew

        if config.use_torch_compile:
            weights, nan_mask = _compute_weights_compiled(
                rews, config.num_samples, config.temperature
            )
        else:
            weights, nan_mask = _compute_weights_impl(
                rews, config.num_samples, config.temperature
            )

        if nan_mask.any():
            loguru.logger.warning(
                f"NaNs or infs in rews: {nan_mask.sum()}/{config.num_samples}"
            )

        if select_best:
            best_idx = torch.argmax(rews).item()
            ctrls_out = ctrls_samples[best_idx]
        else:
            ctrls_out = (weights[:, None, None] * ctrls_samples).sum(dim=0)

        # trace downsampling for visualization
        n_uni = max(0, min(config.num_trace_uniform_samples, config.num_samples))
        n_topk = max(0, min(config.num_trace_topk_samples, config.num_samples))
        idx_uni = (
            torch.linspace(
                0,
                config.num_samples - 1,
                steps=n_uni,
                dtype=torch.long,
                device=config.device,
            )
            if n_uni > 0
            else torch.tensor([], dtype=torch.long, device=config.device)
        )
        idx_top = (
            torch.topk(rews, k=n_topk, largest=True).indices
            if n_topk > 0
            else torch.tensor([], dtype=torch.long, device=config.device)
        )
        sel_idx = torch.cat([idx_uni, idx_top], dim=0).long()

        info = {}
        for k, v in rollout_info.items():
            if k not in [
                "trace",
                "trace_sample",
                "recorded_qpos",
                "recorded_qvel",
                "state_snapshot",
            ]:
                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()
                if v.ndim == 1:
                    info[k + "_max"] = v.max()
                    info[k + "_min"] = v.min()
                    info[k + "_median"] = np.median(v)
                    info[k + "_mean"] = v.mean()

        rews_np = rews.cpu().numpy()
        info["improvement"] = rews_np.max() - rews_np[0]
        info["rew_max"] = rews_np.max()
        info["rew_min"] = rews_np.min()
        info["rew_median"] = np.median(rews_np)
        info["rew_mean"] = rews_np.mean()

        if "trace" in rollout_info:
            info["trace_sample"] = rollout_info["trace"][sel_idx].cpu().numpy()
            info["trace_cost"] = -rews[sel_idx].cpu().numpy()

        if select_best:
            info["recorded_qpos"] = rollout_info["recorded_qpos"][best_idx]
            info["recorded_qvel"] = rollout_info["recorded_qvel"][best_idx]
            info["state_snapshot"] = rollout_info["state_snapshot"]
            info["best_idx"] = best_idx

        return ctrls_out, terminate, info

    return optimize_once


def make_optimize_fn_fast(optimize_once):  # noqa: D103
    def optimize(
        config: Config,
        env,
        ctrls: torch.Tensor,
        ref_slice: tuple[torch.Tensor, ...],
    ):
        """Deterministic optimization with best-trajectory selection and retry."""
        num_iters = config.max_num_iterations
        sample_params_list = [
            {"global_noise_scale": config.beta_traj**i} for i in range(num_iters)
        ]

        all_infos = []
        succeeded = False
        attempt = 0
        error_info = {"max_pos_error": float("inf"), "max_rot_error": float("inf")}
        last_info = None

        while attempt < _MAX_ATTEMPTS and not succeeded:
            for i in range(num_iters):
                is_last = i == num_iters - 1
                ctrls, terminate, info = optimize_once(
                    config,
                    env,
                    ctrls,
                    ref_slice,
                    config.env_params_list[i],
                    sample_params_list[i],
                    select_best=is_last,
                )
                all_infos.append(info)

            last_info = info
            best_qpos = info["recorded_qpos"]

            # Time alignment: ref_slice[0][i] = qpos_ref[sim_step + 1 + i]
            # (built via get_slice(ref_data, s+1, s+horizon+1) in run_mjwp_fast).
            # recorded_qpos[i] is the state AFTER step_env applied ctrls[i],
            # so it lives at sim time (sim_step + i + 1) * sim_dt — exactly
            # ref index sim_step + 1 + i. So recorded_qpos[i] aligns with
            # ref_slice[0][i], not ref_slice[0][i+1] (the previous code was
            # off by one ref step, exaggerating tracking error by ~1 dt of
            # natural reference motion).
            avail = min(config.ctrl_steps, ref_slice[0].shape[0])
            ref_qpos_slice = ref_slice[0][:avail]
            check_qpos = best_qpos[:avail]

            exceeded, error_info = check_tracking_error(
                config, check_qpos, ref_qpos_slice
            )

            if not exceeded:
                succeeded = True
            else:
                attempt += 1
                if attempt < _MAX_ATTEMPTS:
                    loguru.logger.warning(
                        "Attempt {}/{}: tracking error exceeded (pos={:.4f}, rot={:.4f}). Retrying.",
                        attempt,
                        _MAX_ATTEMPTS,
                        error_info["max_pos_error"],
                        error_info["max_rot_error"],
                    )

        if not succeeded:
            loguru.logger.error(
                "Failed to find feasible solution after {} attempts. pos={:.4f}, rot={:.4f}",
                _MAX_ATTEMPTS,
                error_info["max_pos_error"],
                error_info["max_rot_error"],
            )

        # Aggregate infos: pad to fixed length for consistent array shapes
        total_slots = _MAX_ATTEMPTS * num_iters
        skip_keys = {"recorded_qpos", "recorded_qvel", "state_snapshot", "best_idx"}

        fake_info = {}
        for k in all_infos[0]:
            if k in skip_keys:
                continue
            v = all_infos[0][k]
            if isinstance(v, np.ndarray):
                fake_info[k] = np.zeros_like(v)
            elif isinstance(v, (int, float, np.floating, np.integer)):
                fake_info[k] = type(v)(0)

        padded_infos = list(all_infos)
        for _ in range(total_slots - len(padded_infos)):
            padded_infos.append(fake_info)

        info_aggregated = {}
        for k in fake_info:
            vals = [info[k] for info in padded_infos]
            if isinstance(vals[0], np.ndarray):
                info_aggregated[k] = np.stack(vals, axis=0)
            else:
                info_aggregated[k] = np.array(vals)

        total_opt_steps = len(all_infos)
        info_aggregated["opt_steps"] = np.array([total_opt_steps])

        info_aggregated["best_qpos"] = last_info["recorded_qpos"].cpu().numpy()
        info_aggregated["best_qvel"] = last_info["recorded_qvel"].cpu().numpy()
        info_aggregated["state_snapshot"] = last_info["state_snapshot"]
        info_aggregated["best_idx"] = last_info["best_idx"]
        info_aggregated["succeeded"] = np.array([succeeded])
        info_aggregated["num_attempts"] = np.array(
            [attempt + 1 if succeeded else attempt]
        )

        return ctrls, info_aggregated

    return optimize
