import argparse
import os
import pickle
from typing import Any, Dict, List

import numpy as np


def load_trajectories(path: str) -> List[Dict[str, Any]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of trajectories, got {type(data)}")
    return data


def convert_single_traj(traj: Dict[str, Any]) -> Dict[str, Any]:
    states = np.asarray(traj["states"], dtype=np.float32)
    actions = np.asarray(traj["actions"], dtype=np.float32)
    rewards = np.asarray(traj["rewards"], dtype=np.float32)
    dones = np.asarray(traj["dones"], dtype=bool)

    # 将离散动作视为一维连续动作，形状 [T, 1]
    if actions.ndim == 1:
        actions = actions[:, None]

    converted = {
        "observations": states,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
    }

    # 保留原来的调试字段，训练时不会用到
    for key in [
        "workflow_id",
        "returns_to_go",
        "timesteps",
        "task_ids",
        "is_failure_steps",
        "failure_reasons",
    ]:
        if key in traj:
            converted[key] = traj[key]

    return converted


def split_trajectory(
    traj: Dict[str, Any],
    max_len: int,
    stride: int,
) -> List[Dict[str, Any]]:
    states = np.asarray(traj["states"], dtype=np.float32)
    actions = np.asarray(traj["actions"])
    rewards = np.asarray(traj["rewards"], dtype=np.float32)
    dones = np.asarray(traj["dones"], dtype=bool)
    timesteps = np.asarray(traj.get("timesteps", np.arange(len(states))), dtype=np.int32)

    n = len(states)
    if n <= max_len:
        return [traj]

    out: List[Dict[str, Any]] = []
    base_wf_id = traj.get("workflow_id", "traj")
    stride = max(1, int(stride))

    for start in range(0, n, stride):
        end = min(start + max_len, n)
        if end - start <= 0:
            break
        sub = dict(traj)
        sub["workflow_id"] = f"{base_wf_id}_chunk_{start}_{end}"
        sub["states"] = states[start:end]
        sub["actions"] = actions[start:end]
        sub["rewards"] = rewards[start:end]
        sub_dones = np.asarray(dones[start:end], dtype=bool)
        if len(sub_dones) > 0:
            sub_dones[-1] = True
        sub["dones"] = sub_dones
        sub["timesteps"] = np.arange(end - start, dtype=np.int32)

        if "task_ids" in traj:
            sub["task_ids"] = np.asarray(traj["task_ids"])[start:end]
        if "is_failure_steps" in traj:
            sub["is_failure_steps"] = np.asarray(traj["is_failure_steps"], dtype=bool)[start:end]
        if "failure_reasons" in traj:
            sub["failure_reasons"] = np.asarray(traj["failure_reasons"])[start:end]

        if "returns_to_go" in traj:
            sub_rtg = rewards[start:end][::-1].cumsum()[::-1].astype(np.float32)
            sub["returns_to_go"] = sub_rtg

        out.append(sub)
        if end >= n:
            break

    return out


def convert(input_path: str, output_path: str, max_len: int = 0, stride: int = 0) -> None:
    trajectories = load_trajectories(input_path)

    expanded: List[Dict[str, Any]] = []
    if max_len and max_len > 0:
        use_stride = stride if stride and stride > 0 else max_len
        for tr in trajectories:
            expanded.extend(split_trajectory(tr, max_len=max_len, stride=use_stride))
    else:
        expanded = trajectories

    converted = [convert_single_traj(tr) for tr in expanded]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(converted, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入：pretrain 生成的轨迹 pkl 文件")
    parser.add_argument(
        "--output",
        required=True,
        help="输出：decision_unified 可直接使用的 pkl 文件路径",
    )
    parser.add_argument("--max_len", type=int, default=0, help="可选：将长轨迹切分成多个子轨迹的最大长度")
    parser.add_argument("--stride", type=int, default=0, help="可选：切分子轨迹的步长（默认等于 max_len）")
    args = parser.parse_args()

    convert(args.input, args.output, max_len=args.max_len, stride=args.stride)


if __name__ == "__main__":
    main()
