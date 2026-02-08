import os
import sys
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from method.SchedulingAlgorithm import SchedulingAlgorithm


_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from decision_unified.models.decision_unified import DecisionUnified


def load_du_dataset_stats(dataset_path: str) -> Dict[str, Any]:
    with open(dataset_path, "rb") as f:
        trajectories = pickle.load(f)

    states_list = []
    traj_lens = []
    returns = []
    max_action = 0

    for tr in trajectories:
        obs = np.asarray(tr["observations"], dtype=np.float32)
        acts = np.asarray(tr["actions"])  # [T, 1]
        rews = np.asarray(tr["rewards"], dtype=np.float32)

        if obs.size:
            states_list.append(obs)
            traj_lens.append(len(obs))
        returns.append(float(np.sum(rews)))

        if acts.size:
            try:
                max_action = max(max_action, int(np.max(acts)))
            except Exception:
                pass

    if len(states_list) == 0:
        raise ValueError(f"No trajectories with observations found in dataset: {dataset_path}")

    states = np.concatenate(states_list, axis=0)
    state_mean = np.mean(states, axis=0)
    state_std = np.std(states, axis=0) + 1e-6

    returns_arr = np.asarray(returns, dtype=np.float32)
    scale = float(np.max(np.abs(returns_arr))) if returns_arr.size and float(np.max(np.abs(returns_arr))) > 0 else 1.0

    max_ep_len = int(np.max(np.asarray(traj_lens, dtype=np.int32))) if len(traj_lens) else 1000
    discrete_action_bins = int(max_action) + 1

    target_return = float(np.percentile(returns_arr, 90)) if returns_arr.size else 0.0

    return {
        "state_mean": state_mean,
        "state_std": state_std,
        "scale": scale,
        "max_ep_len": max_ep_len,
        "discrete_action_bins": discrete_action_bins,
        "target_return": target_return,
    }


class DecisionUnifiedPolicy(SchedulingAlgorithm):
    def __init__(
        self,
        all_host_ids: List[str],
        model_path: str,
        dataset_path: str,
        env_name: str = "edgecloud",
        dataset_name: str = "pretrain",
        device: str = "cuda",
        K: int = 20,
        embed_dim: int = 256,
        n_layer: int = 3,
        n_head: int = 1,
        dropout: float = 0.1,
        conv_window_size: int = 6,
        activation_function: str = "gelu",
        alpha_T: float = 0.85,
        remove_act_embs: bool = False,
    ):
        self.all_host_ids = list(all_host_ids)
        self.host_to_idx = {hid: i for i, hid in enumerate(self.all_host_ids)}
        self.idx_to_host = {i: hid for hid, i in self.host_to_idx.items()}

        stats = load_du_dataset_stats(dataset_path)
        self.state_mean = torch.from_numpy(np.asarray(stats["state_mean"], dtype=np.float32)).to(device=device)
        self.state_std = torch.from_numpy(np.asarray(stats["state_std"], dtype=np.float32)).to(device=device)
        self.scale = float(stats["scale"])
        self.max_ep_len = int(stats["max_ep_len"])
        self.discrete_action_bins = int(stats["discrete_action_bins"])
        self.target_return = float(stats["target_return"]) / float(self.scale)

        self.device = device
        self.K = int(K)
        self.act_dim = 1
        self.state_dim = int(self.state_mean.shape[0])

        self.model = DecisionUnified(
            env_name=env_name,
            dataset=dataset_name,
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            hidden_size=int(embed_dim),
            max_length=self.K,
            max_ep_len=self.max_ep_len,
            action_tanh=False,
            remove_act_embs=bool(remove_act_embs),
            discrete_action_bins=self.discrete_action_bins,
            n_layer=int(n_layer),
            n_head=int(n_head),
            n_inner=int(4 * embed_dim),
            drop_p=float(dropout),
            activation_function=activation_function,
            window_size=int(conv_window_size),
            resid_pdrop=float(dropout),
            attn_pdrop=float(dropout),
            embd_pdrop=float(dropout),
            alpha_T=float(alpha_T),
        ).to(device=device)

        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self._ctx: Dict[Any, Dict[str, Any]] = {}

    def reset(self):
        self._ctx = {}

    def update(
        self,
        state: Any,
        action: str,
        reward: float,
        next_state: Any,
        done: bool,
        master_idx: int,
        action_info: Dict,
        actor_lr: float,
        critic_lr: float,
        global_step: int,
    ):
        return None

    def save_model(self, path: str):
        raise NotImplementedError

    def load_model(self, path: str):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _ensure_ctx(self, ctx_key: Any):
        if ctx_key in self._ctx:
            return
        self._ctx[ctx_key] = {
            "states": torch.zeros((0, self.state_dim), device=self.device, dtype=torch.float32),
            "actions": torch.zeros((0, self.act_dim), device=self.device, dtype=torch.float32),
            "returns": torch.tensor([[self.target_return]], device=self.device, dtype=torch.float32),
            "timesteps": torch.tensor([0], device=self.device, dtype=torch.long),
            "t": 0,
        }

    def record_reward(self, ctx_key: Any, reward: float):
        self._ensure_ctx(ctx_key)
        ctx = self._ctx[ctx_key]
        cur_rtg = ctx["returns"][-1, 0]
        next_rtg = cur_rtg - float(reward) / float(self.scale)
        ctx["returns"] = torch.cat(
            [ctx["returns"], torch.tensor([[next_rtg]], device=self.device, dtype=torch.float32)],
            dim=0,
        )
        ctx["t"] = int(ctx["t"]) + 1
        t_clamped = min(int(ctx["t"]), int(self.max_ep_len) - 1)
        ctx["timesteps"] = torch.cat(
            [ctx["timesteps"], torch.tensor([t_clamped], device=self.device, dtype=torch.long)],
            dim=0,
        )

    def select_action(
        self,
        state: Any,
        valid_host_ids: List[str],
        master_idx: Any,
        epsilon: float = 0.0,
    ) -> Tuple[Optional[str], Dict]:
        if not valid_host_ids:
            return None, {"reason": "no_valid_hosts"}

        self._ensure_ctx(master_idx)
        ctx = self._ctx[master_idx]

        s = np.asarray(state, dtype=np.float32).reshape(1, -1)
        s = torch.from_numpy(s).to(device=self.device, dtype=torch.float32)
        s = (s - self.state_mean) / self.state_std
        ctx["states"] = torch.cat([ctx["states"], s], dim=0)

        # keep histories aligned to number of states (model expects same lengths)
        t_states = int(ctx["states"].shape[0])
        if int(ctx["returns"].shape[0]) > t_states:
            ctx["returns"] = ctx["returns"][:t_states]
        if int(ctx["timesteps"].shape[0]) > t_states:
            ctx["timesteps"] = ctx["timesteps"][:t_states]

        if ctx["returns"].shape[0] < ctx["states"].shape[0]:
            pad_n = int(ctx["states"].shape[0] - ctx["returns"].shape[0])
            ctx["returns"] = torch.cat(
                [ctx["returns"], ctx["returns"][[-1]].repeat(pad_n, 1)],
                dim=0,
            )
        if ctx["timesteps"].shape[0] < ctx["states"].shape[0]:
            pad_n = int(ctx["states"].shape[0] - ctx["timesteps"].shape[0])
            last = int(ctx["timesteps"][-1].item()) if ctx["timesteps"].numel() else 0
            ctx["timesteps"] = torch.cat(
                [ctx["timesteps"], torch.tensor([last] * pad_n, device=self.device, dtype=torch.long)],
                dim=0,
            )

        if ctx["actions"].shape[0] < ctx["states"].shape[0]:
            pad_n = int(ctx["states"].shape[0] - ctx["actions"].shape[0])
            ctx["actions"] = torch.cat(
                [ctx["actions"], torch.zeros((pad_n, self.act_dim), device=self.device, dtype=torch.float32)],
                dim=0,
            )

        if int(ctx["states"].shape[0]) > int(self.K):
            ctx["states"] = ctx["states"][-int(self.K) :]
            ctx["actions"] = ctx["actions"][-int(self.K) :]
            ctx["returns"] = ctx["returns"][-int(self.K) :]
            ctx["timesteps"] = ctx["timesteps"][-int(self.K) :]

        valid_indices = []
        for hid in valid_host_ids:
            idx = self.host_to_idx.get(hid)
            if idx is not None and 0 <= int(idx) < int(self.discrete_action_bins):
                valid_indices.append(int(idx))

        if not valid_indices:
            return None, {"reason": "no_valid_indices"}

        if float(epsilon) > 0 and np.random.rand() < float(epsilon):
            action_idx = int(np.random.choice(valid_indices))
            ctx["actions"][-1, 0] = float(action_idx)
            return self.idx_to_host[action_idx], {"action_idx": action_idx, "mode": "random"}

        with torch.no_grad():
            logits = self.model.get_action(
                states=ctx["states"],
                actions=ctx["actions"],
                returns_to_go=ctx["returns"],
                timesteps=ctx["timesteps"],
            )

            if logits.ndim != 1 or int(logits.shape[0]) != int(self.discrete_action_bins):
                return None, {"reason": "bad_logits_shape", "shape": tuple(logits.shape)}

            masked = torch.full_like(logits, -1e9)
            masked[torch.tensor(valid_indices, device=logits.device, dtype=torch.long)] = logits[
                torch.tensor(valid_indices, device=logits.device, dtype=torch.long)
            ]
            action_idx = int(torch.argmax(masked).item())

        ctx["actions"][-1, 0] = float(action_idx)
        return self.idx_to_host[action_idx], {"action_idx": action_idx, "mode": "model"}
