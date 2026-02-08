
import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from method.SchedulingAlgorithm import SchedulingAlgorithm


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 256
    update_every: int = 2048
    device: str = "cuda"
    hidden_dim: int = 256


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PPOPolicy(SchedulingAlgorithm):
    def __init__(
        self,
        host_ids: List[str],
        state_dim: int = 256,
        cfg: Optional[PPOConfig] = None,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
    ):
        self.host_ids = list(host_ids)
        self.host_to_idx = {hid: i for i, hid in enumerate(self.host_ids)}
        self.idx_to_host = {i: hid for hid, i in self.host_to_idx.items()}
        self.state_dim = int(state_dim)
        self.act_dim = int(len(self.host_ids))

        self.cfg = cfg or PPOConfig()
        device = self.cfg.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.actor = _MLP(self.state_dim, self.act_dim, int(self.cfg.hidden_dim)).to(self.device)
        self.critic = _MLP(self.state_dim, 1, int(self.cfg.hidden_dim)).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(actor_lr))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=float(critic_lr))

        self.reset()

    def reset(self):
        self._buf: Dict[str, List[Any]] = {
            "states": [],
            "actions": [],
            "logps": [],
            "rewards": [],
            "dones": [],
            "values": [],
            "valid_indices": [],
        }

    def _tensor_state(self, state: Any) -> torch.Tensor:
        s = np.asarray(state, dtype=np.float32).reshape(1, -1)
        if s.shape[1] != self.state_dim:
            if s.shape[1] < self.state_dim:
                pad = np.zeros((1, self.state_dim - s.shape[1]), dtype=np.float32)
                s = np.concatenate([s, pad], axis=1)
            else:
                s = s[:, : self.state_dim]
        return torch.from_numpy(s).to(self.device)

    def _masked_dist(self, logits: torch.Tensor, valid_host_ids: List[str]) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        mask = torch.zeros((self.act_dim,), device=self.device, dtype=torch.bool)
        for hid in valid_host_ids:
            idx = self.host_to_idx.get(hid)
            if idx is not None:
                mask[idx] = True
        masked_logits = logits.squeeze(0)
        masked_logits = masked_logits.masked_fill(~mask, -1e9)
        dist = torch.distributions.Categorical(logits=masked_logits)
        return dist, mask

    def _masked_dist_from_indices(self, logits_1d: torch.Tensor, valid_indices: List[int]) -> torch.distributions.Categorical:
        mask = torch.zeros((self.act_dim,), device=logits_1d.device, dtype=torch.bool)
        for idx in valid_indices:
            if 0 <= int(idx) < self.act_dim:
                mask[int(idx)] = True
        masked_logits = logits_1d.masked_fill(~mask, -1e9)
        return torch.distributions.Categorical(logits=masked_logits)

    def select_action(
        self,
        state: Any,
        valid_host_ids: List[str],
        master_idx: int,
        epsilon: float = 0.0,
    ) -> Tuple[Optional[str], Dict]:
        if not valid_host_ids:
            return None, {"reason": "no_valid_hosts"}

        with torch.no_grad():
            s = self._tensor_state(state)
            logits = self.actor(s)
            value = self.critic(s).squeeze(-1)
            dist, mask = self._masked_dist(logits, valid_host_ids)

            if float(epsilon) > 0.0 and np.random.rand() < float(epsilon):
                act_host_id = str(np.random.choice(valid_host_ids))
                act_idx = int(self.host_to_idx[act_host_id])
                logp = dist.log_prob(torch.tensor(act_idx, device=self.device)).detach()
            else:
                act_idx = int(dist.sample().item())
                logp = dist.log_prob(torch.tensor(act_idx, device=self.device)).detach()
                act_host_id = self.idx_to_host.get(act_idx)

        if act_host_id is None:
            return None, {"reason": "invalid_action"}

        info = {
            "action_idx": int(act_idx),
            "logp": float(logp.item()),
            "value": float(value.item()),
            "valid_action_indices": [int(self.host_to_idx[h]) for h in valid_host_ids if h in self.host_to_idx],
        }
        return act_host_id, info

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
        if actor_lr is not None:
            for pg in self.actor_opt.param_groups:
                pg["lr"] = float(actor_lr)
        if critic_lr is not None:
            for pg in self.critic_opt.param_groups:
                pg["lr"] = float(critic_lr)

        a_idx = self.host_to_idx.get(str(action))
        if a_idx is None:
            return

        self._buf["states"].append(np.asarray(state, dtype=np.float32))
        self._buf["actions"].append(int(a_idx))
        self._buf["rewards"].append(float(reward))
        self._buf["dones"].append(bool(done))
        self._buf["logps"].append(float(action_info.get("logp", 0.0)))
        self._buf["values"].append(float(action_info.get("value", 0.0)))
        self._buf["valid_indices"].append(list(action_info.get("valid_action_indices", [])))

        if len(self._buf["states"]) >= int(self.cfg.update_every):
            self._learn()
            self.reset()

    def _learn(self):
        n = len(self._buf["states"])
        if n <= 1:
            return

        states = torch.from_numpy(np.asarray(self._buf["states"], dtype=np.float32)).to(self.device)
        actions = torch.from_numpy(np.asarray(self._buf["actions"], dtype=np.int64)).to(self.device)
        old_logps = torch.from_numpy(np.asarray(self._buf["logps"], dtype=np.float32)).to(self.device)
        rewards = torch.from_numpy(np.asarray(self._buf["rewards"], dtype=np.float32)).to(self.device)
        dones = torch.from_numpy(np.asarray(self._buf["dones"], dtype=np.float32)).to(self.device)
        values = torch.from_numpy(np.asarray(self._buf["values"], dtype=np.float32)).to(self.device)
        valid_indices = list(self._buf.get("valid_indices", []))

        with torch.no_grad():
            next_values = self.critic(states[1:]).squeeze(-1)
            next_values = torch.cat([next_values, torch.zeros((1,), device=self.device)], dim=0)
            deltas = rewards + float(self.cfg.gamma) * (1.0 - dones) * next_values - values

            adv = torch.zeros_like(rewards)
            gae = 0.0
            for t in reversed(range(n)):
                gae = float(deltas[t].item()) + float(self.cfg.gamma) * float(self.cfg.gae_lambda) * (1.0 - float(dones[t].item())) * gae
                adv[t] = gae
            ret = adv + values
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        idxs = np.arange(n)
        for _ in range(int(self.cfg.update_epochs)):
            np.random.shuffle(idxs)
            for start in range(0, n, int(self.cfg.minibatch_size)):
                mb = idxs[start : start + int(self.cfg.minibatch_size)]
                if len(mb) == 0:
                    continue

                mb_states = states[mb]
                mb_actions = actions[mb]
                mb_old_logps = old_logps[mb]
                mb_adv = adv[mb]
                mb_ret = ret[mb]

                logits = self.actor(mb_states)
                mb_logps = []
                mb_entropy = []
                for j, orig_idx in enumerate(mb):
                    dist = self._masked_dist_from_indices(logits[j], valid_indices[int(orig_idx)])
                    mb_logps.append(dist.log_prob(mb_actions[j]))
                    mb_entropy.append(dist.entropy())
                logps = torch.stack(mb_logps, dim=0)
                entropy = torch.stack(mb_entropy, dim=0).mean()

                ratio = torch.exp(logps - mb_old_logps)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - float(self.cfg.clip_eps), 1.0 + float(self.cfg.clip_eps)) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                v = self.critic(mb_states).squeeze(-1)
                value_loss = F.mse_loss(v, mb_ret)

                loss = policy_loss + float(self.cfg.vf_coef) * value_loss - float(self.cfg.ent_coef) * entropy

                self.actor_opt.zero_grad(set_to_none=True)
                self.critic_opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), float(self.cfg.max_grad_norm))
                self.actor_opt.step()
                self.critic_opt.step()

    def save_model(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save({"state_dict": self.actor.state_dict(), "optimizer": self.actor_opt.state_dict()}, os.path.join(path, "actor.pth"))
        torch.save({"state_dict": self.critic.state_dict(), "optimizer": self.critic_opt.state_dict()}, os.path.join(path, "critic.pth"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(
                {
                    "host_ids": self.host_ids,
                    "state_dim": int(self.state_dim),
                    "cfg": self.cfg.__dict__,
                },
                f,
                indent=2,
            )

    def load_model(self, path: str):
        with open(os.path.join(path, "config.json"), "r") as f:
            cfg = json.load(f)
        ck_actor = torch.load(os.path.join(path, "actor.pth"), map_location=self.device)
        ck_critic = torch.load(os.path.join(path, "critic.pth"), map_location=self.device)
        self.actor.load_state_dict(ck_actor["state_dict"])
        self.critic.load_state_dict(ck_critic["state_dict"])
        if "optimizer" in ck_actor:
            self.actor_opt.load_state_dict(ck_actor["optimizer"])
        if "optimizer" in ck_critic:
            self.critic_opt.load_state_dict(ck_critic["optimizer"])

