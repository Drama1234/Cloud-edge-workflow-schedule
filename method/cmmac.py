import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import os,random
from typing import List, Optional, Tuple, Dict,Any
from method.SchedulingAlgorithm import SchedulingAlgorithm
import json
from typing import List, Dict, Any, Optional, Tuple

class FCBlock(nn.Module):
    def __init__(self, nin, nh, act=nn.ReLU(), init_scale=1.0):
        super().__init__()
        self.fc = nn.Linear(nin, nh)
        self.act = act
        nn.init.xavier_uniform_(self.fc.weight, gain=init_scale)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        return self.act(self.fc(x))
    
class Actor(nn.Module):
    def __init__(self, 
                possible_host_ids: List[str], 
                state_dim: int,
                scope: str = "actor", 
                summaries_dir: Optional[str] = None, 
                device=None):
        nn.Module.__init__(self)
        # 记录基本属性
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = len(possible_host_ids)
        self.scope = scope
        self.summary_writer = SummaryWriter(log_dir=os.path.join(summaries_dir, f"summaries_{scope}")) if summaries_dir else None
        
        # 维护 host_id 和网络输出索引之间的映射
        self.host_to_idx: Dict[str, int] = {host: idx for idx, host in enumerate(possible_host_ids)}
        self.idx_to_host: Dict[int, str] = {idx: host for host, idx in self.host_to_idx.items()}

        # 策略网络
        self.policy_net = nn.Sequential(
            FCBlock(nin=self.state_dim, nh=128, act=nn.ReLU()),
            FCBlock(nin=128, nh=64, act=nn.ReLU()),
            FCBlock(nin=64, nh=32, act=nn.ReLU()),
            FCBlock(nin=32, nh=self.action_dim, act=nn.ReLU()) # 输出 logits
        ).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters())

    def _compute_policy_output(self, policy_state, neighbor_mask):
        policy_state = torch.tensor(policy_state, dtype=torch.float32, device=self.device)
        neighbor_mask = torch.tensor(neighbor_mask, dtype=torch.float32, device=self.device)
        logits = self.policy_net(policy_state) + 1
        valid_logits = logits * neighbor_mask
        softmaxprob = torch.softmax(torch.log(valid_logits + 1e-8), dim=1)
        return softmaxprob
    
    def _create_mask_from_host_ids(self, avail_host_ids: List[str]) -> np.ndarray:
        """
        根据可用的 host_id 列表创建一个掩码。
        掩码中，可用 host_id 对应的索引位置为 1，否则为 0。
        """
        mask = np.zeros(self.action_dim)
        for host_id in avail_host_ids:
            if host_id in self.host_to_idx:
                mask[self.host_to_idx[host_id]] = 1
        return mask
    
    def action(self, state: np.ndarray, avail_actions: List[str], epsilon: float = 0.0) -> Optional[str]:
        """
        param state: 状态
        param avail_actions: 可用的 HOST_ID 列表
        param epsilon: 探索率
        return: 选择的 HOST_ID
        """
        action_info = self._get_action_with_info(state, avail_actions, epsilon)
        return action_info['action']
    
    def _get_action_with_info(self, state: np.ndarray, avail_host_ids: List[str], epsilon: float = 0.0) -> Dict:
        """
        内部方法，返回动作 (host_id) 和训练所需的额外信息。
        """
        if not avail_host_ids:
            return {'action': None, 'action_choosen_mat': np.array([]), 'policy_state': np.array([]), 'neighbor_mask': np.array([])}
        # 1. 创建掩码
        mask = self._create_mask_from_host_ids(avail_host_ids)
        # 增加 batch 维度
        s = np.expand_dims(state, axis=0)
        masks = np.expand_dims(mask, axis=0)
        # 2. 计算所有动作的概率（推理阶段不需要梯度）
        with torch.no_grad():
            action_probs = self._compute_policy_output(s, masks).detach().cpu().numpy().flatten()
        # 3. ε-greedy 选择
        if np.random.rand() < epsilon:
            # 探索：随机选择一个可用的 host_id
            chosen_host_id = np.random.choice(avail_host_ids)
        else:
            # 利用：选择概率最高的可用 host_id
            # 只考虑可用动作的概率
            avail_indices = [self.host_to_idx[host] for host in avail_host_ids if host in self.host_to_idx]
            avail_probs = action_probs[avail_indices]
            chosen_idx_in_avail = np.argmax(avail_probs)
            chosen_host_id = avail_host_ids[chosen_idx_in_avail]
        # 4. 准备训练数据
        # 将 chosen_host_id 转换为 one-hot 向量
        action_one_hot = np.zeros(self.action_dim)
        if chosen_host_id in self.host_to_idx:
            action_one_hot[self.host_to_idx[chosen_host_id]] = 1
        
        return {
            'action': chosen_host_id,
            'action_choosen_mat': np.expand_dims(action_one_hot, axis=0), # 增加 batch 维度
            'policy_state': np.expand_dims(state, axis=0), # 增加 batch 维度
            'neighbor_mask': masks # 包含 batch 维度的掩码
        }
    
    def update(self,policy_state, advantage, action_choosen_mat, neighbor_mask, learning_rate, global_step):
        """
        更新本地的 Actor 网络
        """ 
        for param_group in self.policy_optimizer.param_groups:
            param_group['lr'] = learning_rate

        policy_state_tensor = torch.tensor(policy_state, dtype=torch.float32, device=self.device)
        advantage_tensor = advantage.to(self.device)
        action_mat_tensor = torch.tensor(action_choosen_mat, dtype=torch.float32, device=self.device)
        mask_tensor = torch.tensor(neighbor_mask, dtype=torch.float32, device=self.device)

        logits = self.policy_net(policy_state_tensor) + 1
        valid_logits = logits * mask_tensor
        softmaxprob = torch.softmax(torch.log(valid_logits + 1e-8), dim=1)
        logsoftmaxprob = torch.log(softmaxprob + 1e-8)

        neglogprob = -logsoftmaxprob * action_mat_tensor
        actor_loss = torch.mean(torch.sum(neglogprob * advantage_tensor, dim=1))
        entropy = -torch.mean(torch.sum(softmaxprob * logsoftmaxprob, dim=1))
        policy_loss = actor_loss - 0.01 * entropy

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.summary_writer:
            self.summary_writer.add_scalar(f"{self.scope}/policy_loss", policy_loss.item(), global_step)
            self.summary_writer.add_scalar(f"{self.scope}/advantage_mean", advantage_tensor.mean().item(), global_step)
            self.summary_writer.add_scalar(f"{self.scope}/entropy", entropy.item(), global_step)

        return policy_loss.item()
    
class Critic(nn.Module):
    def __init__(self,state_dim,scope="critic", summaries_dir=None, device=None):
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.scope = scope
        self.summary_writer = SummaryWriter(log_dir=os.path.join(summaries_dir, f"summaries_{scope}")) if summaries_dir else None
        
        # 价值网络
        self.value_net = nn.Sequential(
            FCBlock(nin=self.state_dim, nh=128, act=nn.ReLU()),
            FCBlock(nin=128, nh=64, act=nn.ReLU()),
            FCBlock(nin=64, nh=32, act=nn.ReLU()),
            FCBlock(nin=32, nh=1, act=nn.ReLU()) # 输出状态价值
        ).to(self.device)

        self.value_optimizer = optim.Adam(self.value_net.parameters())
        self.value_criterion = nn.MSELoss()

    def predict(self, s):
        """预测状态价值"""
        self.value_net.eval()
        with torch.no_grad():
            s_tensor = torch.tensor(s, dtype=torch.float32, device=self.device)
            value_output = self.value_net(s_tensor).cpu().numpy()
        self.value_net.train()
        return value_output
    
    def compute_advantage(self, curr_state_value, next_state, node_reward, gamma):
        """计算优势函数"""
        advantage = []
        node_reward = node_reward.flatten()
        qvalue_next = self.predict(next_state).flatten()

        # 简化处理，假设单个 transition
        temp_adv = sum(node_reward) + gamma * sum(qvalue_next) - curr_state_value[0]
        advantage.append(temp_adv)
        
        return torch.tensor(advantage, dtype=torch.float32, device=self.device).unsqueeze(1)
    
    def update(self, s, target, learning_rate, global_step):
        """
        更新全局的 Critic 网络
        """
        for param_group in self.value_optimizer.param_groups:
            param_group['lr'] = learning_rate

        s_tensor = torch.tensor(s, dtype=torch.float32, device=self.device)
        target_tensor = torch.tensor(target, dtype=torch.float32, device=self.device)

        self.value_optimizer.zero_grad()
        value_output = self.value_net(s_tensor)
        value_loss = self.value_criterion(value_output, target_tensor)
        value_loss.backward()
        self.value_optimizer.step()

        if self.summary_writer:
            self.summary_writer.add_scalar(f"{self.scope}/value_loss", value_loss.item(), global_step)
            self.summary_writer.add_scalar(f"{self.scope}/value_output_mean", value_output.mean().item(), global_step)

        return value_loss.item()
    
class GlobalReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, transition):
        """transition: (state, action, reward, next_state, done, actor_info)"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # 确保采样的数量不超过 buffer 中实际的经验数
        actual_batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, actual_batch_size)

    def __len__(self):
        return len(self.buffer)
    
class DistributedAgent:
    def __init__(self, actor: Actor, critic: Critic, global_replay_buffer: GlobalReplayBuffer, gamma=0.99, batch_size=32, update_critic_every=10):
        self.actor = actor
        self.critic = critic
        self.global_replay_buffer = global_replay_buffer
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_critic_every = update_critic_every # 每收集 N 条经验，更新一次 Critic
        self._local_step_counter = 0 # 本地计数器，用于触发 Critic 更新
    
    def select_action(self, state, avail_actions, epsilon=0.0) -> tuple:
        """
        选择一个动作。
        :return: (chosen_host_id, action_info)
        """
        # 直接调用 Actor 的内部方法来获取动作和详细信息
        # 注意：这里假设 state 是单个状态向量 (state_dim,)
        action_info = self.actor._get_action_with_info(state, avail_actions, epsilon)
        return action_info['action'], action_info
    
    def update(self, state, action, reward, next_state, done, action_info, actor_learning_rate, critic_learning_rate, global_step):
        # 1. 将经验存入全局回放池
        self.global_replay_buffer.push((state, action, reward, next_state, done, action_info))
        self._local_step_counter += 1

        # 2. 定期从回放池采样并更新 Critic
        # 当本地积累了一定经验，并且回放池中有足够经验时，触发更新
        if self._local_step_counter % self.update_critic_every == 0 and len(self.global_replay_buffer) >= self.batch_size:
            self._update_critic(critic_learning_rate, global_step)
            # 可以选择在这里也更新 Actor，实现类似 A2C 的同步更新
            # 但当前设计是每个 Agent 各自更新自己的 Actor，更偏向于异步

        # 3. 更新本地 Actor (这是异步的，每个 Agent 自己决定何时更新)
        # 确保 action_info 包含有效的训练数据
        if action_info['policy_state'].size > 0 and action_info['action_choosen_mat'].size > 0:
            # 使用 Critic 计算优势
            # curr_state_value 是基于采取动作时的状态计算的
            curr_state_value = self.critic.predict(action_info['policy_state'])
            
            # advantage 的计算应该基于这次交互的 (s, a, r, s')
            # next_state 和 reward 是这次交互的直接结果
            advantage = self.critic.compute_advantage(
                curr_state_value, 
                np.expand_dims(next_state, axis=0), 
                np.array([reward]), 
                self.gamma
            )
            
            # 更新本地 Actor
            self.actor.update(
                action_info['policy_state'],
                advantage,
                action_info['action_choosen_mat'],
                action_info['neighbor_mask'], # 现在 mask 已经是 (1, action_dim)，可以直接使用
                actor_learning_rate,
                global_step
            )

    def _update_critic(self, learning_rate, global_step):
        """
        从全局回放池采样一个批次的经验，然后更新 Critic 网络。
        """
        batch = self.global_replay_buffer.sample(self.batch_size)
        
        # 从 batch 中提取数据并打包成张量 (batch processing)
        # 这是一个通用的处理方式，将 list of tuples 转换为 tuple of lists
        states, actions, rewards, next_states, dones, actor_infos = zip(*batch)

        # 将 list of numpy arrays 转换为一个大的 numpy array
        states_np = np.array(states)
        next_states_np = np.array(next_states)
        rewards_np = np.array(rewards).reshape(-1, 1)
        dones_np = np.array(dones, dtype=np.float32).reshape(-1, 1)

        # --- 计算 TD 目标 ---
        # V(s')
        with torch.no_grad():
            next_state_values = self.critic.predict(next_states_np)
        
        # TD Target: r + γ * V(s') * (1 - done)
        td_targets = rewards_np + self.gamma * next_state_values * (1 - dones_np)

        # --- 更新 Critic ---
        # 使用批次数据进行更新
        critic_loss = self.critic.update(states_np, td_targets, learning_rate, global_step)
        
        # (可选) 可以返回或记录损失
        return critic_loss

# CMMAC算法适配器
class CMMACAlgorithmAdapter(SchedulingAlgorithm):
    def __init__(self, 
                 host_ids: List[str], 
                 state_dim: int,
                 num_masters: int,
                 gamma: float = 0.95,
                 batch_size: int = 64,
                 buffer_capacity: int = 100000,
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-3,
                 summaries_dir: str = "./logs/"):
        
        self.host_ids = host_ids
        self.state_dim = state_dim
        self.num_masters = num_masters
        
        # 创建全局经验回放池
        self.global_buffer = GlobalReplayBuffer(buffer_capacity)
        
        # 创建全局Critic
        self.critic = Critic(
            state_dim=self.state_dim,
            scope="critic",
            summaries_dir=summaries_dir,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # 为每个master创建Actor和DistributedAgent
        self.distributed_agents: List[DistributedAgent] = []
        self._init_agents(summaries_dir, actor_lr, gamma, batch_size)
    
    def _init_agents(self, summaries_dir: str, actor_lr: float, gamma: float, batch_size: int):
        for agent_id in range(self.num_masters):
            # 创建Actor
            actor = Actor(
                possible_host_ids=self.host_ids,
                state_dim=self.state_dim,
                scope=f"actor_agent_{agent_id}",
                summaries_dir=summaries_dir,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            
            # 设置Actor的学习率
            for param_group in actor.policy_optimizer.param_groups:
                param_group['lr'] = actor_lr
            
            # 创建DistributedAgent
            distributed_agent = DistributedAgent(
                actor=actor,
                critic=self.critic,
                global_replay_buffer=self.global_buffer,
                gamma=gamma,
                batch_size=batch_size,
                update_critic_every=10
            )
            
            self.distributed_agents.append(distributed_agent)
    
    def select_action(self, state: Any, valid_host_ids: List[str], master_idx: int, epsilon: float = 0.0) -> Tuple[Optional[str], Dict]:
        if master_idx < 0 or master_idx >= len(self.distributed_agents):
            return None, {}
        
        agent = self.distributed_agents[master_idx]
        return agent.select_action(state, valid_host_ids, epsilon)
    
    def update(self, state: Any, action: str, reward: float, next_state: Any, done: bool, 
               master_idx: int, action_info: Dict, actor_lr: float, critic_lr: float, global_step: int):
        if master_idx < 0 or master_idx >= len(self.distributed_agents):
            return
        
        agent = self.distributed_agents[master_idx]
        agent.update(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            action_info=action_info,
            actor_learning_rate=actor_lr,
            critic_learning_rate=critic_lr,
            global_step=global_step
        )
    
    def save_model(self, path: str):
        os.makedirs(path, exist_ok=True)
        
        # 保存critic模型
        critic_path = os.path.join(path, "critic.pth")
        torch.save({
            'state_dict': self.critic.state_dict(),
            'optimizer': self.critic.value_optimizer.state_dict()
        }, critic_path)
        
        # 保存每个Actor模型
        for idx, agent in enumerate(self.distributed_agents):
            actor_path = os.path.join(path, f"actor_agent_{idx}.pth")
            torch.save({
                'state_dict': agent.actor.state_dict(),
                'optimizer': agent.actor.policy_optimizer.state_dict()
            }, actor_path)
        
        # 保存配置
        config = {
            'num_masters': self.num_masters,
            'state_dim': self.state_dim,
            'host_ids': self.host_ids
        }
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f)
    
    def load_model(self, path: str):
        # 加载配置
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # 加载critic模型
        critic_path = os.path.join(path, "critic.pth")
        checkpoint = torch.load(critic_path)
        self.critic.load_state_dict(checkpoint['state_dict'])
        self.critic.value_optimizer.load_state_dict(checkpoint['optimizer'])
        
        # 加载每个Actor模型
        for idx, agent in enumerate(self.distributed_agents):
            actor_path = os.path.join(path, f"actor_agent_{idx}.pth")
            checkpoint = torch.load(actor_path)
            agent.actor.load_state_dict(checkpoint['state_dict'])
            agent.actor.policy_optimizer.load_state_dict(checkpoint['optimizer'])

    def reset(self):
        """重置算法内部状态，用于环境重置时清空经验池等。"""
        # 清空全局回放池
        self.global_buffer.buffer.clear()
        self.global_buffer.position = 0


    

    




    



    

    





    





        




        






        




        





    






        


