from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any

# 算法抽象接口
class SchedulingAlgorithm(ABC):
    @abstractmethod
    def select_action(self, state: Any, valid_host_ids: List[str], master_idx: int, epsilon: float = 0.0) -> Tuple[Optional[str], Dict]:
        pass

    @abstractmethod
    def update(self, state: Any, action: str, reward: float, next_state: Any, done: bool, 
               master_idx: int, action_info: Dict, actor_lr: float, critic_lr: float, global_step: int):
        pass
    
    @abstractmethod
    def save_model(self, path: str):
        pass
    
    @abstractmethod
    def load_model(self, path: str):
        pass
    
    @abstractmethod
    def reset(self):
        pass