from enum import Enum
from typing import Optional

class FailureReason(Enum):
    RESOURCE_INSUFFICIENT = "resource_insufficient"  # 资源不足（无可用Docker）
    DOCKER_MISMATCH = "docker_mismatch"              # Docker类型不匹配
    DEPENDENCY_NOT_READY = "dependency_not_ready"    # 依赖未满足
    NETWORK_ERROR = "network_error"                  # 网络故障
    HARDWARE_FAILURE = "hardware_failure"            # 硬件故障
    UNKNOWN = "unknown"                              # 未知原因
    HOST_INDEX_OUT_OF_RANGE = "host_index_out_of_range" # 主机索引越界
    TASK_NOT_PENDING = "task_not_pending"            # 任务不在PENDING状态

    @classmethod
    def is_valid_failure(cls, reason: Optional[str]) -> bool:
        """判断是否为有效失败（值得用于训练）"""
        if reason is None:
            return False
        # 支持直接传入 Enum 或其 value 字符串
        if isinstance(reason, cls):
            value = reason.value
        else:
            try:
                # 按 value 解析，例如 "resource_insufficient"
                value = cls(reason).value
            except ValueError:
                return False

        return value in [
            cls.RESOURCE_INSUFFICIENT.value,
            cls.DOCKER_MISMATCH.value,
            cls.DEPENDENCY_NOT_READY.value,
            cls.HOST_INDEX_OUT_OF_RANGE.value,
            cls.TASK_NOT_PENDING.value
        ]
