from main import DOCKER_CONFIG
from data.tasks import Task, Workflow
from typing import List, Dict, Any, Optional
# -------------------------- 任务-Docker类型映射 --------------------------
def map_task_to_docker_type(task: Task) -> int:
    """
    当任务的 cpu/mem 不是精确匹配某个docker时，按“最小能满足”的策略选择docker类型：
      - 找到所有 cpu >= task.cpu_req 且 mem >= task.mem_req 的 docker
      - 如果没有，扩展条件为 cpu >= task.cpu_req OR mem >= task.mem_req 的近似选择
      - 返回内存最小的满足项（conservative）或 cpu 最接近的
    将 docker_type 写回 task.docker_type 并返回
    """
    task_cpu = task.cpu_req
    task_mem = task.mem_req
    # -------- Step 1: 找到能满足需求的 docker --------
    candidate_dts = []
    for dt, conf in DOCKER_CONFIG.items():
        if conf["cpu"] >= task_cpu and conf["mem"] >= task_mem:
            # 计算浪费（越少越好）
            waste = (conf["cpu"] - task_cpu) + (conf["mem"] - task_mem)
            candidate_dts.append((dt, waste))
    
    # 若找到满足需求的 → 选浪费最少的
    if candidate_dts:
        candidate_dts.sort(key=lambda x: x[1])  # waste最小
        target_dt = candidate_dts[0][0]
        task.docker_type = target_dt
        return target_dt
    # -------- Step 2: fallback：选择资源最大的 docker --------
    # 避免不存在满足需求的 docker 时任务直接无法执行
    target_dt = max(
        DOCKER_CONFIG.keys(),
        key=lambda dt: (DOCKER_CONFIG[dt]["cpu"], DOCKER_CONFIG[dt]["mem"])
    )
    task.docker_type = target_dt
    return target_dt
