import csv
import os
import re
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict
from utils.dependency import build_dag, get_topo_order, is_dag_valid
from data.tasks import Task, Workflow

def extract_numeric_id(task_id: str) -> str:
    """从任务ID中提取纯数字部分（如从"M1"提取"1"，从"R5"提取"5"）"""
    numbers = re.findall(r'\d+', task_id)
    return numbers[0] if numbers else task_id

def parse_task_name(task_name: str) -> Tuple[str, List[str]]:
    """解析任务名称，提取任务ID和依赖关系，处理末尾下划线"""
    cleaned_task_name = task_name.rstrip('_')
    parts = cleaned_task_name.split('_')
    if len(parts) == 1:
        return extract_numeric_id(cleaned_task_name), []
    else:
        task_id = extract_numeric_id(parts[0])
        dependencies = [extract_numeric_id(dep) for dep in parts[1:]]
        return task_id, dependencies
    
def get_all_task(file_path: str) -> List[Workflow]:
    """
    从 CSV 加载所有 Workflow，严格适配 Workflow 类结构：
      - 空初始化 Workflow 实例
      - 分步设置 tasks/dag/topo_order 属性
      - 构建任务依赖关系并验证 DAG 有效性
    返回：List[Workflow]
    """ 
    # 按工作流ID分组存储原始数据
    workflow_builder: defaultdict[str, Dict[str, Any]] = defaultdict(lambda: {
        "tasks_dict": {}, # task_id -> Task 对象
        "dep_ids": {},    # task_id -> List[str] 依赖ID映射
        "task_list": []   # 任务列表（保持CSV顺序）
    })
    global_init_time = None  # 全局时间基准

    # ---------- 1. 读取 CSV 创建 Task 对象 ----------
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader, 1):
            if len(row) < 9:
                print(f"[WARN] 第 {row_idx} 行字段不足，跳过")
                continue
            try:
                task_name = row[0].strip()
                workflow_id = row[2].strip()
                start_time = int(row[5].strip())
                end_time = int(row[6].strip())
                cpu_req = float(row[7].strip()) / 100.0
                mem_req = float(row[8].strip())

                # 全局时间基准
                if global_init_time is None:
                    global_init_time = start_time

                start_time -= global_init_time
                end_time -= global_init_time

                # task_id + 依赖 id 列表
                task_id_raw, dep_ids_raw = parse_task_name(task_name)
                task_id = f"{workflow_id}:{task_id_raw}"
                dep_ids = [f"{workflow_id}:{dep_id}" for dep_id in dep_ids_raw]

                task = Task(
                    task_id=task_id,
                    dependencies=dep_ids,
                    duration=0.0,  # 占位，后续修正
                    cpu_req=cpu_req,
                    mem_req=mem_req,
                    workflow_id=workflow_id,
                    start_time=start_time,
                    end_time=end_time
                )
                # 按工作流分组存储
                wf_data = workflow_builder[workflow_id]
                wf_data["tasks_dict"][task_id] = task
                wf_data["dep_ids"][task_id] = dep_ids
                wf_data["task_list"].append(task)
            
            except Exception as e:
                print(f"[WARN] 行 {row_idx} 解析失败，任务名={row[0] if row else '未知'}，错误：{e}")
    
    # -------------------------- 2. 构建Workflow实例 --------------------------
    workflows: List[Workflow] = []
    invalid_workflows: List[str] = []

    for wf_id, data in workflow_builder.items():
        # 1. 先获取任务字典副本
        tasks_dict = data["tasks_dict"].copy()

        # 2. 按当前 Workflow 定义创建实例
        #   Workflow 的签名为 Workflow(wf_id: str, tasks: Dict[str, Task])
        workflow = Workflow(wf_id, tasks_dict)

        # 3. 后续逻辑继续基于 workflow.tasks 操作（与原代码保持一致）
        workflow.tasks = tasks_dict
        # 2. 构建依赖对象引用
        for task_id, task in workflow.tasks.items():
            # 获取有效依赖
            valid_deps = [d for d in data["dep_ids"][task_id] if d in workflow.tasks]
            # 更新任务的依赖对象
            task.dep_objs = [workflow.tasks[d] for d in valid_deps]
            # 更新任务的依赖ID列表（只保留有效依赖）
            task.dependencies = valid_deps
        # 3. 构建并验证DAG
        workflow.dag = workflow._build_dag()
        # 验证DAG有效性
        if not is_dag_valid(workflow.dag):
            print(f"工作流 {wf_id} 包含环或无效依赖，标记为无效")
            invalid_workflows.append(wf_id)
            continue
        # 4. 计算拓扑序
        workflow.topo_order = workflow._get_topo_order()
        # 验证拓扑序完整性
        if not workflow.topo_order or len(workflow.topo_order) != len(workflow.tasks):
            print(f"工作流 {wf_id} 拓扑排序失败，标记为无效")
            invalid_workflows.append(wf_id)
            continue
        # 5.修正任务时长(根据拓扑排序)
        for task_id in workflow.topo_order:
            task = workflow.tasks[task_id]
            if task.dep_objs:
                # 取所有依赖任务的最晚结束时间
                latest_dep_end = max(dep.end_time for dep in task.dep_objs)
                # 计算实际执行时长（确保非负）
                task.duration = max(0.0, task.end_time - latest_dep_end)
            else:
                # 无依赖任务，直接计算时长
                task.duration = max(0.0, task.end_time - task.start_time)
        # 添加到有效工作流列表
        workflows.append(workflow)
    return workflows

def get_all_task2(file_path: str) -> List[Workflow]:
    """
    从 CSV 加载所有 Workflow，对每个 Workflow：
      - 构建 Task 对象
      - 建立 dep_objs
      - 构建 DAG 并验证
      - 修正 duration（根据 start/end）
      - 生成 Workflow 实例
    返回：List[Workflow]
    """
    workflows_tasks: Dict[str, List[Task]] = defaultdict(list)
    workflows_dep_ids: Dict[str, Dict[str, List[str]]] = defaultdict(dict)

    global_init_time = None

    # ---------- 1. 读取 CSV 创建 Task 对象 ----------
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader, 1):
            if len(row) < 9:
                print(f"[WARN] 第 {row_idx} 行字段不足，跳过")
                continue
            try:
                task_name = row[0].strip()
                workflow_id = row[2].strip()
                start_time = int(row[5].strip())
                end_time = int(row[6].strip())
                cpu_req = float(row[7].strip()) / 100.0
                mem_req = float(row[8].strip())

                # 全局时间基准
                if global_init_time is None:
                    global_init_time = start_time

                start_time -= global_init_time
                end_time -= global_init_time

                # task_id + 依赖 id 列表
                task_id_raw, dep_ids_raw = parse_task_name(task_name)
                task_id = f"{workflow_id}:{task_id_raw}"
                dep_ids = [f"{workflow_id}:{dep_id}" for dep_id in dep_ids_raw]

                task = Task(
                    task_id=task_id,
                    dependencies=dep_ids,
                    duration=0.0,  # 占位，后续修正
                    cpu_req=cpu_req,
                    mem_req=mem_req,
                    workflow_id=workflow_id,
                    start_time=start_time,
                    end_time=end_time
                )
                workflows_tasks[workflow_id].append(task)
                workflows_dep_ids[workflow_id][task_id] = dep_ids

            except Exception as e:
                print(f"[WARN] 行 {row_idx} 解析失败，任务名={row[0] if row else '未知'}，错误：{e}")

    # ------------------------------------
    # 2. 构建 Workflow（dep_objs、拓扑排序、duration 修正）
    # ------------------------------------
    workflows: List[Workflow] = []
    invalid_wf = []

    for wf_id, tasks in workflows_tasks.items():
        task_map = {t.task_id: t for t in tasks}

        # 2.1 构建 dep_objs
        for task in tasks:
                task.dep_objs = [task_map[d] for d in workflows_dep_ids[wf_id][task.task_id] if d in task_map]
                task.dependencies = [x.task_id for x in task.dep_objs]

        # 2.2 构建 DAG
        dag = build_dag(tasks)
        if not is_dag_valid(dag):
            invalid_wf.append(wf_id)
            continue

        topo_order = get_topo_order(dag)
        if not topo_order or len(topo_order) != len(tasks):
            invalid_wf.append(wf_id)
            continue

        # 2.3 修正 duration（根据前驱的真实 end_time）
        for tid in topo_order:
            task = task_map[tid]
            if not task.dep_objs:
                task.duration = task.end_time - task.start_time
            else:
                latest_dep_end = max(dep.end_time for dep in task.dep_objs)
                task.duration = task.end_time - latest_dep_end

        # 2.4 创建 Workflow（使用 Task 列表）
        workflows.append(Workflow(wf_id, tasks))

    print(f"读取 {file_path} 完成: 总工作流={len(workflows_tasks)}, 有效={len(workflows)}, 无效={len(invalid_wf)}")
    return workflows
    
def get_all_task1(
    file_path: str, 
) -> Dict[str, List[Dict[str, Any]]]:
    """
    处理已清洗的工作流任务数据，核心逻辑：
    1. 每个工作流的init_time = 该工作流第一个任务的开始时间
    2. 时间转换：任务开始/结束时间 → 相对于init_time的偏移量
    3. CPU转换：原始值（整数）/ 100.0，内存保持浮点数
    4. 保留所有工作流（无任务数量过滤）
    5. 修正duration计算：头节点=endtime-starttime,子节点=自身endtime-依赖前驱最晚endtime
    6. 输出无效工作流详细信息
    参数:
        file_path: 已清洗的任务CSV文件路径
    返回:
        按工作流名称分组的任务字典（结构：{wf_name: {tasks: [], dag: {}, ...}}）
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"任务数据文件不存在: {file_path}")
    
    global_init_time = None
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                if len(row) >= 9:  # 确保有足够的字段
                    # 提取第一个有效任务的开始时间作为全局基准
                    global_init_time = int(row[5].strip())
                    break  # 只需要第一个任务的时间
            except (ValueError, IndexError):
                continue  # 跳过格式错误的行
    
    workflows: Dict[str, List[Dict[str, Any]]] = {}

    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row_idx, row in enumerate(reader, 1):
            try:
                if len(row) < 9:
                    print(f"警告：第{row_idx}行字段不足，跳过")
                    continue

                # 提取基础信息
                task_name = row[0].strip()
                job_name = row[2].strip()

                start_time = int(row[5].strip()) - global_init_time
                end_time = int(row[6].strip()) - global_init_time

                # 资源转换
                cpu = int(row[7].strip()) / 100.0
                mem = float(row[8].strip())

                # 解析任务ID和依赖
                task_id, dependencies = parse_task_name(task_name)

                # 生成任务字典（先不计算duration，后续按DAG拓扑排序修正）
                task = {
                    "task_id": task_id,
                    "original_task_name": task_name,
                    "dependencies": dependencies,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": 0.0,  # 占位，后续修正
                    "cpu_request": cpu,
                    "mem_request": mem,
                    "is_head": len(dependencies) == 0,  # 无依赖则为头节点
                }
                # 添加到对应工作流（初始化工作流结构）
                if job_name not in workflows:
                    workflows[job_name] = {
                        "tasks": [],
                        "dag": {},
                        "is_valid": False,
                        "invalid_reason": "",
                        "topo_order": [],
                    }
                workflows[job_name]["tasks"].append(task)

            except Exception as e:
                print(f"警告：第{row_idx}行处理失败（任务名：{row[0].strip() if len(row)>0 else '未知'}），错误：{str(e)}，跳过")
                continue

        # 处理每个工作流：验证有效性 + 修正duration计算
        valid_workflows = 0
        invalid_workflows = []  # 存储无效工作流信息，便于后续输出

        for wf_name, wf_data in workflows.items():
            tasks = wf_data["tasks"]
            task_map = {task["task_id"]: task for task in tasks}  # 任务ID→任务字典映射

            # 检查1：是否有任务数据
            if not tasks:
                wf_data["is_valid"] = False
                wf_data["invalid_reason"] = "无有效任务数据"
                invalid_workflows.append((wf_name, wf_data["invalid_reason"], len(tasks)))
                continue

            # 检查2：构建DAG并验证是否有循环依赖
            wf_data["dag"] = build_dag(tasks)
            print(wf_data["dag"])
            if not is_dag_valid(wf_data["dag"]):
                wf_data["is_valid"] = False
                wf_data["invalid_reason"] = "DAG存在循环依赖"
                invalid_workflows.append((wf_name, wf_data["invalid_reason"], len(tasks)))
                continue

            # 检查3：获取拓扑排序并验证完整性
            wf_data["topo_order"] = get_topo_order(wf_data["dag"])
            print(wf_data["topo_order"])
            if not wf_data["topo_order"] or len(wf_data["topo_order"]) != len(tasks):
                wf_data["is_valid"] = False
                wf_data["invalid_reason"] = f"拓扑排序失败（排序结果长度：{len(wf_data['topo_order'])}，任务总数：{len(tasks)}）"
                invalid_workflows.append((wf_name, wf_data["invalid_reason"], len(tasks)))
                continue

            # 所有检查通过：修正每个任务的duration
            for task_id in wf_data["topo_order"]:
                task = task_map[task_id]
                dependencies = task["dependencies"]

                if task["is_head"]:
                    # 头节点：duration = 自身end_time - 自身start_time
                    task["duration"] = task["end_time"] - task["start_time"]
                else:
                    # 子节点：duration = 自身end_time - 所有依赖前驱的最晚end_time
                    dep_end_times = []
                    for dep_id in dependencies:
                        if dep_id in task_map:
                            dep_end_times.append(task_map[dep_id]["end_time"])
                        latest_dep_end = max(dep_end_times)
                        task["duration"] = task["end_time"] - latest_dep_end
            
            # 标记为有效工作流
            wf_data["is_valid"] = True
            valid_workflows += 1
        
        # 输出统计信息 + 无效工作流详情
        print("\n" + "="*80)
        print("任务数据处理完成汇总：")
        print(f"- 全局初始时间（原始）：{global_init_time}")
        print(f"- 工作流总数：{len(workflows)}")
        print(f"- 有效DAG工作流数：{valid_workflows}")
        print(f"- 无效工作流数：{len(invalid_workflows)}")
        print(f"- 总任务数：{sum(len(wf_data['tasks']) for wf_data in workflows.values())}")
        print(f"- 有效任务数（仅有效工作流）：{sum(len(wf_data['tasks']) for wf_data in workflows.values() if wf_data['is_valid'])}")

        # 输出无效工作流详细信息
        if invalid_workflows:
            print("\n" + "-"*80)
            print("无效工作流详情：")
            print(f"{'工作流名称':<20} {'无效原因':<40} {'包含任务数':<10}")
            print("-"*80)
            for wf_name, reason, task_count in invalid_workflows:
                print(f"{wf_name:<20} {reason:<40} {task_count:<10}")
        print("="*80 + "\n")

        return workflows

def add_task_data_size_to_workflows(
    workflows: Dict[str, List[Dict[str, Any]]],
    cpu_weight: float = 0.5,   # CPU权重
    mem_weight: float = 0.3,   # 内存权重
    output_ratio: float = 0.2, # 输出数据量占输入的比例
    min_input: float = 1.0,    # 最小输入数据量（MB）
    max_input: float = 200.0   # 最大输入数据量（MB）
) -> Dict[str, List[Dict[str, Any]]]:
    """
    为工作流中的每个任务添加数据量字段（方案二：基于CPU和内存计算）
    参数:
        workflows: get_all_task返回的工作流字典
        cpu_weight: CPU资源在数据量计算中的权重
        mem_weight: 内存资源在数据量计算中的权重
        output_ratio: 输出数据量与输入数据量的比例
        min_input/max_input: 输入数据量的边界值（防止异常值）
    返回:
        包含数据量字段的工作流字典
    """
    # 遍历每个工作流的所有任务
    for workflow_name, tasks in workflows.items():
        for task in tasks:
            # 基于CPU和内存请求计算输入数据量（MB）
            input_data = (task["cpu_request"] * cpu_weight) + (task["mem_request"] * mem_weight)
            
            # 边界处理：确保数据量在合理范围
            input_data_clamped = max(min_input, min(input_data, max_input))
            
            # 计算输出数据量（输入数据量的固定比例）
            output_data = input_data_clamped * output_ratio
            output_data_clamped = max(0.5, min(output_data, 50.0))  # 输出数据量边界
            
            # 为任务添加数据量字段
            task["input_data_size_mb"] = round(input_data_clamped, 2)
            task["output_data_size_mb"] = round(output_data_clamped, 2)
    
    # 输出数据量统计信息
    all_tasks = [task for tasks in workflows.values() for task in tasks]
    avg_input = sum(t["input_data_size_mb"] for t in all_tasks) / len(all_tasks)
    print(f"数据量计算完成：")
    print(f"- 平均输入数据量：{avg_input:.2f}MB")
    print(f"- 数据量范围：{min_input}MB ~ {max_input}MB")
    return workflows

def get_task_type_counts(
    file_path: str,
) -> Dict[Any, int]:
    """
    统计CSV文件第四列（索引为3）的任务类型及其出现次数
    核心逻辑：
    1. 读取CSV文件，处理可能的格式错误
    2. 提取每行第四列的任务类型值
    3. 统计每种类型的出现次数
    4. 返回类型-次数的字典
    
    参数:
        file_path: CSV文件路径
    
    返回:
        键为任务类型，值为出现次数的字典
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 初始化始化任务类型计数器
    type_counts: Dict[Any, int] = {}
    total_rows = 0
    skipped_rows = 0
    
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row_idx, row in enumerate(reader, 1):
            total_rows += 1
            try:
                # 检查行是否有至少4列（索引0-3）
                if len(row) < 4:
                    skipped_rows += 1
                    print(f"警告：第{row_idx}行字段不足（{len(row)}列），至少需要4列，跳过")
                    continue
                
                # 提取第四列（索引3）并去除首尾空白
                task_type = row[3].strip()
                
                # 统计次数（不存在则初始化为0）
                type_counts[task_type] = type_counts.get(task_type, 0) + 1
                
            except Exception as e:
                skipped_rows += 1
                print(f"警告：第{row_idx}行处理失败（{str(e)}），跳过")
                continue
    
    # 输出统计信息
    print(f"任务类型统计完成：")
    print(f"- 总记录数：{total_rows}")
    print(f"- 跳过记录数：{skipped_rows}")
    print(f"- 不同任务类型总数：{len(type_counts)}")
    print("任务类型分布：")
    for task_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {task_type}: {count}次（占比{count/total_rows*100:.2f}%）")
    
    return type_counts
        

def get_last_two_column_pairs(
    file_path: str, 
) -> Set[Tuple[Any, Any]]:
    """
    统计CSV文件最后两列值的所有组合类型
    核心逻辑：
    1. 读取CSV文件，处理可能的格式错误
    2. 提取每行的最后两列值，形成值对
    3. 收集所有出现过的独特值对组合
    4. 返回所有组合类型
    
    参数:
        file_path: CSV文件路径
    
    返回:
        包含最后两列所有值组合的集合，每个元素是一个元组(倒数第二列值, 最后一列值)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 初始化值对集合
    value_pairs: Set[Tuple[Any, Any]] = set()
    total_rows = 0
    skipped_rows = 0
    
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row_idx, row in enumerate(reader, 1):
            total_rows += 1
            try:
                # 检查行是否有足够的列
                if len(row) < 9:
                    skipped_rows += 1
                    print(f"警告：第{row_idx}行字段不足（{len(row)}列），跳过")
                    continue
                
                cpu = int(row[7].strip())
                mem = float(row[8].strip())

                # 添加值对到集合（集合会自动去重）
                value_pairs.add((cpu, mem))
                
            except Exception as e:
                skipped_rows += 1
                print(f"警告：第{row_idx}行处理失败（{str(e)}），跳过")
                continue
    
    # 输出统计信息
    print(f"列值组合统计完成：")
    print(f"- 总记录数：{total_rows}")
    print(f"- 跳过记录数：{skipped_rows}")
    print(f"- 最后两列值的不同组合总数：{len(value_pairs)}")
    
    return value_pairs


if __name__ == "__main__":
    get_all_task("/root/autodl-fs/edgecloud/pretrain/datasource/master_1_valid_tasks_1k.csv")

    # try:
    #     result = get_last_two_column_pairs("/root/autodl-fs/edgecloud/pretrain/datasource/valid_task_3k.csv")
    #     print("\n最后两列出现的所有值组合类型：")
    #     for pair in sorted(result):
    #         print(f"- {pair[0]}, {pair[1]}")
    # except Exception as e:
    #     print(f"处理失败：{str(e)}")

    # 替换为实际文件路径
    # csv_file_path = "/root/autodl-fs/edgecloud/pretrain/datasource/valid_task.csv"
    # try:
    #     task_types = get_task_type_counts(csv_file_path)
    # except Exception as e:
    #     print(f"处理失败：{e}")

