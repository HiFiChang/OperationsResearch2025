"""
成品油配送问题两阶段启发式算法求解器
遵循 "库存控制 + 路径优化" 的框架
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import random

from data_loader import DataLoader
from problem_model import ProblemModel, DeliveryTask, Route, Solution

@dataclass
class DeliveryRequirement:
    """阶段一生成的配送需求，包含所有计算信息"""
    station_id: int
    product_id: int
    tank_id: int
    quantity: float
    earliest_time: float
    latest_time: float
    min_quantity: float = 0.0
    max_quantity: float = 0.0
    preferred_quantity: float = 0.0

@dataclass
class GAIndividual:
    """遗传算法的个体，包含车辆和任务的双染色体"""
    vehicle_chromosome: List[int] # 车辆ID的排列
    task_chromosome: List[int]    # 任务索引的排列
    fitness: float = 0.0
    solution: Solution = None

class TwoStageSolver:
    """
    实现两阶段启发式算法的求解器
    """
    def __init__(self, data_loader: DataLoader, problem_model: ProblemModel):
        self.data = data_loader
        self.model = problem_model
        print("初始化两阶段求解器...")

    def solve(self):
        """
        执行两阶段求解过程
        """
        # =================================================================
        # 阶段一：库存控制与配送任务生成
        # =================================================================
        print("\n--- 阶段一：库存控制与配送任务生成 ---")
        delivery_tasks, time_windows = self._generate_delivery_tasks()
        print(f"生成了 {len(delivery_tasks)} 个明确的配送任务。")

        # =================================================================
        # 阶段二：路径优化 (使用遗传算法)
        # =================================================================
        print("\n--- 阶段二：路径优化 ---")
        solution = self._optimize_routes_with_ga(delivery_tasks, time_windows)
        
        print("\n求解完成！")
        return solution, time_windows

    def _generate_delivery_tasks(self) -> (List[DeliveryRequirement], Dict):
        """
        根据库存和消耗速度，确定当天需要为哪些加油站配送哪种油品、
        配送多少量，并计算出配送时间窗。
        改进：增加需求量区间处理
        """
        tasks = []
        time_windows = {}
        
        # 追踪已分配的油品总量，确保不超过油库库存
        depot_allocations = {depot_id: {} for depot_id in self.data.depots}

        # 设定库存管理参数
        safety_stock_hours = 4.0  # 安全库存设置为4小时的消耗量
        planning_horizon_hours = 24.0 # 计划 horizon 设置为未来24小时
        target_stock_level_ratio = 0.9 # 补货目标为油罐容量的90%
        working_hours_per_day = 24.0 # 假设油品是24小时匀速消耗的

        # 遍历所有加油站的每个油罐
        for station_id, products in self.data.station_inventory.items():
            for product_id, product_info in products.items():
                if 'tanks' not in product_info: continue
                
                for tank_id, tank_data in product_info['tanks'].items():
                    current_inventory = tank_data['inventory']
                    tank_capacity = tank_data['capacity']

                    # 1. 获取需求量区间信息
                    min_demand, most_likely_demand, max_demand = self.data.get_demand_range(station_id, product_id)
                    if most_likely_demand <= 0:
                        continue # 没有消耗，不需要补货
                    
                    # 2. 使用需求量区间计算消耗速率范围
                    min_consumption_rate = min_demand / working_hours_per_day
                    avg_consumption_rate = most_likely_demand / working_hours_per_day 
                    max_consumption_rate = max_demand / working_hours_per_day

                    # 3. 设定安全库存和补货触发点 (使用最大消耗速率确保安全)
                    safety_stock = max_consumption_rate * safety_stock_hours
                    # 补货触发点 = 安全库存 + 计划周期内的预期消耗量(使用平均值)
                    reorder_point = safety_stock + avg_consumption_rate * planning_horizon_hours

                    # 4. 决策：是否需要补货
                    if current_inventory >= reorder_point:
                        continue # 库存充足，无需补货

                    # 5. 计算补货量范围
                    target_stock_level = tank_capacity * target_stock_level_ratio
                    base_quantity = target_stock_level - current_inventory
                    base_quantity = max(0, base_quantity)
                    
                    # 可接收的最大容量
                    available_space = tank_capacity - current_inventory
                    max_deliverable = min(available_space, max_demand)
                    
                    # 最小补货量：确保至少能维持到下次配送
                    min_delivery_hours = 12.0  # 假设最短12小时后下次配送
                    min_quantity = max(safety_stock - current_inventory, 
                                     avg_consumption_rate * min_delivery_hours)
                    min_quantity = max(0, min(min_quantity, available_space))
                    
                    # 确定实际配送量：在最小和最大之间选择合适的值
                    # 这里使用一个启发式策略：优先满足平均需求
                    if base_quantity <= min_quantity:
                        quantity_to_deliver = min_quantity
                    elif base_quantity >= max_deliverable:
                        quantity_to_deliver = max_deliverable
                    else:
                        quantity_to_deliver = base_quantity

                    # 检查并更新油库A的库存分配
                    depot_id = 8001
                    current_allocated = depot_allocations[depot_id].get(product_id, 0)
                    available_depot_inventory = self.data.depot_inventory.get(depot_id, {}).get(product_id, 0)
                    
                    if current_allocated + quantity_to_deliver > available_depot_inventory:
                        quantity_to_deliver = available_depot_inventory - current_allocated
                    
                    if quantity_to_deliver <= 0:
                        continue
                    
                    # 更新分配量
                    depot_allocations[depot_id][product_id] = current_allocated + quantity_to_deliver

                    # 6. 计算时间窗 (使用区间信息计算更精确的时间窗)
                    # LT (断油时间): 使用最大消耗速率计算最保守的时间
                    if current_inventory > safety_stock:
                        time_to_reach_safety_stock = (current_inventory - safety_stock) / max_consumption_rate
                        latest_time = min(time_to_reach_safety_stock, self.model.working_hours)
                    else:
                        latest_time = 0

                    # ET (不容纳时间): 使用最小消耗速率计算最早可接收时间
                    space_needed = quantity_to_deliver - (tank_capacity - current_inventory)
                    if space_needed > 0:
                        time_to_free_up_space = space_needed / min_consumption_rate
                        earliest_time = min(time_to_free_up_space, self.model.working_hours)
                    else:
                        earliest_time = 0

                    # 确保时间窗合理
                    if earliest_time >= latest_time:
                         earliest_time = 0
                         latest_time = self.model.working_hours

                    # 7. 生成任务（包含需求量区间信息）
                    task_requirement = DeliveryRequirement(
                        station_id=station_id,
                        product_id=product_id,
                        tank_id=tank_id,
                        quantity=quantity_to_deliver,
                        earliest_time=earliest_time,
                        latest_time=latest_time,
                        min_quantity=min_quantity,
                        max_quantity=min(max_deliverable, available_depot_inventory - current_allocated + quantity_to_deliver),
                        preferred_quantity=quantity_to_deliver
                    )
                    
                    tasks.append(task_requirement)
                    
                    # 记录时间窗信息
                    task_key = (station_id, product_id, tank_id)
                    time_windows[task_key] = {
                        'earliest': earliest_time,
                        'latest': latest_time,
                        'min_quantity': task_requirement.min_quantity,
                        'max_quantity': task_requirement.max_quantity,
                        'preferred_quantity': task_requirement.preferred_quantity
                    }
        
        return tasks, time_windows

    def _optimize_routes_with_ga(self, requirements: List[DeliveryRequirement], time_windows: Dict):
        """
        为上一阶段生成的任务列表，分配合适的车辆并规划最优路径。
        """
        self.requirements = requirements
        self.time_windows = time_windows
        self.num_tasks = len(requirements)
        self.available_vehicles = [v_id for v_id, v in self.data.vehicles.items() if v['depot_id'] == 8001]

        # GA 参数 (调整以避免过早收敛)
        population_size = 80      # 增加种群规模以提升多样性
        generations = 300         # 增加迭代次数以进行更充分的搜索
        crossover_rate = 0.8
        mutation_rate = 0.3       # 增加变异率以跳出局部最优
        elite_size = 3            # 减少精英数量以避免过快收敛

        # 初始化种群
        population = self._initialize_population(population_size)

        # 迭代进化
        for gen in range(generations):
            # 评估适应度
            for individual in population:
                self._calculate_fitness(individual)
            
            # 排序并选出精英
            population.sort(key=lambda x: x.fitness, reverse=True)
            new_population = population[:elite_size]

            # 生成新一代
            while len(new_population) < population_size:
                parent1, parent2 = random.choices(population, k=2) # 轮盘赌选择的简化
                child1, child2 = self._crossover(parent1, parent2, crossover_rate)
                self._mutation(child1, mutation_rate)
                self._mutation(child2, mutation_rate)
                new_population.extend([child1, child2])
            
            population = new_population[:population_size]
            
            best_individual = population[0]
            best_solution = best_individual.solution
            
            best_fitness = best_individual.fitness
            best_cost = best_solution.total_cost if best_solution else float('inf')
            
            # 计算未完成任务数
            num_tasks_in_solution = sum(len(r.tasks) for r in best_solution.routes)
            unserved_tasks = self.num_tasks - num_tasks_in_solution

            # 构建日志信息
            log_message = (
                f"第 {gen+1}/{generations} 代: "
                f"最高适应度 = {best_fitness:.6f}, "
                f"最低成本 = {best_cost:.2f}, "
                f"未完成任务 = {unserved_tasks}"
            )
            
            # 如果方案是不可行的，并且有违规信息，则打印第一条
            if not best_solution.is_feasible and best_solution.violations:
                log_message += f", 违规: {best_solution.violations[0]}"

            print(log_message)

        # 返回最优解
        best_individual = population[0]
        self._calculate_fitness(best_individual)
        return best_individual.solution

    def _initialize_population(self, size: int) -> List[GAIndividual]:
        """
        初始化种群 (v2.0):
        - 结合贪心启发式和随机生成，创建高质量的初始种群
        """
        population = []
        num_greedy_individuals = 10 # 创建10个由贪心算法生成的"专家"个体
        
        print(f"生成 {num_greedy_individuals} 个贪心种子...")
        for i in range(num_greedy_individuals):
            greedy_individual = self._create_greedy_individual()
            if greedy_individual:
                population.append(greedy_individual)
        
        print(f"生成 {size - len(population)} 个随机个体...")
        while len(population) < size:
            # 车辆染色体：可用车辆的随机排列
            vehicle_chromo = random.sample(self.available_vehicles, len(self.available_vehicles))
            # 任务染色体：任务索引的随机排列
            task_chromo = random.sample(list(range(self.num_tasks)), self.num_tasks)
            population.append(GAIndividual(vehicle_chromosome=vehicle_chromo, task_chromosome=task_chromo))
            
        return population

    def _create_greedy_individual(self) -> GAIndividual:
        """
        使用贪心插入启发式创建一个高质量的、完整的初始个体。
        这是算法能够成功找到可行解的关键。
        """
        # 1. 初始化一个空解决方案
        temp_solution = Solution(routes=[])
        assigned_task_indices = set()
        
        # 2. 迭代所有任务，将它们插入到成本最低的位置
        for task_idx in range(self.num_tasks):
            best_insertion = {
                "route_idx": -1,
                "task_pos": -1,
                "cost": float('inf'),
                "new_route_obj": None
            }
            
            req = self.requirements[task_idx]
            task_to_insert = DeliveryTask(station_id=req.station_id, product_id=req.product_id, quantity=req.quantity, tank_id=req.tank_id)

            # 2a. 尝试插入到现有路径
            for r_idx, route in enumerate(temp_solution.routes):
                for t_pos in range(len(route.tasks) + 1):
                    # 创建一个插入后的新路径进行测试
                    new_tasks = route.tasks[:t_pos] + [task_to_insert] + route.tasks[t_pos:]
                    
                    # 检查新路径的可行性
                    temp_route = Route(vehicle_id=route.vehicle_id, trip_number=route.trip_number, start_depot=route.start_depot, end_depot=route.end_depot, tasks=new_tasks, start_time=route.start_time)
                    
                    cap_ok, _ = self.model.check_capacity_constraint(temp_route)
                    time_ok, _ = self.model.check_time_constraint(temp_route, self.time_windows) # 此时不考虑全局冲突，因为我们只构建一个方案

                    if cap_ok and time_ok:
                        cost_increase = self.model.calculate_route_cost(temp_route) - self.model.calculate_route_cost(route)
                        if cost_increase < best_insertion["cost"]:
                            best_insertion.update({
                                "route_idx": r_idx, "task_pos": t_pos,
                                "cost": cost_increase, "new_route_obj": temp_route
                            })
            
            # 2b. 如果无法插入现有路径，尝试开启新路径
            # （为了简化，我们这里只考虑第一趟行程；一个更完整的实现会考虑第二趟）
            if best_insertion["route_idx"] == -1:
                # 寻找一个空闲车辆来开启新路径
                used_vehicles = {r.vehicle_id for r in temp_solution.routes if r.trip_number == 1}
                available_vehicles = [v_id for v_id in self.available_vehicles if v_id not in used_vehicles]
                
                if available_vehicles:
                    new_vehicle_id = available_vehicles[0]
                    new_route = Route(vehicle_id=new_vehicle_id, trip_number=1, start_depot=8001, end_depot=8001, tasks=[task_to_insert], start_time=8.0)
                    
                    cap_ok, _ = self.model.check_capacity_constraint(new_route)
                    time_ok, _ = self.model.check_time_constraint(new_route, self.time_windows)
                    
                    if cap_ok and time_ok:
                        # 作为一个新路径，其成本就是全部成本
                        cost = self.model.calculate_route_cost(new_route)
                        if cost < best_insertion["cost"]:
                             best_insertion.update({
                                "route_idx": len(temp_solution.routes), "task_pos": 0,
                                "cost": cost, "new_route_obj": new_route
                            })

            # 3. 执行最优插入
            if best_insertion["new_route_obj"]:
                if best_insertion["route_idx"] < len(temp_solution.routes):
                    # 替换现有路径
                    temp_solution.routes[best_insertion["route_idx"]] = best_insertion["new_route_obj"]
                else:
                    # 添加新路径
                    temp_solution.routes.append(best_insertion["new_route_obj"])
                assigned_task_indices.add(task_idx)

        # 4. 从构建好的方案中提取染色体
        task_chromosome = []
        for route in temp_solution.routes:
            for task in route.tasks:
                # 找到原始任务的索引
                for i in range(self.num_tasks):
                    req = self.requirements[i]
                    if req.station_id == task.station_id and req.product_id == task.product_id and req.tank_id == task.tank_id:
                        if i not in task_chromosome:
                            task_chromosome.append(i)
                            break
        
        # 补全可能因简化而遗漏的任务
        for i in range(self.num_tasks):
            if i not in task_chromosome:
                task_chromosome.append(i)

        vehicle_chromosome = random.sample(self.available_vehicles, len(self.available_vehicles))

        return GAIndividual(vehicle_chromosome=vehicle_chromosome, task_chromosome=task_chromosome)

    def _calculate_fitness(self, individual: GAIndividual):
        solution = self._decode_individual(individual)
        individual.solution = solution
        
        # 新的适应度计算 (v3.0):
        # 核心思想：创建一个平滑的惩罚梯度，引导算法首先完成所有任务，然后再优化成本。
        # 1. 主要惩罚：针对未完成的任务。这是算法优化的首要方向。
        num_tasks_in_solution = sum(len(r.tasks) for r in solution.routes)
        unserved_tasks = self.num_tasks - num_tasks_in_solution
        
        # 为每个未服务的任务设置一个远超任何可能成本的巨大罚分
        # 这确保了"完整性"是第一优先级。
        unserved_task_penalty = unserved_tasks * 1_000_000

        # 2. 次要惩罚：针对其他约束违反（理论上新解码器已避免，作为保险）。
        # 仅当所有任务都完成后，这些次要约束才变得有意义。
        other_violations_penalty = 0
        if not solution.is_feasible and unserved_tasks == 0:
            other_violations_penalty = 500_000 # 例如，处理一些边缘情况的约束

        penalty = unserved_task_penalty + other_violations_penalty
        
        individual.fitness = 1.0 / (1.0 + solution.total_cost + penalty)

    def _decode_individual(self, individual: GAIndividual) -> Solution:
        """
        构造式解码器 (v2.0):
        - 动态管理车辆可用时间
        - 动态构建全局站点日程表，在构建时避免冲突
        """
        routes = []
        assigned_task_indices = [False] * self.num_tasks
        
        vehicle_available_time = {v_id: 8.0 for v_id in self.available_vehicles}
        station_schedules = {} # 全局日程表: {station_id: [(start, end, v_id, trip)]}

        # 遍历车辆排列
        for vehicle_id in individual.vehicle_chromosome:
            # 每辆车最多可以跑两趟
            for trip_number in [1, 2]:
                current_route_tasks = []
                
                # 确定当前趟次的出发时间和起点
                start_time = vehicle_available_time[vehicle_id]
                
                # 确定起点油库
                if trip_number == 1:
                    start_depot = 8001  # 第一趟从油库A出发
                else:
                    # 第二趟的起点取决于第一趟的终点
                    previous_routes = [r for r in routes if r.vehicle_id == vehicle_id and r.trip_number == 1]
                    if previous_routes:
                        start_depot = previous_routes[0].end_depot
                    else:
                        start_depot = 8001  # 如果没有第一趟，默认从油库A开始

                # 遍历任务排列，尝试为当前车辆的当前趟次分配任务
                for task_idx in individual.task_chromosome:
                    if assigned_task_indices[task_idx]:
                        continue # 任务已被分配

                    req = self.requirements[task_idx]
                    
                    potential_tasks = current_route_tasks + [
                        DeliveryTask(station_id=req.station_id, product_id=req.product_id, 
                                     quantity=req.quantity, tank_id=req.tank_id)
                    ]
                    
                    temp_route = Route(
                        vehicle_id=vehicle_id, trip_number=trip_number,
                        start_depot=start_depot, end_depot=8001,
                        tasks=potential_tasks, start_time=start_time
                    )
                    
                    # 约束检查 (现在传入全局日程表)
                    cap_ok, _ = self.model.check_capacity_constraint(temp_route)
                    time_ok, _ = self.model.check_time_constraint(temp_route, self.time_windows, station_schedules)
                    
                    if cap_ok and time_ok:
                        current_route_tasks.append(potential_tasks[-1])
                        assigned_task_indices[task_idx] = True
                
                if current_route_tasks:
                    # 决定路径的终点油库
                    if trip_number == self.model.max_trips_per_vehicle:
                        # 最后一趟必须返回油库A (根据实验要求)
                        end_depot = 8001
                    else:
                        # 非最后一趟可以选择就近的油库
                        if not current_route_tasks:
                            end_depot = 8001
                        else:
                            last_station = current_route_tasks[-1].station_id
                            # 选择距离最近的油库作为终点
                            dist_to_a = self.data.get_distance(last_station, 8001)
                            dist_to_b = self.data.get_distance(last_station, 8002)
                            end_depot = 8001 if dist_to_a <= dist_to_b else 8002
                    
                    initial_route = Route(
                        vehicle_id=vehicle_id, trip_number=trip_number,
                        start_depot=start_depot, end_depot=end_depot, tasks=current_route_tasks,
                        start_time=start_time
                    )
                    
                    # 应用局部搜索优化
                    optimized_route = self._local_search_2_opt(initial_route, self.time_windows, station_schedules)
                    
                    # 应用需求量区间优化
                    loading_optimized_route = self._optimize_loading(optimized_route, self.time_windows)
                    final_route = self._adaptive_quantity_adjustment(loading_optimized_route, self.time_windows)
                    
                    routes.append(final_route)

                    # 更新车辆的下一个可用时间
                    route_duration = self.model.calculate_route_time(final_route)
                    vehicle_available_time[vehicle_id] = final_route.start_time + route_duration

                    # 更新全局站点日程表
                    self._update_station_schedules(station_schedules, final_route)
        
        solution = Solution(routes=routes)
        return self.model.evaluate_solution(solution, self.time_windows, self.num_tasks)

    def _update_station_schedules(self, schedules: Dict, route: Route):
        """用一条已确定的路径来更新全局日程表"""
        current_time = route.start_time
        current_location = route.start_depot
        speed = self.model.data.vehicles[route.vehicle_id]['speed']
        
        visited_stations = []
        for task in route.tasks:
            if task.station_id not in visited_stations:
                # 计算到达和服务时间
                distance = self.model.data.get_distance(current_location, task.station_id)
                travel_time = distance / speed
                arrival_time = current_time + travel_time

                task_key = (task.station_id, task.product_id, task.tank_id)
                tw = self.time_windows[task_key]
                service_start_time = max(arrival_time, tw['earliest'])
                
                unload_time = self.model.data.stations[task.station_id]['unload_time']
                service_end_time = service_start_time + unload_time
                
                # 添加到日程表
                if task.station_id not in schedules:
                    schedules[task.station_id] = []
                schedules[task.station_id].append((service_start_time, service_end_time, route.vehicle_id, route.trip_number))

                current_time = service_end_time
                current_location = task.station_id
                visited_stations.append(task.station_id)

    def _local_search_2_opt(self, route: Route, time_windows: Dict, existing_schedules: Dict) -> Route:
        """对单条路径应用2-Opt局部搜索优化。(现在需要传入日程表)"""
        # 提取唯一的站点访问顺序
        station_sequence = []
        for task in route.tasks:
            if task.station_id not in station_sequence:
                station_sequence.append(task.station_id)
        
        if len(station_sequence) <= 1:
            return route # 任务站点数太少，无法优化

        best_route = route
        best_cost = self.model.calculate_route_cost(route)
        improved = True
        
        while improved:
            improved = False
            for i in range(len(station_sequence) - 1):
                for j in range(i + 1, len(station_sequence)):
                    # 产生新的站点顺序 (2-opt move: reverse segment)
                    new_station_sequence = station_sequence[:i+1] + station_sequence[i+1:j+1][::-1] + station_sequence[j+1:]

                    # 根据新的站点顺序重构任务列表
                    new_tasks = []
                    for station_id in new_station_sequence:
                        for task in route.tasks:
                            if task.station_id == station_id:
                                new_tasks.append(task)
                    
                    # 创建新路径并检查可行性
                    new_route = Route(
                        vehicle_id=route.vehicle_id, trip_number=route.trip_number,
                        start_depot=route.start_depot, end_depot=route.end_depot,
                        tasks=new_tasks, start_time=route.start_time
                    )
                    
                    cap_ok, _ = self.model.check_capacity_constraint(new_route)
                    time_ok, _ = self.model.check_time_constraint(new_route, time_windows, existing_schedules)
                    
                    if cap_ok and time_ok:
                        new_cost = self.model.calculate_route_cost(new_route)
                        if new_cost < best_cost:
                            best_route = new_route
                            best_cost = new_cost
                            station_sequence = new_station_sequence
                            improved = True
                            break 
                if improved:
                    break
        
        return best_route

    def _order_crossover(self, parent1: List, parent2: List) -> Tuple[List, List]:
        """对排列进行顺序交叉 (Order Crossover - OX)"""
        size = len(parent1)
        if size < 2:
            return parent1, parent2

        child1, child2 = [-1] * size, [-1] * size
        start, end = sorted(random.sample(range(size), 2))

        # 子代1
        child1[start:end] = parent1[start:end]
        p2_filtered = [item for item in parent2 if item not in child1]
        idx = 0
        for i in range(size):
            if child1[i] == -1:
                child1[i] = p2_filtered[idx]
                idx += 1
        
        # 子代2
        child2[start:end] = parent2[start:end]
        p1_filtered = [item for item in parent1 if item not in child2]
        idx = 0
        for i in range(size):
            if child2[i] == -1:
                child2[i] = p1_filtered[idx]
                idx += 1
        
        return child1, child2

    def _crossover(self, p1: GAIndividual, p2: GAIndividual, rate: float) -> Tuple[GAIndividual, GAIndividual]:
        c1_task, c2_task = p1.task_chromosome, p2.task_chromosome
        c1_vehicle, c2_vehicle = p1.vehicle_chromosome, p2.vehicle_chromosome

        if random.random() < rate:
            # 对任务染色体使用顺序交叉
            c1_task, c2_task = self._order_crossover(p1.task_chromosome, p2.task_chromosome)

            # 对车辆染色体也使用顺序交叉
            c1_vehicle, c2_vehicle = self._order_crossover(p1.vehicle_chromosome, p2.vehicle_chromosome)

        child1 = GAIndividual(c1_vehicle, c1_task)
        child2 = GAIndividual(c2_vehicle, c2_task)
        
        return (child1, child2)

    def _mutation(self, individual: GAIndividual, rate: float):
        # 任务数小于2时，无法进行交换变异
        if self.num_tasks < 2:
            return

        if random.random() < rate:
            # 任务染色体：交换变异
            i, j = random.sample(range(self.num_tasks), 2)
            individual.task_chromosome[i], individual.task_chromosome[j] = individual.task_chromosome[j], individual.task_chromosome[i]
        
        # 车辆数小于2时，无法进行交换变异
        if len(self.available_vehicles) < 2:
            return

        if random.random() < rate:
            # 车辆染色体：交换变异
            i, j = random.sample(range(len(self.available_vehicles)), 2)
            individual.vehicle_chromosome[i], individual.vehicle_chromosome[j] = individual.vehicle_chromosome[j], individual.vehicle_chromosome[i]

    def _optimize_loading(self, route: Route, time_windows: Dict) -> Route:
        """
        优化车辆装载，利用需求量区间调整配送量以提高装载率
        """
        if not route.tasks:
            return route
            
        vehicle = self.data.vehicles[route.vehicle_id]
        total_capacity = vehicle['total_capacity']
        
        # 计算当前装载量
        current_load = sum(task.quantity for task in route.tasks)
        
        # 如果装载率已经很高（>90%），不需要优化
        if current_load / total_capacity > 0.9:
            return route
            
        # 尝试增加配送量以提高装载率
        remaining_capacity = total_capacity - current_load
        optimized_tasks = []
        
        for task in route.tasks:
            task_key = (task.station_id, task.product_id, task.tank_id)
            if task_key not in time_windows:
                optimized_tasks.append(task)
                continue
                
            tw_info = time_windows[task_key]
            min_qty = tw_info.get('min_quantity', task.quantity)
            max_qty = tw_info.get('max_quantity', task.quantity)
            
            # 计算可以增加的最大量
            max_increase = min(max_qty - task.quantity, remaining_capacity)
            
            if max_increase > 0:
                # 增加配送量，但不超过容量限制
                new_quantity = task.quantity + max_increase
                remaining_capacity -= max_increase
                
                # 创建新任务
                optimized_task = DeliveryTask(
                    station_id=task.station_id,
                    product_id=task.product_id,
                    quantity=new_quantity,
                    tank_id=task.tank_id
                )
                optimized_tasks.append(optimized_task)
            else:
                optimized_tasks.append(task)
                
            if remaining_capacity <= 0:
                break
        
        # 创建优化后的路径
        optimized_route = Route(
            vehicle_id=route.vehicle_id,
            trip_number=route.trip_number,
            start_depot=route.start_depot,
            end_depot=route.end_depot,
            tasks=optimized_tasks,
            start_time=route.start_time
        )
        
        return optimized_route
    
    def _adaptive_quantity_adjustment(self, route: Route, time_windows: Dict) -> Route:
        """
        自适应调整配送量：根据路径长度和时间限制调整任务的配送量
        """
        if not route.tasks:
            return route
            
        # 计算路径总时间
        route_time = self.model.calculate_route_time(route)
        
        # 如果时间超限，尝试减少配送量
        if route_time > self.model.working_hours:
            return self._reduce_quantities_for_time(route, time_windows)
        
        # 如果时间充裕，尝试增加配送量
        elif route_time < self.model.working_hours * 0.8:
            return self._increase_quantities_for_efficiency(route, time_windows)
        
        return route
    
    def _reduce_quantities_for_time(self, route: Route, time_windows: Dict) -> Route:
        """减少配送量以满足时间限制"""
        adjusted_tasks = []
        
        for task in route.tasks:
            task_key = (task.station_id, task.product_id, task.tank_id)
            if task_key not in time_windows:
                adjusted_tasks.append(task)
                continue
                
            tw_info = time_windows[task_key]
            min_qty = tw_info.get('min_quantity', task.quantity)
            
            # 减少到最小可接受量
            new_quantity = max(min_qty, task.quantity * 0.8)
            
            adjusted_task = DeliveryTask(
                station_id=task.station_id,
                product_id=task.product_id,
                quantity=new_quantity,
                tank_id=task.tank_id
            )
            adjusted_tasks.append(adjusted_task)
        
        return Route(
            vehicle_id=route.vehicle_id,
            trip_number=route.trip_number,
            start_depot=route.start_depot,
            end_depot=route.end_depot,
            tasks=adjusted_tasks,
            start_time=route.start_time
        )
    
    def _increase_quantities_for_efficiency(self, route: Route, time_windows: Dict) -> Route:
        """增加配送量以提高效率"""
        vehicle = self.data.vehicles[route.vehicle_id]
        total_capacity = vehicle['total_capacity']
        current_load = sum(task.quantity for task in route.tasks)
        available_capacity = total_capacity - current_load
        
        if available_capacity <= 0:
            return route
            
        adjusted_tasks = []
        remaining_capacity = available_capacity
        
        for task in route.tasks:
            task_key = (task.station_id, task.product_id, task.tank_id)
            if task_key not in time_windows:
                adjusted_tasks.append(task)
                continue
                
            tw_info = time_windows[task_key]
            max_qty = tw_info.get('max_quantity', task.quantity)
            
            # 计算可以增加的量
            possible_increase = min(max_qty - task.quantity, remaining_capacity)
            
            if possible_increase > 0:
                new_quantity = task.quantity + possible_increase
                remaining_capacity -= possible_increase
                
                adjusted_task = DeliveryTask(
                    station_id=task.station_id,
                    product_id=task.product_id,
                    quantity=new_quantity,
                    tank_id=task.tank_id
                )
                adjusted_tasks.append(adjusted_task)
            else:
                adjusted_tasks.append(task)
                
            if remaining_capacity <= 0:
                break
        
        return Route(
            vehicle_id=route.vehicle_id,
            trip_number=route.trip_number,
            start_depot=route.start_depot,
            end_depot=route.end_depot,
            tasks=adjusted_tasks,
            start_time=route.start_time
        ) 