"""
基于遗传算法的成品油配送问题求解器
"""

import random
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import copy
from problem_model import ProblemModel, Solution, Route, DeliveryTask


@dataclass
class Individual:
    """个体（染色体）"""
    chromosome: List[int]  # 任务序列
    vehicle_assignment: List[int]  # 车辆分配
    trip_assignment: List[int]  # 趟次分配
    depot_assignment: List[Tuple[int, int]]  # (起始油库, 结束油库)
    fitness: float = 0.0
    solution: Solution = None

class GeneticAlgorithm:
    """遗传算法求解器"""
    
    def __init__(self, problem_model: ProblemModel, 
                 population_size: int = 100,
                 generations: int = 500,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.2,
                 elite_size: int = 10):
        
        self.model = problem_model
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        
        # 获取所有配送任务
        self.all_tasks = self.model.get_all_delivery_requirements()

        # 调整需求量以符合库存约束
        self._adjust_demands_for_inventory()

        self.num_tasks = len(self.all_tasks)
        
        # 可用车辆列表（只有油库A的车辆）
        self.available_vehicles = [v_id for v_id, v_info in self.model.data.vehicles.items() 
                                 if self.model.data.depots[8001]['vehicle_count'] > 0]
        
        print(f"初始化遗传算法: {self.num_tasks}个任务, {len(self.available_vehicles)}辆车")

    def _adjust_demands_for_inventory(self):
        """
        调整需求量以符合库存约束, 并利用需求的弹性。
        README: 需求量可以随着油罐车的罐容，恰当浮动一些。
        """
        # 统计每种油品的总需求范围
        total_demands = {}
        demand_details = {}  # 记录每个任务的详细信息
        
        # 从原始需求数据而不是预生成的任务列表计算
        for station_id, station_demands in self.model.data.demands.items():
            for product_id, demand_info in station_demands.items():
                if product_id not in total_demands:
                    total_demands[product_id] = {'min': 0, 'likely': 0, 'max': 0}
                    demand_details[product_id] = []
                
                min_d, likely_d, max_d = self.model.data.get_demand_range(station_id, product_id)
                total_demands[product_id]['min'] += min_d
                total_demands[product_id]['likely'] += likely_d
                total_demands[product_id]['max'] += max_d
                
                # 记录每个站点的详细需求，用于后续分配
                demand_details[product_id].append({
                    'station_id': station_id, 
                    'min': min_d, 
                    'likely': likely_d, 
                    'max': max_d
                })

        # 检查库存约束并调整
        adjusted_tasks = []
        for product_id, limits in total_demands.items():
            available_inventory = self.model.data.depot_inventory.get(8001, {}).get(product_id, 0)
            
            # 如果总的最小需求已经超过库存，这是个无解的情况，但我们还是要处理
            if limits['min'] > available_inventory:
                print(f"警告: {self.model.data.products[product_id]['name']}的最小总需求 {limits['min']} 超过库存 {available_inventory}!")
                # 按比例缩减所有需求
                scale_factor = available_inventory / limits['min']
                target_total_demand = available_inventory
            # 如果期望需求超过库存，但在最小需求范围内，我们以库存量为目标配送量
            elif limits['likely'] > available_inventory:
                print(f"警告: {self.model.data.products[product_id]['name']}的期望总需求 {limits['likely']} 超过库存 {available_inventory}，将在需求弹性范围内调整。")
                target_total_demand = available_inventory
            # 如果库存充足，以期望需求为目标
            else:
                target_total_demand = limits['likely']

            # 根据目标总量，按比例分配给各个加油站
            current_total_demand = limits['likely']
            if current_total_demand > 0:
                scale = target_total_demand / current_total_demand
                for demand_info in demand_details[product_id]:
                    # 计算调整后的需求量
                    adj_demand = demand_info['likely'] * scale
                    # 确保调整后的值在站点的上下限内 (或尽可能接近)
                    final_demand = max(demand_info['min'], min(adj_demand, demand_info['max']))
                    
                    if final_demand > 0:
                        # 分配到油罐
                        allocated_tasks = self.model._allocate_demand_to_tanks(
                            demand_info['station_id'], product_id, final_demand
                        )
                        adjusted_tasks.extend(allocated_tasks)
        
        self.all_tasks = adjusted_tasks
        self.num_tasks = len(self.all_tasks)
        
    def create_random_individual(self) -> Individual:
        """创建随机个体"""
        # 随机任务序列
        chromosome = list(range(self.num_tasks))
        random.shuffle(chromosome)
        
        # 随机车辆分配
        vehicle_assignment = [random.choice(self.available_vehicles) for _ in range(self.num_tasks)]
        
        # 随机趟次分配（1或2）
        trip_assignment = [random.randint(1, 2) for _ in range(self.num_tasks)]
        
        # 智能油库分配
        depot_assignment = []
        vehicle_trip_info = {}  # {vehicle_id: {1: end_depot, 2: ...}}
        
        # 为了确保路径的连续性和最终返回A库，我们需要按车辆和趟次来处理
        tasks_by_vehicle_trip = {}
        for i in range(self.num_tasks):
            key = (vehicle_assignment[i], trip_assignment[i])
            if key not in tasks_by_vehicle_trip:
                tasks_by_vehicle_trip[key] = []
            tasks_by_vehicle_trip[key].append(i)

        # 为每个任务设置油库
        temp_depot_assignment = [None] * self.num_tasks
        for vehicle_id in self.available_vehicles:
            # 处理第一趟
            if (vehicle_id, 1) in tasks_by_vehicle_trip:
                start_depot_1 = 8001
                # 第一趟可以在A或B结束
                end_depot_1 = random.choice([8001, 8002])
                for task_idx in tasks_by_vehicle_trip[(vehicle_id, 1)]:
                    temp_depot_assignment[task_idx] = (start_depot_1, end_depot_1)
                
                # 记录第一趟的终点
                if vehicle_id not in vehicle_trip_info:
                    vehicle_trip_info[vehicle_id] = {}
                vehicle_trip_info[vehicle_id][1] = end_depot_1

            # 处理第二趟
            if (vehicle_id, 2) in tasks_by_vehicle_trip:
                # 第二趟的起点是第一趟的终点，如果第一趟不存在，则从A出发
                start_depot_2 = vehicle_trip_info.get(vehicle_id, {}).get(1, 8001)
                # 第二趟必须返回油库A
                end_depot_2 = 8001
                for task_idx in tasks_by_vehicle_trip[(vehicle_id, 2)]:
                    temp_depot_assignment[task_idx] = (start_depot_2, end_depot_2)

        # 确保所有任务都被分配了油库
        depot_assignment = [d if d is not None else (8001, 8001) for d in temp_depot_assignment]

        return Individual(
            chromosome=chromosome,
            vehicle_assignment=vehicle_assignment,
            trip_assignment=trip_assignment,
            depot_assignment=depot_assignment
        )
        
    def decode_individual(self, individual: Individual) -> Solution:
        """将个体解码为解决方案，并尝试保证路径级别的可行性"""
        routes_map = {}  # 使用字典来构建路径: (vehicle_id, trip_number) -> Route
        
        # 按车辆和趟次分组任务
        tasks_by_route_key = {}
        for i, task_idx in enumerate(individual.chromosome):
            route_key = (individual.vehicle_assignment[i], individual.trip_assignment[i])
            if route_key not in tasks_by_route_key:
                tasks_by_route_key[route_key] = []
            tasks_by_route_key[route_key].append(self.all_tasks[task_idx])

        final_routes = []
        
        for route_key, tasks in tasks_by_route_key.items():
            vehicle_id, trip_number = route_key
            
            # 找到该路径对应的油库分配信息 (这里简化处理，取第一个任务的分配)
            # 一个更鲁棒的实现需要确保同一路径的所有任务有相同的油库分配
            task_indices = [i for i, task_idx in enumerate(individual.chromosome) 
                            if (individual.vehicle_assignment[i], individual.trip_assignment[i]) == route_key]
            start_depot, end_depot = individual.depot_assignment[task_indices[0]]

            # 优化路径内的任务顺序
            if len(tasks) > 1:
                tasks = self._optimize_task_order(tasks, start_depot)
            
            # 创建一个Route对象用于评估
            temp_route = Route(
                vehicle_id=vehicle_id,
                trip_number=trip_number,
                start_depot=start_depot,
                end_depot=end_depot,
                tasks=tasks
            )

            # 检查这条路径是否可行
            cap_ok, _ = self.model.check_capacity_constraint(temp_route)
            time_ok, _ = self.model.check_time_constraint(temp_route)
            
            # 只有当路径本身是可行时，才将其加入最终解
            if cap_ok and time_ok:
                final_routes.append(temp_route)

        solution = Solution(routes=final_routes)
        
        # 评估整个解决方案（包括全局约束如库存和同时性）
        # 还需要考虑未被服务的任务作为惩罚项
        evaluated_solution = self.model.evaluate_solution(solution)
        
        # 惩罚未服务的任务
        served_tasks = {id(task) for route in final_routes for task in route.tasks}
        unserved_tasks_count = len(self.all_tasks) - len(served_tasks)
        if unserved_tasks_count > 0:
            evaluated_solution.is_feasible = False
            evaluated_solution.violations.append(f"{unserved_tasks_count} 个任务未被服务")
            
        return evaluated_solution
        
    def _optimize_task_order(self, tasks: List[DeliveryTask], start_depot: int) -> List[DeliveryTask]:
        """使用最近邻算法优化任务顺序"""
        if len(tasks) <= 1:
            return tasks
            
        # 按加油站分组任务
        station_tasks = {}
        for task in tasks:
            if task.station_id not in station_tasks:
                station_tasks[task.station_id] = []
            station_tasks[task.station_id].append(task)
            
        # 使用最近邻算法排序加油站
        unvisited_stations = list(station_tasks.keys())
        ordered_stations = []
        current_location = start_depot
        
        while unvisited_stations:
            # 找到最近的加油站
            nearest_station = min(unvisited_stations, 
                                key=lambda s: self.model.data.get_distance(current_location, s))
            ordered_stations.append(nearest_station)
            unvisited_stations.remove(nearest_station)
            current_location = nearest_station
            
        # 按顺序重新组织任务
        ordered_tasks = []
        for station_id in ordered_stations:
            ordered_tasks.extend(station_tasks[station_id])
            
        return ordered_tasks
        
    def calculate_fitness(self, individual: Individual) -> float:
        """计算个体适应度"""
        solution = self.decode_individual(individual)
        individual.solution = solution
        
        # 基础惩罚：基于违反的约束数量
        penalty = len(solution.violations) * 10000 
        
        # 检查是否有未服务的任务 (这是最严重的违规)
        unserved_task_violations = [v for v in solution.violations if "未被服务" in v]
        if unserved_task_violations:
            # 提取未服务任务的数量
            try:
                unserved_count = int(unserved_task_violations[0].split(" ")[0])
                # 对未服务的任务施加巨大的惩罚
                penalty += unserved_count * 1000000  
            except (ValueError, IndexError):
                pass

        if solution.is_feasible:
            # 可行解：适应度 = 1 / (1 + 成本)
            fitness = 1.0 / (1.0 + solution.total_cost)
        else:
            # 不可行解：适应度 = 1 / (1 + 成本 + 惩罚)
            fitness = 1.0 / (1.0 + solution.total_cost + penalty)
            
        individual.fitness = fitness
        return fitness
        
    def initialize_population(self) -> List[Individual]:
        """初始化种群"""
        population = []

        # 创建随机个体
        for _ in range(self.population_size):
            individual = self.create_random_individual()
            self.calculate_fitness(individual)
            population.append(individual)

        return population
        
    def selection(self, population: List[Individual]) -> List[Individual]:
        """选择操作（锦标赛选择）"""
        selected = []
        tournament_size = 3
        
        for _ in range(self.population_size):
            # 锦标赛选择
            tournament = random.sample(population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            
            # 创建新个体而不是深度拷贝
            new_individual = Individual(
                chromosome=winner.chromosome.copy(),
                vehicle_assignment=winner.vehicle_assignment.copy(),
                trip_assignment=winner.trip_assignment.copy(),
                depot_assignment=winner.depot_assignment.copy(),
                fitness=winner.fitness
                # 不复制solution对象，会在需要时重新计算
            )
            selected.append(new_individual)
            
        return selected
        
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """交叉操作（顺序交叉OX）"""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
            
        # 对任务序列进行顺序交叉
        size = len(parent1.chromosome)
        start, end = sorted(random.sample(range(size), 2))
        
        # 创建子代1
        child1_chromosome = [-1] * size
        child1_chromosome[start:end] = parent1.chromosome[start:end]
        
        # 填充剩余位置
        remaining = [x for x in parent2.chromosome if x not in child1_chromosome]
        j = 0
        for i in range(size):
            if child1_chromosome[i] == -1:
                child1_chromosome[i] = remaining[j]
                j += 1
                
        # 创建子代2
        child2_chromosome = [-1] * size
        child2_chromosome[start:end] = parent2.chromosome[start:end]
        
        remaining = [x for x in parent1.chromosome if x not in child2_chromosome]
        j = 0
        for i in range(size):
            if child2_chromosome[i] == -1:
                child2_chromosome[i] = remaining[j]
                j += 1
                
        # 继承其他属性（随机选择父代之一）
        child1 = Individual(
            chromosome=child1_chromosome,
            vehicle_assignment=parent1.vehicle_assignment.copy() if random.random() < 0.5 else parent2.vehicle_assignment.copy(),
            trip_assignment=parent1.trip_assignment.copy() if random.random() < 0.5 else parent2.trip_assignment.copy(),
            depot_assignment=parent1.depot_assignment.copy() if random.random() < 0.5 else parent2.depot_assignment.copy()
        )
        
        child2 = Individual(
            chromosome=child2_chromosome,
            vehicle_assignment=parent2.vehicle_assignment.copy() if random.random() < 0.5 else parent1.vehicle_assignment.copy(),
            trip_assignment=parent2.trip_assignment.copy() if random.random() < 0.5 else parent1.trip_assignment.copy(),
            depot_assignment=parent2.depot_assignment.copy() if random.random() < 0.5 else parent1.depot_assignment.copy()
        )
        
        return child1, child2
        
    def mutation(self, individual: Individual) -> Individual:
        """变异操作"""
        if random.random() > self.mutation_rate:
            return individual
            
        # 任务序列变异（交换两个位置）
        if random.random() < 0.3:
            i, j = random.sample(range(len(individual.chromosome)), 2)
            individual.chromosome[i], individual.chromosome[j] = individual.chromosome[j], individual.chromosome[i]
            
        # 车辆分配变异
        if random.random() < 0.3:
            i = random.randint(0, len(individual.vehicle_assignment) - 1)
            individual.vehicle_assignment[i] = random.choice(self.available_vehicles)
            
        # 趟次分配变异
        if random.random() < 0.3:
            i = random.randint(0, len(individual.trip_assignment) - 1)
            individual.trip_assignment[i] = random.randint(1, 2)
            
        # 油库分配变异
        if random.random() < 0.3:
            i = random.randint(0, len(individual.depot_assignment) - 1)
            start_depot = 8001
            end_depot = random.choice([8001, 8002])
            individual.depot_assignment[i] = (start_depot, end_depot)
            
        return individual
        
    def solve(self) -> Solution:
        """求解主函数"""
        print("开始遗传算法求解...")
        
        # 初始化种群
        population = self.initialize_population()
        best_individual = max(population, key=lambda x: x.fitness)
        
        print(f"初始最优适应度: {best_individual.fitness:.6f}")
        print(f"初始最优成本: {best_individual.solution.total_cost:.2f}")
        print(f"初始解可行性: {best_individual.solution.is_feasible}")
        
        # 进化过程
        for generation in range(self.generations):
            # 选择
            selected = self.selection(population)
            
            # 交叉和变异
            new_population = []
            
            # 保留精英
            elite = sorted(population, key=lambda x: x.fitness, reverse=True)[:self.elite_size]
            new_population.extend(elite)
            
            # 生成新个体
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                self.calculate_fitness(child1)
                self.calculate_fitness(child2)
                
                new_population.extend([child1, child2])
                
            population = new_population[:self.population_size]
            
            # 更新最优解
            current_best = max(population, key=lambda x: x.fitness)
            if current_best.fitness > best_individual.fitness:
                best_individual = current_best
                
            # 输出进度
            if generation % 50 == 0:
                print(f"第{generation}代: 最优适应度={best_individual.fitness:.6f}, "
                      f"成本={best_individual.solution.total_cost:.2f}, "
                      f"可行性={best_individual.solution.is_feasible}")
                      
        print(f"求解完成！最终适应度: {best_individual.fitness:.6f}")
        print(f"最终成本: {best_individual.solution.total_cost:.2f}")
        print(f"最终可行性: {best_individual.solution.is_feasible}")
        
        return best_individual.solution
