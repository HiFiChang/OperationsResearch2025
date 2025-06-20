"""
启发式求解器 - 用于生成高质量的初始解
"""

import random
import numpy as np
from typing import List, Dict, Tuple
from problem_model import ProblemModel, Solution, Route, DeliveryTask
from data_loader import DataLoader

class HeuristicSolver:
    """启发式求解器"""
    
    def __init__(self, problem_model: ProblemModel):
        self.model = problem_model
        self.data = problem_model.data
        
    def solve(self) -> Solution:
        """使用启发式方法求解"""
        print("开始启发式求解...")
        
        # 获取所有任务
        all_tasks = self.model.get_all_delivery_requirements()
        
        # 调整需求量
        self._adjust_demands(all_tasks)
        
        # 按加油站分组任务
        station_tasks = self._group_tasks_by_station(all_tasks)
        
        # 使用贪心算法构造解
        solution = self._greedy_construction(station_tasks)
        
        # 评估解
        solution = self.model.evaluate_solution(solution)
        
        print(f"启发式求解完成: 成本={solution.total_cost:.2f}, 可行性={solution.is_feasible}")
        
        return solution
        
    def _adjust_demands(self, tasks: List[DeliveryTask]):
        """调整需求量以符合库存约束"""
        # 统计每种油品的总需求
        total_demands = {}
        for task in tasks:
            if task.product_id not in total_demands:
                total_demands[task.product_id] = 0
            total_demands[task.product_id] += task.quantity
            
        # 检查库存约束并调整
        for product_id, total_demand in total_demands.items():
            available_inventory = self.data.depot_inventory.get(8001, {}).get(product_id, 0)
            
            if total_demand > available_inventory:
                scale_factor = available_inventory / total_demand
                print(f"调整{self.data.products[product_id]['name']}需求量，比例={scale_factor:.2f}")
                
                for task in tasks:
                    if task.product_id == product_id:
                        task.quantity *= scale_factor
                        
    def _group_tasks_by_station(self, tasks: List[DeliveryTask]) -> Dict[int, List[DeliveryTask]]:
        """按加油站分组任务"""
        station_tasks = {}
        for task in tasks:
            if task.station_id not in station_tasks:
                station_tasks[task.station_id] = []
            station_tasks[task.station_id].append(task)
        return station_tasks
        
    def _greedy_construction(self, station_tasks: Dict[int, List[DeliveryTask]]) -> Solution:
        """贪心构造算法"""
        routes = []
        unassigned_stations = list(station_tasks.keys())
        
        # 获取可用车辆（按容量排序，优先使用大容量车辆）
        available_vehicles = sorted(
            [v_id for v_id, v_info in self.data.vehicles.items()],
            key=lambda v_id: self.data.vehicles[v_id]['total_capacity'],
            reverse=True
        )
        
        vehicle_index = 0
        
        while unassigned_stations and vehicle_index < len(available_vehicles):
            vehicle_id = available_vehicles[vehicle_index]
            vehicle = self.data.vehicles[vehicle_id]
            
            # 为当前车辆构造路径（最多2趟）
            for trip_number in [1, 2]:
                if not unassigned_stations:
                    break
                    
                route = self._construct_route(
                    vehicle_id, trip_number, unassigned_stations, station_tasks
                )
                
                if route and route.tasks:
                    routes.append(route)
                    
                    # 移除已分配的加油站
                    assigned_stations = set(task.station_id for task in route.tasks)
                    unassigned_stations = [s for s in unassigned_stations if s not in assigned_stations]
                    
            vehicle_index += 1
            
        return Solution(routes=routes)
        
    def _construct_route(self, vehicle_id: int, trip_number: int, 
                        available_stations: List[int], 
                        station_tasks: Dict[int, List[DeliveryTask]]) -> Route:
        """为单个车辆构造一条路径"""
        
        vehicle = self.data.vehicles[vehicle_id]
        compartment_capacity = vehicle['compartment1_capacity']  # 假设两个车仓容量相等
        
        route = Route(
            vehicle_id=vehicle_id,
            trip_number=trip_number,
            start_depot=8001,  # 从油库A开始
            end_depot=8001,    # 返回油库A
            tasks=[]
        )
        
        current_load = {60001: 0, 60002: 0, 60003: 0}  # 每种油品的当前载重
        current_location = 8001
        current_time = 8.0  # 从8点开始
        
        remaining_stations = available_stations.copy()
        
        while remaining_stations:
            # 找到最适合的下一个加油站
            best_station = None
            best_score = float('-inf')
            
            for station_id in remaining_stations:
                # 检查是否可以添加这个加油站的任务
                station_load = self._calculate_station_load(station_tasks[station_id])
                
                if self._can_add_station(current_load, station_load, compartment_capacity):
                    # 计算评分（距离越近越好）
                    distance = self.data.get_distance(current_location, station_id)
                    score = 1.0 / (1.0 + distance)  # 距离越近分数越高
                    
                    if score > best_score:
                        best_score = score
                        best_station = station_id
                        
            if best_station is None:
                break  # 没有可以添加的加油站
                
            # 添加最佳加油站的任务
            station_load = self._calculate_station_load(station_tasks[best_station])
            
            # 更新载重
            for product_id, quantity in station_load.items():
                current_load[product_id] += quantity
                
            # 添加任务到路径
            route.tasks.extend(station_tasks[best_station])
            
            # 更新当前位置和时间
            distance = self.data.get_distance(current_location, best_station)
            travel_time = distance / vehicle['speed']
            unload_time = self.data.stations[best_station]['unload_time']
            current_time += travel_time + unload_time
            current_location = best_station
            
            # 移除已分配的加油站
            remaining_stations.remove(best_station)
            
            # 检查时间约束
            if current_time > self.model.working_hours - 1.0:  # 留1小时返回
                break
                
        return route
        
    def _calculate_station_load(self, tasks: List[DeliveryTask]) -> Dict[int, float]:
        """计算加油站的总载重需求"""
        load = {}
        for task in tasks:
            if task.product_id not in load:
                load[task.product_id] = 0
            load[task.product_id] += task.quantity
        return load
        
    def _can_add_station(self, current_load: Dict[int, float], 
                        station_load: Dict[int, float], 
                        compartment_capacity: float) -> bool:
        """检查是否可以添加加油站的载重"""
        
        # 计算添加后的总载重
        total_load = current_load.copy()
        for product_id, quantity in station_load.items():
            if product_id not in total_load:
                total_load[product_id] = 0
            total_load[product_id] += quantity
            
        # 检查油品种类数量（最多2种）
        non_zero_products = [pid for pid, qty in total_load.items() if qty > 0]
        if len(non_zero_products) > 2:
            return False
            
        # 检查容量约束
        if len(non_zero_products) == 1:
            # 只有一种油品，检查总容量
            total_quantity = sum(total_load.values())
            return total_quantity <= 2 * compartment_capacity
        elif len(non_zero_products) == 2:
            # 两种油品，检查是否可以分别装入两个车仓
            quantities = [total_load[pid] for pid in non_zero_products]
            return all(qty <= compartment_capacity for qty in quantities)
        else:
            return True
            
    def improve_solution(self, solution: Solution) -> Solution:
        """改进解决方案"""
        print("开始解改进...")
        
        # 使用2-opt改进每条路径
        for route in solution.routes:
            if len(route.tasks) > 3:
                route.tasks = self._two_opt_improve(route.tasks, route.start_depot)
                
        # 重新评估
        solution = self.model.evaluate_solution(solution)
        
        print(f"解改进完成: 成本={solution.total_cost:.2f}, 可行性={solution.is_feasible}")
        
        return solution
        
    def _two_opt_improve(self, tasks: List[DeliveryTask], start_depot: int) -> List[DeliveryTask]:
        """使用2-opt算法改进任务顺序"""
        if len(tasks) <= 2:
            return tasks
            
        # 按加油站分组
        station_tasks = {}
        for task in tasks:
            if task.station_id not in station_tasks:
                station_tasks[task.station_id] = []
            station_tasks[task.station_id].append(task)
            
        stations = list(station_tasks.keys())
        if len(stations) <= 2:
            return tasks
            
        best_order = stations.copy()
        best_distance = self._calculate_tour_distance(best_order, start_depot)
        
        improved = True
        while improved:
            improved = False
            for i in range(len(stations)):
                for j in range(i + 2, len(stations)):
                    # 尝试2-opt交换
                    new_order = stations.copy()
                    new_order[i:j+1] = reversed(new_order[i:j+1])
                    
                    new_distance = self._calculate_tour_distance(new_order, start_depot)
                    
                    if new_distance < best_distance:
                        best_distance = new_distance
                        best_order = new_order
                        stations = new_order
                        improved = True
                        break
                if improved:
                    break
                    
        # 重新组织任务
        improved_tasks = []
        for station_id in best_order:
            improved_tasks.extend(station_tasks[station_id])
            
        return improved_tasks
        
    def _calculate_tour_distance(self, stations: List[int], start_depot: int) -> float:
        """计算巡回距离"""
        if not stations:
            return 0.0
            
        total_distance = 0.0
        current = start_depot
        
        for station in stations:
            total_distance += self.data.get_distance(current, station)
            current = station
            
        # 返回起点
        total_distance += self.data.get_distance(current, start_depot)
        
        return total_distance
