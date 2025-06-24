"""
成品油配送问题数学建模
定义问题的约束条件、目标函数和解的表示
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from data_loader import DataLoader

@dataclass
class DeliveryTask:
    """配送任务"""
    station_id: int
    product_id: int
    quantity: float
    tank_id: Optional[int] = None # 对应的油罐ID
    
@dataclass
class Route:
    """车辆路径"""
    vehicle_id: int
    trip_number: int  # 1 或 2
    start_depot: int
    end_depot: int
    tasks: List[DeliveryTask]
    start_time: float = 8.0  # 开始时间（小时）
    is_feasible: bool = True
    violations: List[str] = field(default_factory=list)

@dataclass
class Solution:
    """解决方案"""
    routes: List[Route]
    total_cost: float = 0.0
    is_feasible: bool = True
    violations: List[str] = None

class ProblemModel:
    """
    问题建模与评估器.
    该类现在主要作为辅助类，提供成本计算和约束检查功能。
    """
    
    def __init__(self, data_loader: DataLoader):
        self.data = data_loader
        self.working_hours = 9.0  # 8:00-17:00
        self.max_trips_per_vehicle = 2
        
    def calculate_route_cost(self, route: Route) -> float:
        """计算路径成本"""
        if not route.tasks:
            return 0.0
            
        vehicle = self.data.vehicles[route.vehicle_id]
        cost_per_km = vehicle['cost_per_km']
        
        total_distance = 0.0
        current_location = route.start_depot
        
        # 访问每个加油站
        visited_stations = []
        for task in route.tasks:
            if task.station_id not in visited_stations:
                distance = self.data.get_distance(current_location, task.station_id)
                total_distance += distance
                current_location = task.station_id
                visited_stations.append(task.station_id)
        
        # 返回终点油库
        if current_location != route.end_depot:
            distance = self.data.get_distance(current_location, route.end_depot)
            total_distance += distance
            
        return total_distance * cost_per_km
        
    def calculate_route_time(self, route: Route) -> float:
        """计算路径总时间"""
        if not route.tasks:
            return 0.0
            
        vehicle = self.data.vehicles[route.vehicle_id]
        speed = vehicle['speed']
        
        total_time = 0.0
        current_location = route.start_depot
        
        # 访问每个加油站
        visited_stations = []
        for task in route.tasks:
            if task.station_id not in visited_stations:
                # 行驶时间
                distance = self.data.get_distance(current_location, task.station_id)
                travel_time = distance / speed
                total_time += travel_time
                
                # 卸油时间
                unload_time = self.data.stations[task.station_id]['unload_time']
                total_time += unload_time
                
                current_location = task.station_id
                visited_stations.append(task.station_id)
        
        # 返回终点油库的时间
        if current_location != route.end_depot:
            distance = self.data.get_distance(current_location, route.end_depot)
            travel_time = distance / speed
            total_time += travel_time
            
        return total_time
        
    def check_capacity_constraint(self, route: Route) -> Tuple[bool, str]:
        """检查容量约束，考虑"一仓一罐"的强约束"""
        vehicle = self.data.vehicles[route.vehicle_id]
        compartment1_capacity = vehicle['compartment1_capacity']
        compartment2_capacity = vehicle['compartment2_capacity']
        
        # 按油品分组任务
        product_quantities = {}
        for task in route.tasks:
            if task.product_id not in product_quantities:
                product_quantities[task.product_id] = 0
            product_quantities[task.product_id] += task.quantity
        
        # 检查油品种类是否超过2种
        if len(product_quantities) > 2:
            return False, f"车辆不能同时运输{len(product_quantities)}种油品（最多2种）"

        # 检查每种油品的总量是否能装入一个仓
        quantities = list(product_quantities.values())
        if len(quantities) == 1:
            if quantities[0] > compartment1_capacity + compartment2_capacity:
                 return False, f"单一油品数量{quantities[0]}超过车辆总容量{compartment1_capacity + compartment2_capacity}"
        elif len(quantities) == 2:
            q1, q2 = quantities[0], quantities[1]
            if not ((q1 <= compartment1_capacity and q2 <= compartment2_capacity) or \
                    (q1 <= compartment2_capacity and q2 <= compartment1_capacity)):
                return False, f"两种油品数量{[q1, q2]}无法分别装入车仓{[compartment1_capacity, compartment2_capacity]}"

        # 检查"一仓一罐"约束：统计路径中独立的(加油站, 油罐)对的数量
        # 注意：这里的tank_id可能为None，我们将其视为一个独立的标识符
        unique_tanks_served = set()
        for task in route.tasks:
            unique_tanks_served.add((task.station_id, task.tank_id))
        
        num_independent_deliveries = len(unique_tanks_served)

        # 车辆只有两个仓，最多只能服务两个独立的油罐
        if num_independent_deliveries > 2:
            return False, f"路径服务了 {num_independent_deliveries} 个独立的油罐，超过了车辆的2个车仓限制。"

        return True, ""
            
    def check_time_constraint(self, route: Route, time_windows: Dict, existing_schedules: Dict = None) -> Tuple[bool, str]:
        """
        检查时间约束 (现在从外部接收时间窗)
        新增: 接收一个可选的、已存在的站点日程表，以检查同时服务冲突
        """
        route_time = self.calculate_route_time(route)
        
        if route_time > self.working_hours:
            return False, f"路径时间{route_time:.2f}小时超过工作时间{self.working_hours}小时"
            
        # 检查时间窗口约束
        current_time = route.start_time
        vehicle = self.data.vehicles[route.vehicle_id]
        speed = vehicle['speed']
        current_location = route.start_depot
        
        visited_stations = []
        for task in route.tasks:
            # 使用任务中携带的时间窗信息
            task_key = (task.station_id, task.product_id, task.tank_id)
            if task_key not in time_windows:
                 return False, f"任务 {task_key} 没有找到对应的时间窗信息"

            tw = time_windows[task_key]
            earliest = tw['earliest']
            latest = tw['latest']

            if task.station_id not in visited_stations:
                # 到达时间
                distance = self.data.get_distance(current_location, task.station_id)
                travel_time = distance / speed
                arrival_time = current_time + travel_time
                
                if arrival_time < earliest:
                    # 允许等待
                    current_time = earliest
                else:
                    current_time = arrival_time
                    
                if current_time > latest:
                    return False, f"到达加油站{task.station_id}时间{current_time:.2f}晚于最晚时间{latest:.2f}"
                
                # 计算并检查与现有日程的冲突
                unload_time = self.data.stations[task.station_id]['unload_time']
                service_start_time = current_time
                service_end_time = service_start_time + unload_time

                if existing_schedules and task.station_id in existing_schedules:
                    for start, end, v_id, trip in existing_schedules[task.station_id]:
                        # 检查新时段 [service_start_time, service_end_time] 是否与 [start, end] 重叠
                        if max(service_start_time, start) < min(service_end_time, end):
                            return False, f"与车辆{v_id}(趟次{trip})在加油站{task.station_id}发生时间冲突"

                # 更新当前时间和位置
                current_time = service_end_time
                current_location = task.station_id
                visited_stations.append(task.station_id)
                
        return True, ""
        
    def check_inventory_constraint(self, solution: Solution) -> Tuple[bool, str]:
        """检查库存约束"""
        # 统计每个油库每种油品的总配送量
        depot_usage = {}
        for route in solution.routes:
            start_depot = route.start_depot
            if start_depot not in depot_usage:
                depot_usage[start_depot] = {}
                
            for task in route.tasks:
                product_id = task.product_id
                if product_id not in depot_usage[start_depot]:
                    depot_usage[start_depot][product_id] = 0
                depot_usage[start_depot][product_id] += task.quantity
                
        # 检查是否超过油库库存
        for depot_id, product_usage in depot_usage.items():
            for product_id, total_usage in product_usage.items():
                available = self.data.depot_inventory.get(depot_id, {}).get(product_id, 0)
                if total_usage > available:
                    product_name = self.data.products[product_id]['name']
                    depot_name = self.data.depots[depot_id]['name']
                    return False, f"{depot_name}的{product_name}库存不足：需要{total_usage:.0f}升，可用{available:.0f}升"
                    
        return True, ""
        
    def check_simultaneous_delivery_constraint(self, solution: Solution) -> Tuple[bool, str]:
        """检查同时配送约束：每个加油站在同一时间只能由一辆车提供配送服务"""
        # 收集所有路径的时间段和访问的加油站
        station_schedules = {}  # {station_id: [(start_time, end_time, vehicle_id, trip)]}
        
        for route in solution.routes:
            if not route.tasks:
                continue
                
            vehicle = self.data.vehicles[route.vehicle_id]
            speed = vehicle['speed']
            current_time = route.start_time
            current_location = route.start_depot
            
            visited_stations = []
            for task in route.tasks:
                if task.station_id not in visited_stations:
                    # 计算到达时间
                    distance = self.data.get_distance(current_location, task.station_id)
                    travel_time = distance / speed
                    arrival_time = current_time + travel_time
                    
                    # 计算离开时间
                    unload_time = self.data.stations[task.station_id]['unload_time']
                    departure_time = arrival_time + unload_time
                    
                    # 记录该站点的占用时间段
                    if task.station_id not in station_schedules:
                        station_schedules[task.station_id] = []
                    station_schedules[task.station_id].append((arrival_time, departure_time, route.vehicle_id, route.trip_number))
                    
                    # 更新当前时间和位置
                    current_time = departure_time
                    current_location = task.station_id
                    visited_stations.append(task.station_id)
        
        # 检查每个加油站是否有时间冲突
        for station_id, schedules in station_schedules.items():
            if len(schedules) <= 1:
                continue
                
            # 按开始时间排序
            schedules.sort(key=lambda x: x[0])
            
            for i in range(len(schedules) - 1):
                end_time1 = schedules[i][1]
                start_time2 = schedules[i + 1][0]
                
                # 检查是否有重叠
                if end_time1 > start_time2:
                    vehicle1 = schedules[i][2]
                    trip1 = schedules[i][3]
                    vehicle2 = schedules[i + 1][2]
                    trip2 = schedules[i + 1][3]
                    return False, f"加油站{station_id}在时间段{start_time2:.2f}-{end_time1:.2f}被车辆{vehicle1}(趟次{trip1})和车辆{vehicle2}(趟次{trip2})同时占用"
        
        return True, ""
        
    def check_demand_range_constraint(self, solution: Solution, time_windows: Dict) -> Tuple[bool, str]:
        """
        检查配送量是否在合理的需求区间内
        """
        violations = []
        
        for route in solution.routes:
            for task in route.tasks:
                task_key = (task.station_id, task.product_id, task.tank_id)
                
                if task_key in time_windows:
                    tw_info = time_windows[task_key]
                    min_qty = tw_info.get('min_quantity', 0)
                    max_qty = tw_info.get('max_quantity', float('inf'))
                    
                    if task.quantity < min_qty:
                        violations.append(
                            f"任务{task_key}配送量{task.quantity:.0f}升低于最小需求{min_qty:.0f}升"
                        )
                    elif task.quantity > max_qty:
                        violations.append(
                            f"任务{task_key}配送量{task.quantity:.0f}升超过最大需求{max_qty:.0f}升"
                        )
        
        if violations:
            return False, "; ".join(violations)
        return True, ""
        
    def check_route_depot_constraint(self, solution: Solution) -> Tuple[bool, str]:
        """
        检查配送车最终返回油库A的约束
        根据实验要求：配送车一天的配送完成后最终返回油库A
        """
        for route in solution.routes:
            # 检查每辆车的最后一趟是否返回油库A
            vehicle_routes = [r for r in solution.routes if r.vehicle_id == route.vehicle_id]
            if not vehicle_routes:
                continue
                
            # 找到该车辆的最后一趟配送
            last_trip = max(vehicle_routes, key=lambda r: r.trip_number)
            if last_trip.end_depot != 8001:  # 8001是油库A的编码
                return False, f"车辆{route.vehicle_id}的最后一趟配送未返回油库A（当前终点：{last_trip.end_depot}）"
                
        return True, ""
        
    def evaluate_solution(self, solution: Solution, time_windows: Dict, total_tasks_generated: int) -> Solution:
        """
        评估解决方案 (v2.0)
        - 新增: 检查所有任务是否都被服务
        - 移除冗余的同时服务检查
        """
        total_cost = 0.0
        violations = []
        is_feasible = True
        
        # 计算总成本
        for route in solution.routes:
            total_cost += self.calculate_route_cost(route)
            
        # 检查约束
        for route in solution.routes:
            # 容量约束
            feasible, msg = self.check_capacity_constraint(route)
            if not feasible:
                is_feasible = False
                violations.append(f"路径{route.vehicle_id}-{route.trip_number}: {msg}")
                
            # 时间约束 (注意：这里的检查是事后验证，不包含动态的冲突规避)
            feasible, msg = self.check_time_constraint(route, time_windows)
            if not feasible:
                is_feasible = False
                violations.append(f"路径{route.vehicle_id}-{route.trip_number}: {msg}")
                
        # 1. 检查所有任务是否都已分配
        num_tasks_in_solution = sum(len(r.tasks) for r in solution.routes)
        if num_tasks_in_solution < total_tasks_generated:
            is_feasible = False
            violations.append(f"任务未全部完成: {num_tasks_in_solution}/{total_tasks_generated}")

        # 2. 库存约束
        feasible, msg = self.check_inventory_constraint(solution)
        if not feasible:
            is_feasible = False
            violations.append(msg)
            
        # 3. 同时配送约束 (使用最终方案重新检查)
        feasible, msg = self.check_simultaneous_delivery_constraint(solution)
        if not feasible:
            is_feasible = False
            violations.append(msg)
            
        # 4. 需求量区间约束
        feasible, msg = self.check_demand_range_constraint(solution, time_windows)
        if not feasible:
            is_feasible = False
            violations.append(msg)
            
        # 5. 最终返回油库A的约束
        feasible, msg = self.check_route_depot_constraint(solution)
        if not feasible:
            is_feasible = False
            violations.append(msg)
            
        solution.total_cost = total_cost
        solution.is_feasible = is_feasible
        solution.violations = violations
        
        return solution
        
    def create_empty_solution(self) -> Solution:
        """创建空解决方案"""
        return Solution(routes=[], violations=[])
