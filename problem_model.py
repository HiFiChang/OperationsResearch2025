"""
成品油配送问题数学建模
定义问题的约束条件、目标函数和解的表示
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from data_loader import DataLoader

@dataclass
class DeliveryTask:
    """配送任务"""
    station_id: int
    product_id: int
    quantity: float
    tank_id: int = None
    
@dataclass
class Route:
    """车辆路径"""
    vehicle_id: int
    trip_number: int  # 1 或 2
    start_depot: int
    end_depot: int
    tasks: List[DeliveryTask]
    start_time: float = 8.0  # 开始时间（小时）
    
@dataclass
class Solution:
    """解决方案"""
    routes: List[Route]
    total_cost: float = 0.0
    is_feasible: bool = True
    violations: List[str] = None

class ProblemModel:
    """问题建模类"""
    
    def __init__(self, data_loader: DataLoader):
        self.data = data_loader
        self.working_hours = 9.0  # 8:00-17:00
        self.max_trips_per_vehicle = 2
        
        # 计算时间窗口
        self._calculate_time_windows()
        
    def _calculate_time_windows(self):
        """计算每个加油站每种油品的时间窗口"""
        self.time_windows = {}

        for station_id in self.data.stations:
            self.time_windows[station_id] = {}

            for product_id in self.data.products:
                if station_id in self.data.demands and product_id in self.data.demands[station_id]:
                    # 获取需求量和当前库存
                    daily_demand = self.data.get_expected_demand(station_id, product_id)
                    current_inventory = self.data.station_inventory.get(station_id, {}).get(product_id, {}).get('total_inventory', 0)
                    total_capacity = self.data.station_inventory.get(station_id, {}).get(product_id, {}).get('total_capacity', 0)

                    if daily_demand > 0:
                        # 计算消耗速率（升/小时）
                        consumption_rate = daily_demand / 24.0
                        
                        # 计算断油时间点（库存量归零的时间）
                        if current_inventory > 0 and consumption_rate > 0:
                            stockout_time = min(current_inventory / consumption_rate, self.working_hours)
                        else:
                            stockout_time = 0.0  # 立即断油
                        
                        # 计算不容纳时间点（油罐容量限制无法继续接收的时间）
                        available_capacity = total_capacity - current_inventory
                        if available_capacity > 0 and consumption_rate > 0:
                            # 当前库存继续消耗，什么时候能腾出足够空间接收配送
                            no_accept_time = max(0, (current_inventory + daily_demand - total_capacity) / consumption_rate)
                        else:
                            no_accept_time = self.working_hours  # 无法接收
                        
                        # 配送时间窗口：不能早于不容纳时间点，不能晚于断油时间点
                        earliest_time = max(0, no_accept_time)
                        latest_time = min(stockout_time, self.working_hours)
                        
                        # 如果时间窗口不合理，调整为宽松窗口
                        if earliest_time >= latest_time:
                            earliest_time = 0.0
                            latest_time = self.working_hours

                        self.time_windows[station_id][product_id] = {
                            'earliest': earliest_time,
                            'latest': latest_time,
                            'stockout_time': stockout_time,
                            'no_accept_time': no_accept_time
                        }
                    else:
                        # 没有需求的情况
                        self.time_windows[station_id][product_id] = {
                            'earliest': 0,
                            'latest': self.working_hours,
                            'stockout_time': float('inf'),
                            'no_accept_time': 0
                        }
                else:
                    # 没有需求数据的情况
                    self.time_windows[station_id][product_id] = {
                        'earliest': 0,
                        'latest': self.working_hours,
                        'stockout_time': float('inf'),
                        'no_accept_time': 0
                    }
                        
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
            
    def check_time_constraint(self, route: Route) -> Tuple[bool, str]:
        """检查时间约束"""
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
            if task.station_id not in visited_stations:
                # 到达时间
                distance = self.data.get_distance(current_location, task.station_id)
                travel_time = distance / speed
                arrival_time = current_time + travel_time
                
                # 检查时间窗口
                time_window = self.time_windows.get(task.station_id, {}).get(task.product_id, {})
                earliest = time_window.get('earliest', 0)
                latest = time_window.get('latest', self.working_hours)
                
                if arrival_time < earliest:
                    return False, f"到达加油站{task.station_id}时间{arrival_time:.2f}早于最早时间{earliest:.2f}"
                if arrival_time > latest:
                    return False, f"到达加油站{task.station_id}时间{arrival_time:.2f}晚于最晚时间{latest:.2f}"
                
                # 更新当前时间和位置
                unload_time = self.data.stations[task.station_id]['unload_time']
                current_time = arrival_time + unload_time
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
        
    def evaluate_solution(self, solution: Solution) -> Solution:
        """评估解决方案"""
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
                
            # 时间约束
            feasible, msg = self.check_time_constraint(route)
            if not feasible:
                is_feasible = False
                violations.append(f"路径{route.vehicle_id}-{route.trip_number}: {msg}")
                
        # 库存约束
        feasible, msg = self.check_inventory_constraint(solution)
        if not feasible:
            is_feasible = False
            violations.append(msg)
            
        # 同时配送约束
        feasible, msg = self.check_simultaneous_delivery_constraint(solution)
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
        
    def get_all_delivery_requirements(self) -> List[DeliveryTask]:
        """获取所有配送需求"""
        tasks = []
        for station_id, station_demands in self.data.demands.items():
            for product_id, demand_info in station_demands.items():
                quantity = demand_info['most_likely']
                if quantity > 0:
                    # 为每种油品的需求分配到具体的油罐
                    allocated_tasks = self._allocate_demand_to_tanks(station_id, product_id, quantity)
                    tasks.extend(allocated_tasks)
        return tasks
    
    def _allocate_demand_to_tanks(self, station_id: int, product_id: int, total_quantity: float) -> List[DeliveryTask]:
        """将需求量分配到具体的油罐"""
        tasks = []
        
        # 获取该加油站该油品的所有油罐
        tank_info = self.data.station_inventory.get(station_id, {}).get(product_id, {}).get('tanks', {})
        
        if not tank_info:
            # 如果没有油罐信息，创建一个默认任务
            tasks.append(DeliveryTask(
                station_id=station_id,
                product_id=product_id,
                quantity=total_quantity,
                tank_id=None
            ))
            return tasks
        
        # 按油罐可接收容量排序（优先给空余容量大的油罐配送）
        sorted_tanks = []
        for tank_id, tank_data in tank_info.items():
            capacity = tank_data['capacity']
            inventory = tank_data['inventory']
            available_capacity = capacity - inventory
            sorted_tanks.append((tank_id, available_capacity, capacity))
        
        # 按可接收容量降序排序
        sorted_tanks.sort(key=lambda x: x[1], reverse=True)
        
        remaining_quantity = total_quantity
        
        for tank_id, available_capacity, tank_capacity in sorted_tanks:
            if remaining_quantity <= 0:
                break
                
            # 该油罐能接收的最大配送量
            deliverable = min(remaining_quantity, available_capacity)
            
            if deliverable > 0:
                tasks.append(DeliveryTask(
                    station_id=station_id,
                    product_id=product_id,
                    quantity=deliverable,
                    tank_id=tank_id
                ))
                remaining_quantity -= deliverable
        
        # 如果还有剩余需求无法分配，创建一个任务记录
        if remaining_quantity > 0:
            tasks.append(DeliveryTask(
                station_id=station_id,
                product_id=product_id,
                quantity=remaining_quantity,
                tank_id=None  # 表示无法分配到具体油罐
            ))
        
        return tasks
