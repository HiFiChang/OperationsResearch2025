"""
结果分析和可视化模块
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from problem_model import ProblemModel, Solution, Route
from data_loader import DataLoader

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ResultAnalyzer:
    """结果分析器"""
    
    def __init__(self, problem_model: ProblemModel, data_loader: DataLoader):
        self.model = problem_model
        self.data = data_loader
        
    def analyze_solution(self, solution: Solution):
        """分析解决方案"""
        print("\n" + "=" * 50)
        print("解决方案分析")
        print("=" * 50)
        
        # 基本信息
        print(f"\n基本信息:")
        print(f"  总成本: {solution.total_cost:,.2f}元")
        print(f"  可行性: {'是' if solution.is_feasible else '否'}")
        print(f"  路径数量: {len(solution.routes)}")
        
        if not solution.is_feasible:
            print(f"\n约束违反情况:")
            for violation in solution.violations:
                print(f"  - {violation}")
                
        # 车辆使用情况
        self._analyze_vehicle_usage(solution)
        
        # 路径分析
        self._analyze_routes(solution)
        
        # 油品配送分析
        self._analyze_product_delivery(solution)
        
        # 时间分析
        self._analyze_time_usage(solution)
        
    def _analyze_vehicle_usage(self, solution: Solution):
        """分析车辆使用情况"""
        print(f"\n车辆使用情况:")
        
        used_vehicles = set()
        vehicle_trips = {}
        vehicle_loads = {}
        
        for route in solution.routes:
            used_vehicles.add(route.vehicle_id)
            
            if route.vehicle_id not in vehicle_trips:
                vehicle_trips[route.vehicle_id] = 0
                vehicle_loads[route.vehicle_id] = 0
                
            vehicle_trips[route.vehicle_id] += 1
            
            # 计算载重
            total_load = sum(task.quantity for task in route.tasks)
            vehicle_loads[route.vehicle_id] += total_load
            
        print(f"  使用车辆数: {len(used_vehicles)}/{len(self.data.vehicles)}")
        print(f"  平均每车趟次: {np.mean(list(vehicle_trips.values())):.2f}")
        
        # 车辆类型分析
        vehicle_types = {}
        for vehicle_id in used_vehicles:
            capacity = self.data.vehicles[vehicle_id]['total_capacity']
            if capacity not in vehicle_types:
                vehicle_types[capacity] = 0
            vehicle_types[capacity] += 1
            
        print(f"  车辆类型使用:")
        for capacity, count in sorted(vehicle_types.items()):
            print(f"    {capacity:,}升容量: {count}辆")
            
    def _analyze_routes(self, solution: Solution):
        """分析路径情况"""
        print(f"\n路径分析:")
        
        route_distances = []
        route_times = []
        route_loads = []
        stations_per_route = []
        
        for route in solution.routes:
            # 计算路径距离
            distance = self._calculate_route_distance(route)
            route_distances.append(distance)
            
            # 计算路径时间
            time = self.model.calculate_route_time(route)
            route_times.append(time)
            
            # 计算载重
            load = sum(task.quantity for task in route.tasks)
            route_loads.append(load)
            
            # 计算访问的加油站数
            unique_stations = len(set(task.station_id for task in route.tasks))
            stations_per_route.append(unique_stations)
            
        if route_distances:
            print(f"  平均路径距离: {np.mean(route_distances):.2f}公里")
            print(f"  平均路径时间: {np.mean(route_times):.2f}小时")
            print(f"  平均载重: {np.mean(route_loads):,.0f}升")
            print(f"  平均访问加油站数: {np.mean(stations_per_route):.1f}")
            
            print(f"  最长路径距离: {max(route_distances):.2f}公里")
            print(f"  最长路径时间: {max(route_times):.2f}小时")
            
    def _calculate_route_distance(self, route: Route) -> float:
        """计算路径总距离"""
        if not route.tasks:
            return 0.0
            
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
            
        return total_distance
        
    def _analyze_product_delivery(self, solution: Solution):
        """分析油品配送情况"""
        print(f"\n油品配送分析:")
        
        product_delivery = {}
        for route in solution.routes:
            for task in route.tasks:
                product_id = task.product_id
                if product_id not in product_delivery:
                    product_delivery[product_id] = 0
                product_delivery[product_id] += task.quantity
                
        for product_id, total_delivery in product_delivery.items():
            product_name = self.data.products[product_id]['name']
            print(f"  {product_name}: {total_delivery:,.0f}升")
            
        # 需求满足率
        print(f"\n需求满足情况:")
        for product_id in self.data.products:
            product_name = self.data.products[product_id]['name']
            
            total_demand = 0
            for station_id in self.data.demands:
                if product_id in self.data.demands[station_id]:
                    total_demand += self.data.demands[station_id][product_id]['most_likely']
                    
            delivered = product_delivery.get(product_id, 0)
            satisfaction_rate = delivered / total_demand if total_demand > 0 else 0
            
            print(f"  {product_name}: {satisfaction_rate:.1%} ({delivered:,.0f}/{total_demand:,.0f}升)")
            
    def _analyze_time_usage(self, solution: Solution):
        """分析时间使用情况"""
        print(f"\n时间使用分析:")
        
        total_working_time = 0
        max_time = 0
        
        for route in solution.routes:
            route_time = self.model.calculate_route_time(route)
            total_working_time += route_time
            max_time = max(max_time, route_time)
            
        print(f"  总工作时间: {total_working_time:.2f}小时")
        print(f"  最长单次路径时间: {max_time:.2f}小时")
        print(f"  工作时间限制: {self.model.working_hours}小时")
        
        if max_time <= self.model.working_hours:
            print(f"  时间约束: 满足")
        else:
            print(f"  时间约束: 违反（超出{max_time - self.model.working_hours:.2f}小时）")
            
    def save_results(self, solution: Solution, filename: str):
        """保存结果到JSON文件"""
        results = {
            'summary': {
                'total_cost': solution.total_cost,
                'is_feasible': solution.is_feasible,
                'num_routes': len(solution.routes),
                'violations': solution.violations
            },
            'routes': []
        }
        
        for i, route in enumerate(solution.routes):
            route_data = {
                'route_id': i + 1,
                'vehicle_id': route.vehicle_id,
                'trip_number': route.trip_number,
                'start_depot': route.start_depot,
                'end_depot': route.end_depot,
                'start_time': route.start_time,
                'tasks': [],
                'cost': self.model.calculate_route_cost(route),
                'time': self.model.calculate_route_time(route),
                'distance': self._calculate_route_distance(route)
            }
            
            for task in route.tasks:
                task_data = {
                    'station_id': task.station_id,
                    'station_name': self.data.stations[task.station_id]['name'],
                    'product_id': task.product_id,
                    'product_name': self.data.products[task.product_id]['name'],
                    'quantity': task.quantity
                }
                route_data['tasks'].append(task_data)
                
            results['routes'].append(route_data)
            
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"\n结果已保存到: {filename}")
        
    def generate_report(self, solution: Solution, filename: str):
        """生成详细报告"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("成品油配送问题求解报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 问题概述
            f.write("1. 问题概述\n")
            f.write("-" * 20 + "\n")
            f.write(f"加油站数量: {len(self.data.stations)}\n")
            f.write(f"油库数量: {len(self.data.depots)}\n")
            f.write(f"车辆数量: {len(self.data.vehicles)}\n")
            f.write(f"油品种类: {len(self.data.products)}\n")
            
            all_tasks = self.model.get_all_delivery_requirements()
            f.write(f"配送任务数: {len(all_tasks)}\n")
            
            total_demand = sum(task.quantity for task in all_tasks)
            f.write(f"总需求量: {total_demand:,.0f}升\n\n")
            
            # 解决方案概述
            f.write("2. 解决方案概述\n")
            f.write("-" * 20 + "\n")
            f.write(f"总成本: {solution.total_cost:,.2f}元\n")
            f.write(f"可行性: {'是' if solution.is_feasible else '否'}\n")
            f.write(f"路径数量: {len(solution.routes)}\n")
            
            if not solution.is_feasible:
                f.write("\n约束违反情况:\n")
                for violation in solution.violations:
                    f.write(f"  - {violation}\n")
                    
            f.write("\n")
            
            # 详细路径信息
            f.write("3. 详细路径信息\n")
            f.write("-" * 20 + "\n")
            
            for i, route in enumerate(solution.routes):
                f.write(f"\n路径 {i+1}:\n")
                f.write(f"  车辆: {self.data.vehicles[route.vehicle_id]['name']}\n")
                f.write(f"  趟次: {route.trip_number}\n")
                f.write(f"  起点: {self.data.depots[route.start_depot]['name']}\n")
                f.write(f"  终点: {self.data.depots[route.end_depot]['name']}\n")
                f.write(f"  成本: {self.model.calculate_route_cost(route):.2f}元\n")
                f.write(f"  时间: {self.model.calculate_route_time(route):.2f}小时\n")
                f.write(f"  距离: {self._calculate_route_distance(route):.2f}公里\n")
                
                f.write("  配送任务:\n")
                for task in route.tasks:
                    station_name = self.data.stations[task.station_id]['name']
                    product_name = self.data.products[task.product_id]['name']
                    f.write(f"    {station_name} - {product_name}: {task.quantity:,.0f}升\n")
                    
        print(f"详细报告已保存到: {filename}")
        
    def create_visualization(self, solution: Solution):
        """创建可视化图表"""
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 车辆使用情况
        vehicle_usage = {}
        for route in solution.routes:
            capacity = self.data.vehicles[route.vehicle_id]['total_capacity']
            if capacity not in vehicle_usage:
                vehicle_usage[capacity] = 0
            vehicle_usage[capacity] += 1
            
        capacities = list(vehicle_usage.keys())
        counts = list(vehicle_usage.values())
        
        ax1.bar([f"{c//1000}吨" for c in capacities], counts)
        ax1.set_title('车辆类型使用情况')
        ax1.set_ylabel('使用数量')
        
        # 2. 油品配送量
        product_delivery = {}
        for route in solution.routes:
            for task in route.tasks:
                product_id = task.product_id
                if product_id not in product_delivery:
                    product_delivery[product_id] = 0
                product_delivery[product_id] += task.quantity
                
        products = [self.data.products[pid]['name'] for pid in product_delivery.keys()]
        quantities = list(product_delivery.values())
        
        ax2.pie(quantities, labels=products, autopct='%1.1f%%')
        ax2.set_title('油品配送量分布')
        
        # 3. 路径时间分布
        route_times = [self.model.calculate_route_time(route) for route in solution.routes]
        
        ax3.hist(route_times, bins=10, alpha=0.7)
        ax3.axvline(self.model.working_hours, color='red', linestyle='--', label='工作时间限制')
        ax3.set_title('路径时间分布')
        ax3.set_xlabel('时间（小时）')
        ax3.set_ylabel('路径数量')
        ax3.legend()
        
        # 4. 成本分析
        route_costs = [self.model.calculate_route_cost(route) for route in solution.routes]
        
        ax4.hist(route_costs, bins=10, alpha=0.7)
        ax4.set_title('路径成本分布')
        ax4.set_xlabel('成本（元）')
        ax4.set_ylabel('路径数量')
        
        plt.tight_layout()
        plt.savefig('solution_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("可视化图表已保存到: solution_visualization.png")
