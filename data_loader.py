"""
成品油配送问题数据加载器
负责读取和预处理所有数据文件
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os

class DataLoader:
    """数据加载和预处理类"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.stations = {}
        self.depots = {}
        self.vehicles = {}
        self.products = {}
        self.demands = {}
        self.station_inventory = {}
        self.depot_inventory = {}
        self.depot_station_distances = {}
        self.station_distances = {}
        
    def load_all_data(self):
        """加载所有数据文件"""
        print("开始加载数据...")
        
        # 加载基础信息
        self._load_stations()
        self._load_depots()
        self._load_vehicles()
        self._load_products()
        
        # 加载需求和库存
        self._load_demands()
        self._load_station_inventory()
        self._load_depot_inventory()
        
        # 加载距离矩阵
        self._load_distances()
        
        # 数据验证
        self._validate_data()
        
        print("数据加载完成！")
        
    def _load_stations(self):
        """加载加油站信息"""
        df = pd.read_csv(f"{self.data_dir}/加油站信息.csv")
        for _, row in df.iterrows():
            self.stations[row['编码']] = {
                'id': row['编码'],
                'name': row['加油站名称'],
                'unload_time': row['卸油时间']
            }
        print(f"加载了 {len(self.stations)} 个加油站")
        
    def _load_depots(self):
        """加载油库信息"""
        df = pd.read_csv(f"{self.data_dir}/油库信息.csv")
        for _, row in df.iterrows():
            self.depots[row['编码']] = {
                'id': row['编码'],
                'name': row['油库名称'],
                'vehicle_count': row['油罐车数量']
            }
        print(f"加载了 {len(self.depots)} 个油库")
        
    def _load_vehicles(self):
        """加载油罐车信息"""
        df = pd.read_csv(f"{self.data_dir}/油罐车信息.csv")
        for _, row in df.iterrows():
            self.vehicles[row['编码']] = {
                'id': row['编码'],
                'name': row['名称'],
                'plate': row['车牌号'],
                'compartment1_capacity': row['车仓1（升）'],
                'compartment2_capacity': row['车仓2（升）'],
                'total_capacity': row['车仓1（升）'] + row['车仓2（升）'],
                'speed': row['车速（km/hr）'],
                'cost_per_km': row['单位距离运输成本'],
                'depot_id': 8001  # 根据README，所有车辆都属于油库A
            }
        print(f"加载了 {len(self.vehicles)} 辆油罐车")
        
    def _load_products(self):
        """加载油品信息"""
        df = pd.read_csv(f"{self.data_dir}/油品信息.csv")
        for _, row in df.iterrows():
            self.products[row['编码']] = {
                'id': row['编码'],
                'name': row['油品名称']
            }
        print(f"加载了 {len(self.products)} 种油品")
        
    def _load_demands(self):
        """加载加油站需求量"""
        df = pd.read_csv(f"{self.data_dir}/加油站需求量.csv")
        for _, row in df.iterrows():
            station_id = row['加油站编码']
            product_id = row['油品编码']
            
            if station_id not in self.demands:
                self.demands[station_id] = {}
                
            self.demands[station_id][product_id] = {
                'min_demand': row['需求量下限（升）'],
                'most_likely': row['最可能需求量（升）'],
                'max_demand': row['需求量上限（升）']
            }
        print(f"加载了 {len(self.demands)} 个加油站的需求数据")
        
    def _load_station_inventory(self):
        """加载加油站库存"""
        df = pd.read_csv(f"{self.data_dir}/加油站库存.csv")
        for _, row in df.iterrows():
            station_id = row['加油站编码']
            product_id = row['油品编码']
            tank_id = row['油罐号']
            
            if station_id not in self.station_inventory:
                self.station_inventory[station_id] = {}
            if product_id not in self.station_inventory[station_id]:
                self.station_inventory[station_id][product_id] = {
                    'tanks': {},
                    'total_capacity': 0,
                    'total_inventory': 0
                }
                
            self.station_inventory[station_id][product_id]['tanks'][tank_id] = {
                'capacity': row['罐容'],
                'inventory': row['库存（升）']
            }
            self.station_inventory[station_id][product_id]['total_capacity'] += row['罐容']
            self.station_inventory[station_id][product_id]['total_inventory'] += row['库存（升）']
            
        print(f"加载了 {len(self.station_inventory)} 个加油站的库存数据")
        
    def _load_depot_inventory(self):
        """加载油库库存"""
        df = pd.read_csv(f"{self.data_dir}/油库库存.csv")
        for _, row in df.iterrows():
            depot_id = row['油库编码']
            product_id = row['油品编码']
            
            if depot_id not in self.depot_inventory:
                self.depot_inventory[depot_id] = {}
                
            self.depot_inventory[depot_id][product_id] = row['库存（升）']
            
        print(f"加载了 {len(self.depot_inventory)} 个油库的库存数据")
        
    def _load_distances(self):
        """加载距离矩阵"""
        # 加载库站运距
        df_depot_station = pd.read_csv(f"{self.data_dir}/库站运距.csv")
        for _, row in df_depot_station.iterrows():
            depot_id = row['油库编码']
            station_id = row['加油站编码']
            distance = row['运距']
            
            if depot_id not in self.depot_station_distances:
                self.depot_station_distances[depot_id] = {}
            self.depot_station_distances[depot_id][station_id] = distance
            
        # 加载站站运距
        df_station_station = pd.read_csv(f"{self.data_dir}/站站运距.csv")
        for _, row in df_station_station.iterrows():
            station1_id = row['加油站1编码']
            station2_id = row['加油站2编码']
            distance = row['运距']
            
            if station1_id not in self.station_distances:
                self.station_distances[station1_id] = {}
            if station2_id not in self.station_distances:
                self.station_distances[station2_id] = {}
                
            self.station_distances[station1_id][station2_id] = distance
            self.station_distances[station2_id][station1_id] = distance
            
        print(f"加载了距离矩阵数据")
        
    def _validate_data(self):
        """验证数据完整性"""
        print("\n数据验证:")
        
        # 验证需求量是否超过油库库存
        total_demand = {}
        for station_id, station_demands in self.demands.items():
            for product_id, demand_info in station_demands.items():
                if product_id not in total_demand:
                    total_demand[product_id] = 0
                total_demand[product_id] += demand_info['most_likely']
                
        total_supply = {}
        for depot_id, depot_inv in self.depot_inventory.items():
            for product_id, inventory in depot_inv.items():
                if product_id not in total_supply:
                    total_supply[product_id] = 0
                total_supply[product_id] += inventory
                
        print("供需平衡检查:")
        for product_id in total_demand:
            product_name = self.products[product_id]['name']
            demand = total_demand[product_id]
            supply = total_supply.get(product_id, 0)
            ratio = supply / demand if demand > 0 else float('inf')
            print(f"  {product_name}: 需求={demand:,.0f}升, 供应={supply:,.0f}升, 比例={ratio:.2f}")
            
        # 验证车辆容量
        vehicle_capacities = [v['total_capacity'] for v in self.vehicles.values()]
        print(f"\n车辆容量分布:")
        print(f"  最小容量: {min(vehicle_capacities):,}升")
        print(f"  最大容量: {max(vehicle_capacities):,}升")
        print(f"  平均容量: {np.mean(vehicle_capacities):,.0f}升")
        
    def get_distance(self, from_node: int, to_node: int) -> float:
        """获取两点间距离"""
        # 如果是同一个节点
        if from_node == to_node:
            return 0.0

        # 如果是从油库到加油站
        if from_node in self.depots and to_node in self.stations:
            distance = self.depot_station_distances.get(from_node, {}).get(to_node, None)
            if distance is not None:
                return distance

        # 如果是从加油站到油库
        elif from_node in self.stations and to_node in self.depots:
            distance = self.depot_station_distances.get(to_node, {}).get(from_node, None)
            if distance is not None:
                return distance

        # 如果是加油站之间
        elif from_node in self.stations and to_node in self.stations:
            distance = self.station_distances.get(from_node, {}).get(to_node, None)
            if distance is not None:
                return distance
            # 如果没有直接连接，认为不可通行
            else:
                return float('inf')

        # 油库之间的距离（假设为0，因为题目中没有给出）
        elif from_node in self.depots and to_node in self.depots:
            return 0.0

        # 其他情况返回无穷大
        return float('inf')
            
    def get_expected_demand(self, station_id: int, product_id: int) -> float:
        """获取期望需求量（使用最可能值）"""
        return self.demands.get(station_id, {}).get(product_id, {}).get('most_likely', 0)
    
    def get_adjustable_demand(self, station_id: int, product_id: int, available_capacity: float = None) -> float:
        """获取可调整的需求量，考虑上下限和油罐车容量"""
        demand_info = self.demands.get(station_id, {}).get(product_id, {})
        if not demand_info:
            return 0
        
        min_demand = demand_info['min_demand']
        most_likely = demand_info['most_likely']
        max_demand = demand_info['max_demand']
        
        # 获取加油站该油品的可接收容量
        station_capacity = self.get_station_available_capacity(station_id, product_id)
        
        # 调整需求量：不能超过油罐可接收容量
        adjusted_demand = min(most_likely, station_capacity)
        
        # 如果指定了车辆容量，可以在合理范围内浮动
        if available_capacity is not None:
            # 在不超过上限和可接收容量的前提下，尽量利用车辆容量
            max_deliverable = min(max_demand, station_capacity, available_capacity)
            min_deliverable = max(min_demand, 0)
            
            # 选择一个合理的配送量
            adjusted_demand = min(max_deliverable, max(min_deliverable, adjusted_demand))
        
        return adjusted_demand
    
    def get_demand_range(self, station_id: int, product_id: int) -> Tuple[float, float, float]:
        """获取需求量范围（下限，最可能值，上限）"""
        demand_info = self.demands.get(station_id, {}).get(product_id, {})
        if not demand_info:
            return 0, 0, 0
        return demand_info['min_demand'], demand_info['most_likely'], demand_info['max_demand']
        
    def get_station_available_capacity(self, station_id: int, product_id: int) -> float:
        """获取加油站可接收容量"""
        if station_id not in self.station_inventory or product_id not in self.station_inventory[station_id]:
            return 0
        
        inv_info = self.station_inventory[station_id][product_id]
        return inv_info['total_capacity'] - inv_info['total_inventory']

if __name__ == "__main__":
    # 测试数据加载
    loader = DataLoader()
    loader.load_all_data()
