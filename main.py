"""
成品油配送问题求解主程序
"""

import time
import json
from data_loader import DataLoader
from problem_model import ProblemModel
from genetic_algorithm import GeneticAlgorithm
from result_analyzer import ResultAnalyzer


def main():
    """主函数"""
    print("=" * 60)
    print("成品油配送问题求解系统")
    print("=" * 60)
    
    # 1. 数据加载
    print("\n1. 数据加载阶段")
    start_time = time.time()
    
    data_loader = DataLoader()
    data_loader.load_all_data()
    
    load_time = time.time() - start_time
    print(f"数据加载完成，耗时: {load_time:.2f}秒")
    
    # 2. 问题建模
    print("\n2. 问题建模阶段")
    start_time = time.time()
    
    problem_model = ProblemModel(data_loader)
    
    model_time = time.time() - start_time
    print(f"问题建模完成，耗时: {model_time:.2f}秒")
    
    # 显示问题规模
    print(f"\n问题规模:")
    print(f"  加油站数量: {len(data_loader.stations)}")
    print(f"  油库数量: {len(data_loader.depots)}")
    print(f"  车辆数量: {len(data_loader.vehicles)}")
    print(f"  油品种类: {len(data_loader.products)}")
    
    all_tasks = problem_model.get_all_delivery_requirements()
    print(f"  配送任务数: {len(all_tasks)}")
    
    total_demand = sum(task.quantity for task in all_tasks)
    print(f"  总需求量: {total_demand:,.0f}升")
    
    # 3. 算法求解
    print("\n3. 算法求解阶段")
    start_time = time.time()
    
    # 创建遗传算法求解器
    ga = GeneticAlgorithm(
        problem_model=problem_model,
        population_size=100,
        generations=3000,
        crossover_rate=0.8,
        mutation_rate=0.2,
        elite_size=10
    )
    
    # 求解
    best_solution = ga.solve()
    
    solve_time = time.time() - start_time
    print(f"算法求解完成，耗时: {solve_time:.2f}秒")
    
    # 4. 结果分析
    print("\n4. 结果分析阶段")
    start_time = time.time()
    
    analyzer = ResultAnalyzer(problem_model, data_loader)
    analyzer.analyze_solution(best_solution)
    
    # 保存结果
    analyzer.save_results(best_solution, "solution_results.json")
    analyzer.generate_report(best_solution, "solution_report.txt")

    # 5. 可视化（跳过图形输出）
    print("\n5. 可视化阶段")
    print("在无图形环境下跳过可视化，结果已保存到JSON和文本文件")

    analysis_time = time.time() - start_time
    print(f"结果分析完成，耗时: {analysis_time:.2f}秒")
    
    # 总结
    total_time = load_time + model_time + solve_time + analysis_time
    print(f"\n总耗时: {total_time:.2f}秒")
    
    print("\n" + "=" * 60)
    print("求解完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
