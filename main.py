"""
成品油配送问题求解主程序
"""

import time
from data_loader import DataLoader
from problem_model import ProblemModel
from solver import TwoStageSolver  # 导入新的求解器
from result_analyzer import ResultAnalyzer


def main():
    """主函数"""
    start_time = time.time()
    
    print("=" * 60)
    print("成品油配送问题求解系统 (两阶段启发式算法)")
    print("=" * 60)
    
    # 1. 数据加载
    print("\n1. 数据加载阶段")
    data_loader = DataLoader()
    data_loader.load_all_data()
    print(f"数据加载完成，耗时: {time.time() - start_time:.2f}秒")
    
    # 2. 问题建模 (作为辅助工具类)
    print("\n2. 问题建模阶段")
    problem_model = ProblemModel(data_loader)
    print("问题模型初始化完成。")

    # 3. 算法求解
    print("\n3. 算法求解阶段")
    solver = TwoStageSolver(data_loader, problem_model)
    best_solution, time_windows = solver.solve()
    
    if best_solution:
        # 4. 结果分析与输出
        print("\n4. 结果分析阶段")
        analyzer = ResultAnalyzer(problem_model, data_loader)
        analyzer.analyze_solution(best_solution, time_windows)
        
        # 保存结果
        analyzer.save_results(best_solution, "solution_results.json")
    else:
        print("\n未能找到解决方案。")

    end_time = time.time()
    print(f"\n程序总耗时: {end_time - start_time:.2f}秒")

if __name__ == "__main__":
    main()
