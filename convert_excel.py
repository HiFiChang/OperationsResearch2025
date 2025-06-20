import pandas as pd
import os

def convert_excel_to_csv():
    """将Excel文件转换为CSV格式，每个sheet保存为单独的CSV文件"""
    
    # 创建data目录
    if not os.path.exists('data'):
        os.makedirs('data')
        print("创建了data目录")
    
    # 读取Excel文件
    excel_file = '成品油配送数据-V2.0.xlsx'
    
    try:
        # 读取所有sheet
        excel_data = pd.read_excel(excel_file, sheet_name=None)
        
        print(f"Excel文件包含 {len(excel_data)} 个sheet:")
        
        # 遍历每个sheet并保存为CSV
        for sheet_name, df in excel_data.items():
            # 清理sheet名称，移除特殊字符
            clean_name = sheet_name.replace('/', '_').replace('\\', '_').replace(':', '_')
            csv_filename = f'data/{clean_name}.csv'
            
            # 保存为CSV
            df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            print(f"  - {sheet_name} -> {csv_filename} (形状: {df.shape})")
            
            # 显示前几行数据预览
            if not df.empty:
                print(f"    前5行预览:")
                print(df.head().to_string(index=False))
                print()
        
        print("转换完成！")
        
    except Exception as e:
        print(f"转换过程中出现错误: {e}")

if __name__ == "__main__":
    convert_excel_to_csv()
