import pandas as pd
import os
import torch
import gc
import sys

# 确保可以导入项目中的 unified_framework
from unified_framework import MODEL_REGISTRY

def main():
    # 1. 配置路径
    input_file = os.path.join('data', 'test.csv')
    output_dir = 'results'
    output_file = os.path.join(output_dir, 'test_results.csv')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建结果目录: {output_dir}")

    # 2. 读取数据
    print(f"正在读取数据: {input_file} ...")
    try:
        # 该 CSV 文件似乎没有列名，第一列是索引，第二列是 X，第三列是 y
        df = pd.read_csv(input_file, header=None, names=['id', 'X', 'y'])
    except Exception as e:
        print(f"读取 CSV 失败: {e}")
        return

    # 检查列名 (现在已经通过 names 指定了)
    if 'X' not in df.columns or 'y' not in df.columns:
        print("错误: CSV 处理逻辑异常，未定义 'X' 或 'y'。")
        return

    # 3. 初始化模型 (参考 dreamcoder 调用方式)
    model_key = 'dreamcoder'
    if model_key not in MODEL_REGISTRY:
        print(f"错误: MODEL_REGISTRY 中未找到 '{model_key}'")
        return

    config = MODEL_REGISTRY[model_key]
    print(f"正在加载模型 {model_key} ({config['id']})...")
    
    try:
        model_instance = config["class"](config["id"])
        model_instance.load()
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 4. 执行推理
    results = []
    total_rows = len(df)
    
    print(f"开始推理，共 {total_rows} 条数据...")

    for index, row in df.iterrows():
        x_val = row['X']
        y_val = row['y']
        
        print(f"[{index + 1}/{total_rows}] 正在处理...")
        
        try:
            # 参考 unified_framework 中的 generate 调用
            # 可以根据需要调整 max_new_tokens, steps, temperature 等参数
            output = model_instance.generate(
                x_val, 
                max_new_tokens=768, 
                steps=768, 
                temperature=0.1
            )
            
            results.append({
                'X': x_val,
                'output': output,
                'y': y_val
            })
            
        except Exception as e:
            print(f"处理第 {index} 行时出错: {e}")
            results.append({
                'X': x_val,
                'output': f"ERROR: {str(e)}",
                'y': y_val
            })
            
        # 定期保存/清理显存防止 OOM
        if (index + 1) % 5 == 0:
            pd.DataFrame(results).to_csv(output_file, index=False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # 5. 保存最终结果
    final_df = pd.DataFrame(results)
    final_df.to_csv(output_file, index=False)
    print(f"\n推理完成！结果已保存至: {output_file}")

if __name__ == "__main__":
    main()