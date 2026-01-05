import pandas as pd
import os
import torch
import gc
import sys

# 导入框架中的注册表
from unified_framework import MODEL_REGISTRY

def main():
    # 1. 配置路径
    input_file = os.path.join('data', 'test.csv')
    output_dir = 'results'
    output_file = os.path.join(output_dir, 'raw_denoising_results.csv')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. 读取数据 (仅读取 1 行进行验证)
    print(f"正在读取数据: {input_file} ...")
    try:
        df = pd.read_csv(input_file, header=None, names=['id', 'X', 'y'], nrows=1)
    except Exception as e:
        print(f"读取 CSV 失败: {e}")
        return

    # 3. 初始化并加载模型
    model_key = 'dreamcoder'
    config = MODEL_REGISTRY[model_key]
    print(f"正在加载模型 {model_key} 进行原生去噪测试...")
    
    try:
        model_instance = config["class"](config["id"])
        model_instance.load() # 加载后 model_instance.model 和 model_instance.tokenizer 即可使用
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 获取底层模型和分词器
    raw_model = model_instance.model
    tokenizer = model_instance.tokenizer

    # 4. 执行原始去噪推理
    results = []
    print(f"开始原始去噪推理...")

    for index, row in df.iterrows():
        x_val = row['X']
        y_val = row['y']
        
        print(f"[{index + 1}/{len(df)}] 正在处理 (不使用 Chat Template)...")
        
        try:
            # 1. 替换噪声占位符为模型识别的特殊 Token: <|mask|>
            x_val_processed = x_val.replace('[MASK]', '<|mask|>')

            # --- 核心修改：原始 Tokenize，不添加对话装饰 ---
            inputs = tokenizer(x_val_processed, return_tensors="pt")
            input_ids = inputs.input_ids.to(device="cuda")
            attention_mask = inputs.attention_mask.to(device="cuda")

            # 打印调试信息：确认是否包含 Mask ID (151666)
            mask_id = tokenizer.mask_token_id
            if mask_id in input_ids:
                print(f"成功识别到噪声 Token <|mask|> (ID: {mask_id})")

            # 2. 直接调用底层 diffusion_generate
            # 设置 max_new_tokens=1 以避开 transformers 的 strict 校验
            # 虽然会多生成一个 token，但主要目的（修复内部 MASK）不受影响
            output = raw_model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1, 
                steps=768,
                temperature=0.1,
                top_p=0.95,
                alg="entropy",
                alg_temp=0.,
            )
            
            # 获取序列数据
            seqs = output.sequences if hasattr(output, "sequences") else output

            # --- 关键：解码完整序列 g.tolist() 而不是 g[len(p):] ---
            # 这样我们才能看到原本 [MASK] 位置被替换成了什么
            full_out_text = tokenizer.decode(seqs[0].tolist(), skip_special_tokens=True)
            
            print("\n" + "="*20 + " 原始输入 X (含 MASK) " + "="*20)
            print(x_val)
            print("\n" + "="*20 + " 完整生成结果 (去噪后) " + "="*20)
            print(full_out_text)
            
            results.append({
                'X_original': x_val,
                'output_full': full_out_text,
                'y_ground_truth': y_val
            })
            
        except Exception as e:
            print(f"处理出错: {e}")
            
    # 5. 保存结果
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"\n推理完成！结果已保存至: {output_file}")

if __name__ == "__main__":
    main()