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
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. 读取数据 (可以增加 nrows 处理更多数据进行测试)
    print(f"正在读取数据: {input_file} ...")
    try:
        # 默认读取 5 行进行验证，您可以根据需要调整
        df = pd.read_csv(input_file, header=None, names=['id', 'X', 'y'], nrows=5)
    except Exception as e:
        print(f"读取 CSV 失败: {e}")
        return

    # 3. 待测试的模型列表
    # 用户明确要求测试 DiffuCoder, LLADA 和 DreamCoder
    models_to_test = ['diffucoder', 'llada', 'dreamcoder']
    
    for model_key in models_to_test:
        if model_key not in MODEL_REGISTRY:
            print(f"\n跳过 {model_key}: 不在 MODEL_REGISTRY 中")
            continue
            
        print(f"\n{'='*20} 正在加载模型: {model_key} {'='*20}")
        
        config = MODEL_REGISTRY[model_key]
        try:
            model_instance = config["class"](config["id"])
            model_instance.load()
        except Exception as e:
            print(f"加载模型 {model_key} 失败: {e}")
            continue

        # 获取底层模型和分词器
        raw_model = model_instance.model
        tokenizer = model_instance.tokenizer

        # 4. 执行推理
        results = []
        print(f"开始为 {model_key} 执行去噪/生成测试...")

        for index, row in df.iterrows():
            x_val = str(row['X'])
            y_val = str(row['y'])
            
            print(f"[{model_key}] 正在处理 ({index + 1}/{len(df)})...")
            
            try:
                if model_key == 'llada':
                    # LLADA 特殊逻辑
                    # 默认使用 126336 作为 mask_id
                    mask_id = 126336
                    # 将 [MASK] 替换为 LLADA 识别的 mask token id
                    # LLADA 官方 generate 会在 prompt 后面接 gen_length 个 mask
                    # 如果要在 prompt 内部填充，需要手动构造 input_ids 并调用 generate
                    
                    # 尝试从 unified_framework 获取已导入的 llada_generate
                    from unified_framework import llada_generate
                    
                    if llada_generate is None:
                        # 如果框架里没导入成功，尝试手动处理路径
                        llada_path = os.path.join(os.path.dirname(__file__), 'external_repos', 'LLaDA')
                        if os.path.exists(llada_path) and llada_path not in sys.path:
                            sys.path.append(llada_path)
                        import generate as llada_gen_mod
                        llada_generate_func = llada_gen_mod.generate
                    else:
                        llada_generate_func = llada_generate

                    # 构造包含 mask 的 input_ids
                    # llada 的 tokenizer.decode([mask_id]) 应该是 [MASK] 或者类似的
                    # 我们直接用 tokenizer 处理
                    input_text = x_val.replace('[MASK]', tokenizer.decode([mask_id]))
                    inputs = tokenizer(input_text, return_tensors="pt")
                    input_ids = inputs.input_ids.to('cuda')
                    
                    output = llada_generate_func(
                        raw_model, 
                        input_ids, 
                        steps=128, 
                        gen_length=0, # 设置为 0，因为我们只想填充 prompt 内部已有的 mask
                        mask_id=mask_id
                    )
                    full_out_text = tokenizer.decode(output[0], skip_special_tokens=True)

                else:
                    # DiffuCoder 和 DreamCoder 逻辑
                    # 这两个模型通常使用 <|mask|> 作为 mask token
                    mask_token = '<|mask|>'
                    x_val_processed = x_val.replace('[MASK]', mask_token)

                    inputs = tokenizer(x_val_processed, return_tensors="pt")
                    input_ids = inputs.input_ids.to(device=raw_model.device)
                    attention_mask = inputs.attention_mask.to(device=raw_model.device)

                    # 检查是否包含 Mask ID
                    actual_mask_id = tokenizer.convert_tokens_to_ids(mask_token)
                    if actual_mask_id in input_ids:
                        print(f"  成功识别到噪声 Token {mask_token} (ID: {actual_mask_id})")

                    # 调用底层 diffusion_generate
                    # 根据模型设置不同的默认 steps 和 temperature
                    gen_steps = 768 if model_key == 'dreamcoder' else 256
                    gen_temp = 0.1 if model_key == 'dreamcoder' else 0.3
                    
                    output = raw_model.diffusion_generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=1, 
                        steps=gen_steps,
                        temperature=gen_temp,
                        top_p=0.95,
                        alg="entropy",
                        alg_temp=0.,
                    )
                    
                    seqs = output.sequences if hasattr(output, "sequences") else output
                    full_out_text = tokenizer.decode(seqs[0], skip_special_tokens=True)

                results.append({
                    'id': row['id'],
                    'X_original': x_val,
                    'output_full': full_out_text,
                    'y_ground_truth': y_val
                })
                
            except Exception as e:
                print(f"  处理出错: {e}")
                
        # 5. 保存结果
        output_file = os.path.join(output_dir, f'{model_key}_results.csv')
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"\n{model_key} 推理完成！结果已保存至: {output_file}")

        # 6. 清理内存，防止 OOM
        del model_instance
        if 'raw_model' in locals(): del raw_model
        if 'tokenizer' in locals(): del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print("\n所有指定模型测试完成！")

if __name__ == "__main__":
    main()