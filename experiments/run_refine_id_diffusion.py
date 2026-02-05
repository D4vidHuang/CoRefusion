import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import re
import sys

# ================= 配置区 =================
DATA_PATH = "data/test_filtered_1024.csv"
RESULTS_DIR = "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 扩散模型元数据配置
# 注意：我们将 [MASK] 替换为 4 个模型特定的掩码 token
MODEL_METADATA = {
    "DiffuCoder-7B-Instruct": {
        "id": "apple/DiffuCoder-7B-Instruct",
        "mask_token": "<|mask|>",
        "type": "diffucoder"
    },
    "LLaDA-8B-Instruct": {
        "id": "GSAI-ML/LLaDA-8B-Instruct",
        "mask_id": 126336,
        "type": "llada"
    },
    "Dream-Coder-v0-Instruct-7B": {
        "id": "Dream-org/Dream-Coder-v0-Instruct-7B",
        "mask_token": "<|mask|>",
        "type": "dreamcoder"
    }
}

# 设置 LLaDA 路径以便导入其专门的生成逻辑
project_root = os.getcwd()
llada_path = os.path.join(project_root, 'external_repos', 'LLaDA')
if os.path.exists(llada_path) and llada_path not in sys.path:
    sys.path.append(llada_path)

llada_generate = None
try:
    import importlib.util
    gen_path = os.path.join(llada_path, 'generate.py')
    if os.path.exists(gen_path):
        spec = importlib.util.spec_from_file_location("llada_generate_mod", gen_path)
        llada_gen_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(llada_gen_mod)
        llada_generate = llada_gen_mod.generate
except Exception as e:
    print(f"Warning: 无法导入 LLaDA 生成模块: {e}")

def clean_identifier(text):
    """从去噪后的片段中提取干净的标识符"""
    text = text.strip().split('\n')[0].strip('`"\' ')
    match = re.search(r'[a-zA-Z_][a-zA-Z0-9_]*', text)
    return match.group(0) if match else text

def run_diffusion_experiment():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    print(f"正在读取数据: {DATA_PATH}...")
    try:
        # CSV 格式预预期为: id, masked_code, target
        df = pd.read_csv(DATA_PATH, header=None, names=['id', 'masked_code', 'target'])
    except Exception as e:
        print(f"读取数据失败: {e}")
        return

    for model_name, meta in MODEL_METADATA.items():
        print(f"\n{'='*60}")
        print(f"运行扩散模型实验: {model_name}")
        print(f"{'='*60}")

        try:
            # 加载分词器和基础 Diffusion 模型
            tokenizer = AutoTokenizer.from_pretrained(meta['id'], trust_remote_code=True)
            model = AutoModel.from_pretrained(
                meta['id'], 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True
            ).to(DEVICE).eval()
        except Exception as e:
            print(f"加载模型 {model_name} 失败: {e}")
            continue

        results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Testing {model_name}"):
            item_id = row['id']
            masked_code = str(row['masked_code'])
            ground_truth = str(row['target']).strip()

            try:
                # 1. 构造输入：将 [MASK] 替换为 4 个掩码 token
                if meta['type'] == 'llada':
                    m_text = tokenizer.decode([meta['mask_id']])
                    input_text = masked_code.replace("[MASK]", m_text * 4)
                    inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
                    # 确保 encoded 的确实是 mask_id
                    input_ids = inputs.input_ids
                    m_encoded = tokenizer.encode(m_text, add_special_tokens=False)
                    for m_id in m_encoded:
                        input_ids[input_ids == m_id] = meta['mask_id']
                else:
                    input_text = masked_code.replace("[MASK]", meta['mask_token'] * 4)
                    inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
                    input_ids = inputs.input_ids

                # 2. 调用扩散生成 (直接 infilling，无需 Prompt)
                with torch.no_grad():
                    if meta['type'] == 'llada':
                        if llada_generate is None:
                            refined_code = "Error: LLaDA module missing"
                        else:
                            # gen_length=0 控制仅填充现有 mask
                            out = llada_generate(
                                model, input_ids, steps=128, gen_length=0, 
                                block_length=32, mask_id=meta['mask_id']
                            )
                            refined_code = tokenizer.decode(out[0], skip_special_tokens=True)
                    else:
                        # DiffuCoder / DreamCoder 逻辑
                        output = model.diffusion_generate(
                            input_ids,
                            attention_mask=inputs.attention_mask,
                            max_new_tokens=0, # 填空模式不增加新 token 长度
                            steps=256,
                            temperature=0.2, # 较低温度以提高确定性
                            top_p=0.95,
                            alg="entropy"
                        )
                        seqs = output.sequences if hasattr(output, "sequences") else output
                        refined_code = tokenizer.decode(seqs[0], skip_special_tokens=True)

                # 3. 提取预测出的标识符
                # 基于上下文锚点定位
                try:
                    parts = masked_code.split("[MASK]")
                    prefix_anchor = parts[0][-25:]
                    suffix_anchor = parts[1][:25] if len(parts) > 1 else ""
                    
                    pattern = re.escape(prefix_anchor) + r"(.*?)" + re.escape(suffix_anchor)
                    match = re.search(pattern, refined_code, re.DOTALL)
                    prediction = clean_identifier(match.group(1)) if match else "unknown"
                except:
                    prediction = "unknown"

                results.append({
                    "id": item_id,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "refined_code": refined_code,
                    "correct": (prediction == ground_truth)
                })

            except Exception as e:
                results.append({"id": item_id, "error": str(e)})

        # 结果保存
        output_name = model_name.split('/')[-1]
        output_file = os.path.join(RESULTS_DIR, f"{output_name}_refineID_diffusion.csv")
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"测试完成，结果保存至: {output_file}")

        # 显存清理
        del model, tokenizer
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        import gc; gc.collect()

if __name__ == "__main__":
    run_diffusion_experiment()
