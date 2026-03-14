import sys
import torch
import pandas as pd
import os
import re
from transformers import AutoTokenizer, AutoModel

# --- Mock torchvision for compatibility ---
class MockModule:
    def __getattr__(self, name): return MockModule()
    def __call__(self, *args, **kwargs): return MockModule()

sys.modules['torchvision'] = MockModule()
sys.modules['torchvision.ops'] = MockModule()
sys.modules['torchvision.transforms'] = MockModule()

if not hasattr(torch.ops, 'torchvision'):
    class DummyOps:
        def nms(*args, **kwargs): return torch.tensor([])
    torch.ops.torchvision = DummyOps()
# ------------------------------------------

MODELS = {
    "diffucoder": {
        "id": "apple/DiffuCoder-7B-Instruct",
        "mask_token": "<|mask|>",
    },
    "dreamcoder": {
        "id": "Dream-org/Dream-Coder-v0-Instruct-7B",
        "mask_token": "<|mask|>",
    }
}

BAD_NAMES = ["x", "temp", "data", "obj", "var", "foo", "bar", "val"]

def load_model(model_name="diffucoder"):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    
    cfg = MODELS[model_name]
    print(f"Loading {model_name} ({cfg['id']})...")
    tokenizer = AutoTokenizer.from_pretrained(cfg['id'], trust_remote_code=True)
    model = AutoModel.from_pretrained(cfg['id'], torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    return model, tokenizer, cfg

def load_data(path, limit=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    
    df = pd.read_csv(path, header=None, names=['id', 'X', 'y'])
    if limit:
        df = df.head(limit)
    return df

def find_token_indices(tokenizer, input_ids, target_str):
    """
    Find the indices of the tokens corresponding to target_str in input_ids.
    Returns a list of (start, end) tuples.
    Tries both the raw string and the string with a leading space.
    """
    input_list = input_ids.tolist() if torch.is_tensor(input_ids) else input_ids
    indices = []
    
    # Try multiple variations of the target string to account for tokenization differences
    candidates = [target_str, " " + target_str]
    
    for cand in candidates:
        target_ids = tokenizer.encode(cand, add_special_tokens=False)
        if not target_ids:
            continue
            
        # Simple sliding window search
        for i in range(len(input_list) - len(target_ids) + 1):
            if input_list[i:i+len(target_ids)] == target_ids:
                indices.append((i, i+len(target_ids)))
    
    # Remove duplicates and sort
    return sorted(list(set(indices)))
