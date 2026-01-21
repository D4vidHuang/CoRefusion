import os
import torch
import sys
import re
from collections import Counter
from transformers import AutoTokenizer, AutoModel
from datetime import datetime

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

# --------------------------------------------------
# Style Helpers
# --------------------------------------------------
def classify_style(name):
    if '_' in name:
        return 'snake_case'
    if any(c.isupper() for c in name) and name[0].islower():
        return 'camelCase'
    if name[0].isupper():
        return 'PascalCase'
    return 'lowercase' # Ambiguous

def get_style_outlier(text):
    """
    Returns the name of the identifier that violates the dominant style.
    """
    try:
        from tree_sitter_languages import get_parser
        parser = get_parser('java')
        tree = parser.parse(bytes(text, "utf8"))
        
        java_keywords = {"public", "static", "int", "void", "class", "for", "return"}
        names = []
        
        def traverse(node):
            if node.type == 'identifier':
                name = text[node.start_byte:node.end_byte]
                if name not in java_keywords:
                    names.append(name)
            for child in node.children:
                traverse(child)
        traverse(tree.root_node)
    except:
        # Fallback to regex
        names = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', text)
        names = [n for n in names if n not in {"int", "public", "static", "void"}]

    # Count styles of non-ambiguous names
    styles = []
    for n in set(names):
        style = classify_style(n)
        if style != 'lowercase':
            styles.append((n, style))
    
    if not styles: return None, None

    style_counts = Counter([s[1] for s in styles])
    dominant_style = style_counts.most_common(1)[0][0]
    
    outliers = [n for n, style in styles if style != dominant_style]
    
    if outliers:
        return outliers[0], dominant_style
    return None, dominant_style

# --------------------------------------------------
# Experiment Engine
# --------------------------------------------------
def run_style_experiment(tokenizer, model, code_snippet, mask_token_id):
    print("\n" + "-"*50)
    print("Original Code Snippet:")
    print(code_snippet)
    
    outlier_name, dominant_style = get_style_outlier(code_snippet)
    
    if not outlier_name:
        print("No style outlier detected.")
        return

    print(f"\nDetected Outlier: '{outlier_name}' (Dominant Style: {dominant_style})")
    print(f"Action: Masking all occurrences of '{outlier_name}'...")

    # Accurate Masking: only whole words
    masked_code = re.sub(r'\b' + outlier_name + r'\b', '<|mask|><|mask|><|mask|>', code_snippet)
    
    inputs = tokenizer(masked_code, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")

    with torch.no_grad():
        output = model.diffusion_generate(
            input_ids,
            attention_mask=inputs.attention_mask.to("cuda"),
            max_length=input_ids.shape[1] + 16,
            steps=256,
            temperature=0, # Use Greedy for consistent style check
            return_dict_in_generate=True,
        )
    
    result_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
    print(result_text)
    # Simple extraction of what replaced the mask
    # We find the word at the position where <|mask|> used to be
    diff_match = re.search(r'\b[A-Za-z0-9_]+\b', result_text[masked_code.find('<|mask|>'):])
    fixed_name = diff_match.group(0) if diff_match else "Unknown"

    print(f"\nDLLM Filled Name: '{fixed_name}'")
    fixed_style = classify_style(fixed_name)
    print(f"Filled Style: {fixed_style}")
    
    if fixed_style == dominant_style:
        print("✅ SUCCESS: DLLM followed the code style consistency!")
    else:
        print("❌ FAILURE: DLLM did not follow the code style.")

def main():
    model_id = "apple/DiffuCoder-7B-Instruct"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    mask_token_id = tokenizer.convert_tokens_to_ids('<|mask|>')

    test_cases = [
        
        """
        public class matrix_utils {
            public void scale_matrix(int[][] matrix, int factor) {
                int rowCount = matrix.length;
                for (int i = 0; i < rowCount; i++) {
                    matrix[i][0] *= factor;
                }
            }
        }
        """
    ]

    for case in test_cases:
        run_style_experiment(tokenizer, model, case.strip(), mask_token_id)

if __name__ == "__main__":
    main()
