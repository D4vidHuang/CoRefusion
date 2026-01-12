
import sys
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

# --- Mock torchvision ---
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

def get_identifier_token_mask(code, tokenizer, input_ids):
    """
    Uses tree-sitter to find identifier byte ranges and maps them to tokenizer token indices.
    """
    # 1. Initialize Tree-Sitter
    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)
    tree = parser.parse(bytes(code, "utf8"))
    
    # 2. Query for identifiers
    query = PY_LANGUAGE.query("(identifier) @id")
    captures = query.captures(tree.root_node)
    
    ident_ranges = []
    for node, _ in captures:
        ident_ranges.append((node.start_byte, node.end_byte))
    
    # 3. Use offset_mapping to find which tokens fall into these ranges
    # Note: encoding might add special tokens (BOS/EOS), we need to track them.
    encoding = tokenizer(code, return_offsets_mapping=True, return_tensors="pt")
    offset_mapping = encoding.offset_mapping[0] # [seq_len, 2]
    
    # Create a boolean mask for tokens that are identifiers
    is_identifier_token = torch.zeros(input_ids.size(1), dtype=torch.bool)
    
    for i, (start, end) in enumerate(offset_mapping):
        # Skip special tokens (usually have 0,0 offset)
        if start == 0 and end == 0:
            continue
        
        # Check if this token overlaps with any identifier range
        for i_start, i_end in ident_ranges:
            if start >= i_start and end <= i_end:
                is_identifier_token[i] = True
                break
                
    return is_identifier_token

def run_constrained_diffusion(model_name, model_id, code_snippet, total_steps=20):
    print(f"\n{'='*20} Constrained Diffusion: {model_name} {'='*20}")
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"results/{model_name}_constrained_{timestamp}.txt"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    
    # Tokenize
    inputs = tokenizer(code_snippet, return_tensors="pt").to("cuda")
    x = inputs.input_ids.clone()
    x_original = inputs.input_ids.clone()
    attention_mask = inputs.attention_mask
    
    # Get mask for identifier tokens
    print("Extracting identifier locations using Tree-Sitter...")
    ident_mask = get_identifier_token_mask(code_snippet, tokenizer, x).to("cuda")
    
    num_idents = ident_mask.sum().item()
    print(f"Total tokens identified as parts of identifiers: {num_idents} / {x.size(1)}")

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Constrained: Only identifiers allowed to change.\n")
        f.write(f"Identifier Tokens: {num_idents}\n")
        f.write(f"Original Code:\n{code_snippet}\n" + "="*40 + "\n\n")
        
        for step in range(total_steps):
            with torch.no_grad():
                logits = model(x, attention_mask=attention_mask.bool()).logits
                x_pred = torch.argmax(logits, dim=-1)
                
                # KEY STEP: Constrain update!
                # Only update tokens where ident_mask is True
                x_next = torch.where(ident_mask, x_pred, x)
                
                if torch.equal(x, x_next):
                    f.write(f"Step {step+1}: Converged.\n")
                    break
                
                x = x_next
                decoded = tokenizer.decode(x[0], skip_special_tokens=True)
                
                f.write(f"--- Step {step+1} ---\n{decoded}\n\n")
                if (step + 1) % 5 == 0:
                    print(f"Step {step+1} recorded.")

        f.write("\nFinal Output:\n" + tokenizer.decode(x[0], skip_special_tokens=True))

    print(f"Finished. Results: {output_filename}")

if __name__ == "__main__":
    terrible_sort = """def a(b):
    c = len(b)
    for d in range(c):
        for e in range(0, c - d - 1):
            if b[e] > b[e + 1]:
                f = b[e]
                b[e] = b[e + 1]
                b[e + 1] = f
    return b"""

    # Run for DreamCoder or DiffuCoder
    run_constrained_diffusion("dreamcoder", "Dream-org/Dream-Coder-v0-Instruct-7B", terrible_sort)
