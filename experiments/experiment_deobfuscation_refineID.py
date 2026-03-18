"""
Deobfuscation Experiment using RefineID Dataset

This experiment:
1. Loads RefineID dataset (Java code with [MASK] and target)
2. Reconstructs original code by replacing [MASK] with target
3. Extracts all variable positions using Java AST (via javalang)
4. Obfuscates the code by replacing all variables with meaningless names (a, b, c, ...)
5. Uses diffusion models (DiffuCoder, DreamCoder) to deobfuscate
6. Evaluates the quality of recovered variable names

Diffusion Steps: 32
Models: DiffuCoder-7B-Instruct, Dream-Coder-v0-Instruct-7B
"""

import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
import re
import json
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import argparse

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

# Try to import javalang for Java AST parsing
try:
    import javalang
    HAS_JAVALANG = True
except ImportError:
    HAS_JAVALANG = False
    print("Warning: javalang not installed. Using regex-based identifier extraction.")
    print("Install with: pip install javalang")


def extract_java_identifiers_regex(code: str) -> List[Dict]:
    """
    Extract Java identifiers using regex (fallback when javalang is not available).
    This is less accurate but works for most cases.
    """
    # Java keywords to exclude
    JAVA_KEYWORDS = {
        'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char',
        'class', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum',
        'extends', 'final', 'finally', 'float', 'for', 'goto', 'if', 'implements',
        'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new',
        'package', 'private', 'protected', 'public', 'return', 'short', 'static',
        'strictfp', 'super', 'switch', 'synchronized', 'this', 'throw', 'throws',
        'transient', 'try', 'void', 'volatile', 'while', 'true', 'false', 'null',
        'var', 'record', 'sealed', 'permits', 'yield'
    }

    # Common Java types to exclude
    COMMON_TYPES = {
        'String', 'Integer', 'Long', 'Double', 'Float', 'Boolean', 'Character',
        'Object', 'Class', 'System', 'Math', 'List', 'ArrayList', 'Map', 'HashMap',
        'Set', 'HashSet', 'Collection', 'Iterator', 'Exception', 'RuntimeException',
        'Thread', 'Runnable', 'Comparable', 'Serializable', 'Override', 'Deprecated',
        'SuppressWarnings', 'FunctionalInterface', 'Test', 'Before', 'After'
    }

    results = []
    lines = code.split('\n')

    # Pattern for Java identifiers
    identifier_pattern = re.compile(r'\b([a-z_][a-zA-Z0-9_]*)\b')

    for line_num, line in enumerate(lines, 1):
        # Skip comments
        stripped = line.strip()
        if stripped.startswith('//') or stripped.startswith('*') or stripped.startswith('/*'):
            continue

        for match in identifier_pattern.finditer(line):
            name = match.group(1)
            col = match.start()

            # Skip keywords and common types
            if name in JAVA_KEYWORDS or name in COMMON_TYPES:
                continue

            # Skip if it looks like a method call (followed by '(')
            after_match = line[match.end():match.end()+1] if match.end() < len(line) else ''
            if after_match == '(':
                continue

            results.append({
                'name': name,
                'line': line_num,
                'col': col,
                'type': 'identifier'
            })

    # Remove duplicates based on position
    seen = set()
    unique_results = []
    for item in results:
        key = (item['line'], item['col'])
        if key not in seen:
            seen.add(key)
            unique_results.append(item)

    return sorted(unique_results, key=lambda x: (x['line'], x['col']))


def extract_java_identifiers_ast(code: str) -> List[Dict]:
    """
    Extract Java identifiers using javalang AST parser.
    More accurate than regex but requires valid Java code.
    """
    if not HAS_JAVALANG:
        return extract_java_identifiers_regex(code)

    results = []

    try:
        # Try to parse as a compilation unit
        tree = javalang.parse.parse(code)

        # Walk the AST and collect identifiers
        for path, node in tree:
            if isinstance(node, javalang.tree.LocalVariableDeclaration):
                for declarator in node.declarators:
                    if hasattr(declarator, 'name') and declarator.name:
                        results.append({
                            'name': declarator.name,
                            'type': 'local_variable',
                            'node_type': type(node).__name__
                        })
            elif isinstance(node, javalang.tree.VariableDeclarator):
                if hasattr(node, 'name') and node.name:
                    results.append({
                        'name': node.name,
                        'type': 'variable',
                        'node_type': type(node).__name__
                    })
            elif isinstance(node, javalang.tree.FormalParameter):
                if hasattr(node, 'name') and node.name:
                    results.append({
                        'name': node.name,
                        'type': 'parameter',
                        'node_type': type(node).__name__
                    })
            elif isinstance(node, javalang.tree.MemberReference):
                if hasattr(node, 'member') and node.member:
                    # This could be a variable reference
                    results.append({
                        'name': node.member,
                        'type': 'member_reference',
                        'node_type': type(node).__name__
                    })

    except Exception as e:
        # Fall back to regex if AST parsing fails
        print(f"AST parsing failed, using regex: {e}")
        return extract_java_identifiers_regex(code)

    # If AST didn't find much, supplement with regex
    if len(results) < 3:
        return extract_java_identifiers_regex(code)

    return results


def extract_java_identifiers_with_positions(code: str) -> List[Dict]:
    """
    Extract Java identifiers with their exact positions in the code.
    Uses regex for position tracking since javalang doesn't provide positions.
    """
    # Get unique identifier names from AST or regex
    identifiers = extract_java_identifiers_regex(code)

    # Get unique names
    unique_names = set()
    for item in identifiers:
        unique_names.add(item['name'])

    # Now find all occurrences with positions
    results = []
    lines = code.split('\n')

    for line_num, line in enumerate(lines, 1):
        for name in unique_names:
            # Find all occurrences of this identifier in the line
            pattern = re.compile(r'\b' + re.escape(name) + r'\b')
            for match in pattern.finditer(line):
                col = match.start()
                results.append({
                    'name': name,
                    'line': line_num,
                    'col': col,
                    'type': 'identifier'
                })

    return sorted(results, key=lambda x: (x['line'], x['col']))


def obfuscate_java_code(code: str) -> Tuple[str, Dict[str, str], List[Dict]]:
    """
    Obfuscate Java code by replacing all identifiers with meaningless names.

    Returns:
        obfuscated_code: Code with obfuscated variable names
        mapping: Dictionary mapping original names to obfuscated names
        identifiers: List of identifier positions
    """
    identifiers = extract_java_identifiers_with_positions(code)

    if not identifiers:
        return code, {}, []

    # Get unique names in order of first appearance
    unique_names = []
    seen = set()
    for item in identifiers:
        name = item['name']
        if name not in seen:
            unique_names.append(name)
            seen.add(name)

    # Create obfuscation mapping
    obfuscation_map = {}
    for idx, name in enumerate(unique_names):
        if idx < 26:
            obfuscated = chr(ord('a') + idx)
        else:
            first = (idx - 26) // 26
            second = (idx - 26) % 26
            obfuscated = chr(ord('a') + first) + chr(ord('a') + second)
        obfuscation_map[name] = obfuscated

    # Sort identifiers by position (reverse order to avoid offset issues)
    sorted_identifiers = sorted(identifiers, key=lambda x: (x['line'], x['col']), reverse=True)

    # Replace identifiers in code
    lines = code.split('\n')
    for item in sorted_identifiers:
        line_idx = item['line'] - 1
        col_idx = item['col']
        original_name = item['name']
        obfuscated_name = obfuscation_map[original_name]

        if line_idx < len(lines):
            line = lines[line_idx]
            before = line[:col_idx]
            after = line[col_idx + len(original_name):]
            lines[line_idx] = before + obfuscated_name + after

    obfuscated_code = '\n'.join(lines)

    # Update identifier positions for obfuscated code
    obfuscated_identifiers = extract_java_identifiers_with_positions(obfuscated_code)

    return obfuscated_code, obfuscation_map, obfuscated_identifiers


def get_token_mask_from_identifiers(code: str, tokenizer, input_ids: torch.Tensor,
                                    identifiers: List[Dict]) -> torch.Tensor:
    """
    Create a token-level mask for identifier positions.
    """
    if not identifiers:
        return torch.zeros(len(input_ids), dtype=torch.bool)

    # Get unique identifier names
    ident_names = set(item['name'] for item in identifiers)

    # Calculate byte offsets for each identifier
    lines = code.split('\n')
    ident_ranges = []
    for item in identifiers:
        line_idx = item['line'] - 1
        col_idx = item['col']
        if line_idx < len(lines):
            start_offset = sum(len(l) + 1 for l in lines[:line_idx]) + col_idx
            end_offset = start_offset + len(item['name'])
            ident_ranges.append((start_offset, end_offset, item['name']))

    # Tokenize and create mask
    tokens_decoded = [tokenizer.decode([tid]) for tid in input_ids]
    is_ident_token = torch.zeros(len(input_ids), dtype=torch.bool)

    current_char_pos = 0

    for i, tok_text in enumerate(tokens_decoded):
        if not tok_text or tok_text in ["<s>", "</s>", "<unk>", "<pad>", "<|im_start|>", "<|im_end|>"]:
            continue

        # Handle special tokenizer characters
        clean_tok = tok_text.replace('▁', ' ').replace('Ġ', ' ')
        search_text = clean_tok.strip()

        if not search_text:
            continue

        start_find = code.find(search_text, current_char_pos)
        if start_find != -1:
            tok_start = start_find
            tok_end = start_find + len(search_text)
            current_char_pos = tok_end

            # Check overlap with any identifier range
            for i_start, i_end, i_name in ident_ranges:
                if max(tok_start, i_start) < min(tok_end, i_end):
                    # Only mask if token is pure identifier (no syntax symbols)
                    pure_content = clean_tok.lstrip(' ')
                    if not re.search(r"[^a-zA-Z0-9_]", pure_content):
                        is_ident_token[i] = True
                    break

    return is_ident_token


def run_constrained_diffusion(model, tokenizer, obfuscated_code: str,
                              identifiers: List[Dict], total_steps: int = 32,
                              device: str = "cuda") -> Tuple[str, List[str]]:
    """
    Run constrained diffusion to deobfuscate code.
    Only updates tokens at identifier positions.
    """
    # Tokenize
    input_ids_full = tokenizer.encode(obfuscated_code, return_tensors="pt").to(device)
    input_ids = input_ids_full[0]

    # Get mask
    ident_mask = get_token_mask_from_identifiers(obfuscated_code, tokenizer, input_ids, identifiers).to(device)

    # Initial state
    x = input_ids_full.clone()
    history = []

    # Diffusion loop
    for step in range(total_steps):
        with torch.no_grad():
            logits = model(x).logits
            x_pred = torch.argmax(logits, dim=-1)

            # Apply updates ONLY to identifier positions
            x_next = torch.where(ident_mask, x_pred, x)

            # Record history
            decoded = tokenizer.decode(x_next[0], skip_special_tokens=True)
            history.append(decoded)

            # Check convergence
            if torch.equal(x, x_next):
                break

            x = x_next

    final_code = tokenizer.decode(x[0], skip_special_tokens=True)
    return final_code, history


def calculate_metrics(original_code: str, obfuscated_code: str,
                     deobfuscated_code: str, obfuscation_map: Dict[str, str]) -> Dict:
    """
    Calculate evaluation metrics.
    """
    # Extract identifiers from original and deobfuscated code
    original_identifiers = extract_java_identifiers_with_positions(original_code)
    deobfuscated_identifiers = extract_java_identifiers_with_positions(deobfuscated_code)

    # Create reverse mapping (obfuscated -> original)
    reverse_map = {v: k for k, v in obfuscation_map.items()}

    # Get unique names
    original_names = set(item['name'] for item in original_identifiers)
    deobfuscated_names = set(item['name'] for item in deobfuscated_identifiers)

    # Calculate metrics
    total_original = len(original_names)
    exact_matches = len(original_names & deobfuscated_names)

    # Check if deobfuscated names are meaningful (not single letters)
    meaningful_names = sum(1 for name in deobfuscated_names if len(name) > 1)

    return {
        'total_original_identifiers': total_original,
        'total_deobfuscated_identifiers': len(deobfuscated_names),
        'exact_matches': exact_matches,
        'exact_match_rate': exact_matches / total_original if total_original > 0 else 0,
        'meaningful_names': meaningful_names,
        'meaningful_rate': meaningful_names / len(deobfuscated_names) if deobfuscated_names else 0,
        'original_names': list(original_names),
        'deobfuscated_names': list(deobfuscated_names)
    }


def run_deobfuscation_experiment(model_name: str, model_id: str,
                                 data_path: str, total_steps: int = 32,
                                 device: str = "cuda", max_samples: int = None):
    """
    Run deobfuscation experiment on RefineID dataset.
    """
    print(f"\n{'='*80}")
    print(f"Deobfuscation Experiment: {model_name}")
    print(f"Model ID: {model_id}")
    print(f"Diffusion Steps: {total_steps}")
    print(f"{'='*80}")

    # Create output directory
    os.makedirs("results/deobfuscation_refineID", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/deobfuscation_refineID/{model_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path, header=None, names=['id', 'masked_code', 'target'])

    if max_samples:
        df = df.head(max_samples)

    print(f"Total samples: {len(df)}")

    # Load model
    print(f"\nLoading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    ).to(device).eval()

    # Run experiment
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Testing {model_name}"):
        item_id = row['id']
        masked_code = str(row['masked_code'])
        target = str(row['target']).strip()

        try:
            # Step 1: Reconstruct original code
            original_code = masked_code.replace('[MASK]', target)

            # Step 2: Obfuscate the code
            obfuscated_code, obfuscation_map, identifiers = obfuscate_java_code(original_code)

            if not identifiers:
                print(f"Warning: No identifiers found in sample {item_id}, skipping...")
                continue

            # Step 3: Run constrained diffusion
            deobfuscated_code, history = run_constrained_diffusion(
                model, tokenizer, obfuscated_code, identifiers,
                total_steps=total_steps, device=device
            )

            # Step 4: Calculate metrics
            metrics = calculate_metrics(original_code, obfuscated_code,
                                        deobfuscated_code, obfuscation_map)

            results.append({
                'id': item_id,
                'target': target,
                'num_identifiers': len(set(item['name'] for item in identifiers)),
                'exact_match_rate': metrics['exact_match_rate'],
                'meaningful_rate': metrics['meaningful_rate'],
                'obfuscation_map': json.dumps(obfuscation_map),
                'original_names': json.dumps(metrics['original_names']),
                'deobfuscated_names': json.dumps(metrics['deobfuscated_names']),
                'converged_steps': len(history)
            })

        except Exception as e:
            print(f"Error on sample {item_id}: {e}")
            results.append({
                'id': item_id,
                'error': str(e)
            })

    # Save results
    results_df = pd.DataFrame(results)
    results_file = os.path.join(output_dir, "results.csv")
    results_df.to_csv(results_file, index=False)

    # Calculate summary statistics
    successful_results = [r for r in results if 'error' not in r]
    if successful_results:
        avg_exact_match = sum(r['exact_match_rate'] for r in successful_results) / len(successful_results)
        avg_meaningful = sum(r['meaningful_rate'] for r in successful_results) / len(successful_results)
        avg_steps = sum(r['converged_steps'] for r in successful_results) / len(successful_results)
    else:
        avg_exact_match = 0
        avg_meaningful = 0
        avg_steps = 0

    summary = {
        'model_name': model_name,
        'model_id': model_id,
        'total_samples': len(df),
        'successful_samples': len(successful_results),
        'failed_samples': len(df) - len(successful_results),
        'diffusion_steps': total_steps,
        'average_exact_match_rate': avg_exact_match,
        'average_meaningful_rate': avg_meaningful,
        'average_converged_steps': avg_steps,
        'timestamp': timestamp
    }

    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print(f"EXPERIMENT SUMMARY: {model_name}")
    print(f"{'='*80}")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Successful: {summary['successful_samples']}")
    print(f"Failed: {summary['failed_samples']}")
    print(f"Average exact match rate: {summary['average_exact_match_rate']:.2%}")
    print(f"Average meaningful rate: {summary['average_meaningful_rate']:.2%}")
    print(f"Average converged steps: {summary['average_converged_steps']:.1f}")
    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*80}\n")

    # Cleanup
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    return summary


def main():
    parser = argparse.ArgumentParser(description='Deobfuscation Experiment on RefineID Dataset')
    parser.add_argument('--model', type=str, default='both',
                        choices=['diffucoder', 'dreamcoder', 'both'],
                        help='Model to use (default: both)')
    parser.add_argument('--data', type=str, default='data/test_filtered_1024.csv',
                        help='Path to RefineID dataset')
    parser.add_argument('--steps', type=int, default=32,
                        help='Number of diffusion steps (default: 32)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process')

    args = parser.parse_args()

    # Model configurations
    models = {
        'diffucoder': {
            'name': 'DiffuCoder-7B',
            'id': 'apple/DiffuCoder-7B-Instruct'
        },
        'dreamcoder': {
            'name': 'DreamCoder-7B',
            'id': 'Dream-org/Dream-Coder-v0-Instruct-7B'
        }
    }

    # Run experiments
    all_summaries = []

    if args.model == 'both':
        models_to_run = ['diffucoder', 'dreamcoder']
    else:
        models_to_run = [args.model]

    for model_key in models_to_run:
        model_config = models[model_key]
        summary = run_deobfuscation_experiment(
            model_name=model_config['name'],
            model_id=model_config['id'],
            data_path=args.data,
            total_steps=args.steps,
            device=args.device,
            max_samples=args.max_samples
        )
        all_summaries.append(summary)

    # Print comparison if both models were run
    if len(all_summaries) == 2:
        print(f"\n{'='*80}")
        print("MODEL COMPARISON")
        print(f"{'='*80}")
        print(f"{'Model':<20} {'Exact Match':<15} {'Meaningful':<15} {'Avg Steps':<15}")
        print("-" * 80)
        for s in all_summaries:
            print(f"{s['model_name']:<20} {s['average_exact_match_rate']:<15.2%} "
                  f"{s['average_meaningful_rate']:<15.2%} {s['average_converged_steps']:<15.1f}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
