"""
Experiment: Mathematical Noise vs. Code Smell Noise in Diffusion Language Models
=================================================================================

Thesis Hypothesis:
    Code smells (e.g., terrible variable names) behave as a form of "noise"
    in diffusion language models, analogous to mathematical Gaussian noise
    in traditional diffusion processes. The denoising process naturally performs
    code refactoring (removing code smells).

This experiment compares THREE conditions:
    Group A (Math Noise):    Replace identifier tokens with [MASK] tokens
                             → Measures classical diffusion denoising behavior
    Group B (Smell Noise):   Replace identifiers with terrible names (e.g., "xxx", "tmp1")
                             → Measures code-smell-as-noise denoising behavior
    Group C (Control):       Keep the original clean code unchanged
                             → Baseline: how stable are identifiers in clean code

Metrics collected per-step:
    1. Stabilization Step:   The step at which the target identifier tokens stop changing
    2. Confidence:           Model's softmax probability for the predicted token at each step
    3. Entropy:              Shannon entropy of the logit distribution at target positions
    4. Token Match:          Whether the current prediction matches the ground truth

Output:
    - Per-sample CSV with step-by-step metrics
    - Summary CSV with aggregate statistics per group
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import csv
import re
import json
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# --- Mock torchvision for DreamCoder/DiffuCoder compatibility ---
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
# -----------------------------------------------------------------

# ======================== Configuration ==========================
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'test_filtered_1024.csv')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')

# Model configuration - can switch between DiffuCoder and DreamCoder
MODELS = {
    "diffucoder": {
        "id": "apple/DiffuCoder-7B-Instruct",
        "mask_token": "<|mask|>",
    },
    # Uncomment to also run DreamCoder:
    # "dreamcoder": {
    #     "id": "Dream-org/Dream-Coder-v0-Instruct-7B",
    #     "mask_token": "<|mask|>",
    # },
}

# Bad identifier names for Group B (Code Smell)
BAD_NAMES = [
    "xxx", "tmp1", "data1", "var1", "a1", "thing", "stuff",
    "foo", "x", "temp", "val"
]

# Experiment parameters
TOTAL_STEPS = 256       # Diffusion steps
LIMIT = 50              # Number of data points to process
REPEATS = 10            # Number of repetitions per data point
MAX_TOKENS = 1024       # Skip samples exceeding this token count
TEMPERATURE = 0.3       # Diffusion temperature
# =================================================================


def get_java_identifier_metadata(text, tokenizer, input_ids_tensor):
    """
    Identifies Java identifiers in the text and maps them to token indices.
    Returns (mask, identifier_groups) where each group contains:
        {'name': str, 'indices': list[int], 'range': (start_byte, end_byte)}
    """
    input_ids = input_ids_tensor[0].tolist()

    java_keywords = {
        "public", "static", "int", "if", "return", "void", "class", "for", "new", "boolean",
        "private", "protected", "final", "else", "while", "this", "null", "true", "false",
        "long", "double", "float", "char", "byte", "short", "import", "package", "try",
        "catch", "throw", "throws", "extends", "implements", "abstract", "interface",
        "switch", "case", "default", "break", "continue", "super", "instanceof",
        "String", "System", "Override", "Object", "List", "Map", "Set", "ArrayList",
        "HashMap", "HashSet", "Iterator", "Exception", "Integer", "Boolean",
    }

    id_ranges = []
    try:
        from tree_sitter_languages import get_parser
        parser = get_parser('java')
        tree = parser.parse(bytes(text, "utf8"))
        def traverse(node):
            if node.type == 'identifier':
                name = text[node.start_byte:node.end_byte]
                if name not in java_keywords:
                    id_ranges.append((node.start_byte, node.end_byte, name))
            for child in node.children:
                traverse(child)
        traverse(tree.root_node)
    except Exception:
        for m in re.finditer(r'\b[A-Za-z_][A-Za-z0-9_]*\b', text):
            if m.group(0) not in java_keywords:
                id_ranges.append((m.start(), m.end(), m.group(0)))

    # Calculate token-to-character offsets
    token_offsets = []
    for i in range(len(input_ids)):
        prefix = tokenizer.decode(input_ids[:i], skip_special_tokens=False)
        full = tokenizer.decode(input_ids[:i+1], skip_special_tokens=False)
        token_offsets.append((len(prefix), len(full)))

    identifier_groups = []
    mask = torch.zeros(len(input_ids), dtype=torch.bool)

    for start_byte, end_byte, id_name in id_ranges:
        group_indices = []
        for i, (t_start, t_end) in enumerate(token_offsets):
            t_mid = (t_start + t_end) / 2
            if start_byte <= t_mid < end_byte:
                group_indices.append(i)
                mask[i] = True
        if group_indices:
            identifier_groups.append({
                'name': id_name,
                'indices': group_indices,
                'range': (start_byte, end_byte)
            })

    return mask, identifier_groups


def find_subsequence_indices(sequence, subsequence):
    """Finds the first occurrence of subsequence in sequence."""
    seq_len = len(sequence)
    sub_len = len(subsequence)
    for i in range(seq_len - sub_len + 1):
        if sequence[i: i + sub_len] == subsequence:
            return i, i + sub_len
    return None


def get_num_transfer_tokens(mask_index, steps):
    """Compute the schedule of how many tokens to unmask at each step."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


def add_gumbel_noise(logits, temperature):
    """Add Gumbel noise for sampling."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def run_diffusion_with_tracking(
    tokenizer, model, input_ids, attention_mask,
    target_indices, ground_truth_tokens, mask_token_id,
    total_steps=256, temperature=0.3
):
    """
    Run diffusion denoising and track per-step metrics at the target token positions.

    Args:
        tokenizer:          The tokenizer
        model:              The diffusion model
        input_ids:          Input token IDs (1, seq_len) - potentially modified with masks or bad names
        attention_mask:     Attention mask (1, seq_len)
        target_indices:     List of token indices to track (the identifier positions)
        ground_truth_tokens: List of ground truth token IDs at those positions
        mask_token_id:      The mask token ID
        total_steps:        Number of diffusion steps
        temperature:        Sampling temperature

    Returns:
        step_metrics: list of dicts, one per step, containing:
            - step: int
            - target_tokens: list of current token IDs at target positions
            - target_decoded: str, decoded identifier at target positions
            - confidence: float, avg softmax prob of predicted token at target positions
            - entropy: float, avg Shannon entropy at target positions
            - matches_gt: bool, whether current tokens match ground truth
            - is_mask: bool, whether target positions are still masked
    """
    x = input_ids.clone()
    step_metrics = []

    # Determine initial mask positions for the diffusion schedule
    initial_mask_index = (x == mask_token_id)

    # If there are mask tokens, use the standard diffusion loop
    has_masks = initial_mask_index.any()

    if has_masks:
        num_transfer_tokens = get_num_transfer_tokens(initial_mask_index, total_steps)

    for step_i in range(total_steps):
        with torch.no_grad():
            current_mask_index = (x == mask_token_id)

            # Forward pass: get logits
            outputs = model(x, attention_mask=attention_mask.bool())
            logits = outputs.logits  # (1, seq_len, vocab_size)

            # --- Collect metrics at target positions ---
            target_logits = logits[0, target_indices, :]  # (n_target, vocab_size)

            # Softmax probabilities
            probs = F.softmax(target_logits.float(), dim=-1)

            # Shannon entropy: H = -sum(p * log(p))
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum(dim=-1)  # (n_target,)
            avg_entropy = entropy.mean().item()

            # Predicted tokens (argmax without noise for metric purposes)
            predicted_tokens = torch.argmax(target_logits, dim=-1).tolist()

            # Confidence: probability of the predicted token
            pred_probs = probs[range(len(target_indices)), predicted_tokens]
            avg_confidence = pred_probs.mean().item()

            # Current tokens at target positions
            current_target_tokens = x[0, target_indices].tolist()
            is_mask = all(t == mask_token_id for t in current_target_tokens)

            # Check if matches ground truth
            matches_gt = (current_target_tokens == ground_truth_tokens)

            # Decoded identifier
            target_decoded = tokenizer.decode(current_target_tokens).strip()
            target_decoded_clean = re.sub(r'[^A-Za-z0-9_]', '', target_decoded)

            step_metrics.append({
                'step': step_i,
                'target_tokens': current_target_tokens,
                'target_decoded': target_decoded_clean,
                'confidence': avg_confidence,
                'entropy': avg_entropy,
                'matches_gt': matches_gt,
                'is_mask': is_mask,
                'predicted_decoded': tokenizer.decode(predicted_tokens).strip(),
            })

            # --- Standard diffusion update (only if there are masks) ---
            if has_masks and current_mask_index.any():
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                p = F.softmax(logits.float(), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )

                x0 = torch.where(current_mask_index, x0, x)
                confidence = torch.where(current_mask_index, x0_p, torch.tensor(-np.inf, device=x.device))

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                if num_transfer_tokens.shape[1] > step_i:
                    k_transfer = num_transfer_tokens[0, step_i].item()
                else:
                    k_transfer = 0

                if k_transfer > 0:
                    _, select_index = torch.topk(confidence[0], k=int(k_transfer))
                    transfer_index[0, select_index] = True
                    x[transfer_index] = x0[transfer_index]
            elif not has_masks:
                # For Group B (smell) and Group C (control):
                # The model sees the full tokens (no masks).
                # We still run the forward pass to collect metrics,
                # but there's nothing to "denoise" in the traditional sense.
                # We track what the model PREDICTS at target positions.
                # After collecting metrics, we can exit early since the input
                # doesn't change without masks.
                # However, to make the comparison fair, we simulate what
                # WOULD happen if we ran the diffusion process on these tokens.

                # Strategy: Replace target tokens with model's predicted tokens
                # and observe if/when it converges.
                # This simulates "what does the model want to put here?"
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # Only update target positions (identifier positions)
                for idx in target_indices:
                    x[0, idx] = x0[0, idx]

    return step_metrics


def run_single_sample(
    tokenizer, model, mask_token_id,
    original_code, ground_truth_name, mask_start_char,
    sample_id, run_id
):
    """
    Run all three experimental groups (Math Noise, Smell Noise, Control)
    for a single data sample.

    Returns a list of result dicts (one per group).
    """
    results = []

    # ---- Prepare Ground Truth ----
    clean_code = original_code.replace("[MASK]", ground_truth_name)

    # Find the target identifier in the clean code
    inputs_clean = tokenizer(clean_code, return_tensors="pt").to("cuda")
    clean_ids = inputs_clean.input_ids[0].tolist()

    gt_token_ids = tokenizer.encode(ground_truth_name, add_special_tokens=False)
    # Try to find the ground truth tokens in the clean code tokens
    target_result = find_subsequence_indices(clean_ids, gt_token_ids)
    if target_result is None:
        # Try with a leading space
        gt_token_ids_space = tokenizer.encode(" " + ground_truth_name, add_special_tokens=False)
        target_result = find_subsequence_indices(clean_ids, gt_token_ids_space)
        if target_result is not None:
            gt_token_ids = gt_token_ids_space

    if target_result is None:
        return []  # Cannot locate identifier in tokenized sequence

    target_start, target_end = target_result
    target_indices = list(range(target_start, target_end))
    ground_truth_tokens = clean_ids[target_start:target_end]

    # ===== GROUP A: Mathematical Noise (Mask Tokens) =====
    group_a_ids = inputs_clean.input_ids.clone()
    for idx in target_indices:
        group_a_ids[0, idx] = mask_token_id

    try:
        metrics_a = run_diffusion_with_tracking(
            tokenizer, model,
            input_ids=group_a_ids,
            attention_mask=inputs_clean.attention_mask,
            target_indices=target_indices,
            ground_truth_tokens=ground_truth_tokens,
            mask_token_id=mask_token_id,
            total_steps=TOTAL_STEPS,
            temperature=TEMPERATURE,
        )

        # Find stabilization step (first step where it matches GT and stays)
        stab_step_a = find_stabilization_step(metrics_a, ground_truth_tokens, mask_token_id)

        results.append(build_result_dict(
            "math_noise", sample_id, run_id, ground_truth_name,
            metrics_a, stab_step_a
        ))
    except Exception as e:
        print(f"  [Group A Error] {e}")

    # ===== GROUP B: Code Smell Noise (Bad Naming) =====
    import random
    bad_name = random.choice(BAD_NAMES)
    smell_code = original_code.replace("[MASK]", bad_name)

    inputs_smell = tokenizer(smell_code, return_tensors="pt").to("cuda")
    smell_ids = inputs_smell.input_ids[0].tolist()

    # Find the bad name in the tokenized sequence
    bad_token_ids = tokenizer.encode(bad_name, add_special_tokens=False)
    smell_target = find_subsequence_indices(smell_ids, bad_token_ids)
    if smell_target is None:
        bad_token_ids_space = tokenizer.encode(" " + bad_name, add_special_tokens=False)
        smell_target = find_subsequence_indices(smell_ids, bad_token_ids_space)
        if smell_target is not None:
            bad_token_ids = bad_token_ids_space

    if smell_target is not None:
        smell_start, smell_end = smell_target
        smell_indices = list(range(smell_start, smell_end))

        # Ground truth for smell group: what is the GT at the equivalent position?
        # Note: token counts may differ, so we use the smell_indices for tracking
        # and compare against what the model predicts
        smell_gt_tokens = smell_ids[smell_start:smell_end]  # These are the "bad" tokens

        try:
            metrics_b = run_diffusion_with_tracking(
                tokenizer, model,
                input_ids=inputs_smell.input_ids.clone(),
                attention_mask=inputs_smell.attention_mask,
                target_indices=smell_indices,
                ground_truth_tokens=smell_gt_tokens,
                mask_token_id=mask_token_id,
                total_steps=TOTAL_STEPS,
                temperature=TEMPERATURE,
            )

            # For smell: stabilization = when tokens stop changing
            stab_step_b = find_change_stabilization_step(metrics_b)

            results.append(build_result_dict(
                "smell_noise", sample_id, run_id, ground_truth_name,
                metrics_b, stab_step_b, bad_name=bad_name
            ))
        except Exception as e:
            print(f"  [Group B Error] {e}")

    # ===== GROUP C: Control (Clean Code) =====
    try:
        metrics_c = run_diffusion_with_tracking(
            tokenizer, model,
            input_ids=inputs_clean.input_ids.clone(),
            attention_mask=inputs_clean.attention_mask,
            target_indices=target_indices,
            ground_truth_tokens=ground_truth_tokens,
            mask_token_id=mask_token_id,
            total_steps=TOTAL_STEPS,
            temperature=TEMPERATURE,
        )

        stab_step_c = find_change_stabilization_step(metrics_c)

        results.append(build_result_dict(
            "control", sample_id, run_id, ground_truth_name,
            metrics_c, stab_step_c
        ))
    except Exception as e:
        print(f"  [Group C Error] {e}")

    return results


def find_stabilization_step(metrics, ground_truth_tokens, mask_token_id):
    """
    Find the first step where target tokens match ground truth
    and remain matched for all subsequent steps.
    Returns -1 if never stabilized to GT.
    """
    n = len(metrics)
    for i in range(n):
        if metrics[i]['matches_gt'] and not metrics[i]['is_mask']:
            # Check if it stays matched for all remaining steps
            all_match = all(metrics[j]['matches_gt'] for j in range(i, n))
            if all_match:
                return i
    return -1


def find_change_stabilization_step(metrics):
    """
    Find the step at which the decoded identifier stops changing.
    This is useful for Group B (smell) and Group C (control)
    where we track when the model's prediction stabilizes.
    Returns -1 if it never stabilizes (always changing).
    """
    if not metrics:
        return -1

    n = len(metrics)
    # Find the last step where the decoded content changed
    for i in range(n - 1, 0, -1):
        if metrics[i]['target_decoded'] != metrics[i-1]['target_decoded']:
            return i
    # If it never changed, it was stable from the start
    return 0


def build_result_dict(group, sample_id, run_id, gt_name, metrics, stab_step, bad_name=None):
    """Build a standardized result dictionary."""
    # Aggregate trajectory data
    confidences = [m['confidence'] for m in metrics]
    entropies = [m['entropy'] for m in metrics]

    # First stable decoded value
    final_decoded = metrics[-1]['target_decoded'] if metrics else ""
    first_decoded = metrics[0]['target_decoded'] if metrics else ""

    # Did the identifier change from the initial value?
    changed = (first_decoded != final_decoded)

    # Recovery (for math noise: did it recover GT? for smell: did it change from bad name?)
    recovered_gt = any(m['matches_gt'] for m in metrics) if group == "math_noise" else None

    result = {
        'group': group,
        'sample_id': sample_id,
        'run_id': run_id,
        'ground_truth': gt_name,
        'bad_name': bad_name if bad_name else "",
        'initial_decoded': first_decoded,
        'final_decoded': final_decoded,
        'stabilization_step': stab_step,
        'changed': changed,
        'recovered_gt': recovered_gt if recovered_gt is not None else "",
        'avg_confidence': np.mean(confidences) if confidences else 0,
        'avg_entropy': np.mean(entropies) if entropies else 0,
        'min_entropy': np.min(entropies) if entropies else 0,
        'max_entropy': np.max(entropies) if entropies else 0,
        'final_confidence': confidences[-1] if confidences else 0,
        'final_entropy': entropies[-1] if entropies else 0,
        # Trajectory (sampled at 10 evenly spaced points for compact storage)
        'confidence_trajectory': json.dumps(
            [confidences[i] for i in np.linspace(0, len(confidences)-1, 10, dtype=int).tolist()]
        ) if confidences else "[]",
        'entropy_trajectory': json.dumps(
            [entropies[i] for i in np.linspace(0, len(entropies)-1, 10, dtype=int).tolist()]
        ) if entropies else "[]",
    }
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare Math Noise vs Code Smell Noise in Diffusion LMs")
    parser.add_argument("--model", type=str, default="diffucoder", choices=list(MODELS.keys()),
                        help="Which diffusion model to use")
    parser.add_argument("--limit", type=int, default=LIMIT,
                        help="Number of data points to process")
    parser.add_argument("--repeats", type=int, default=REPEATS,
                        help="Number of repetitions per data point")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS,
                        help="Number of diffusion steps")
    args = parser.parse_args()

    global TOTAL_STEPS, LIMIT, REPEATS
    TOTAL_STEPS = args.steps
    LIMIT = args.limit
    REPEATS = args.repeats

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Load Model ----
    model_cfg = MODELS[args.model]
    model_id = model_cfg["id"]
    print(f"Loading model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda").eval()
    mask_token_id = tokenizer.convert_tokens_to_ids(model_cfg["mask_token"])
    print(f"Model loaded. Mask token ID: {mask_token_id}")

    # ---- Load Data ----
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, header=None, names=['id', 'X', 'y'], nrows=LIMIT)
    print(f"Loaded {len(df)} samples.")

    # ---- Setup Output ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_csv = os.path.join(RESULTS_DIR, f"noise_comparison_detail_{args.model}_{timestamp}.csv")
    summary_csv = os.path.join(RESULTS_DIR, f"noise_comparison_summary_{args.model}_{timestamp}.csv")

    # CSV fields
    fields = [
        'group', 'sample_id', 'run_id', 'ground_truth', 'bad_name',
        'initial_decoded', 'final_decoded', 'stabilization_step',
        'changed', 'recovered_gt',
        'avg_confidence', 'avg_entropy', 'min_entropy', 'max_entropy',
        'final_confidence', 'final_entropy',
        'confidence_trajectory', 'entropy_trajectory',
    ]

    with open(detail_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

    print(f"\n{'='*60}")
    print(f" Experiment: Mathematical Noise vs. Code Smell Noise")
    print(f" Model:      {args.model} ({model_id})")
    print(f" Samples:    {len(df)}")
    print(f" Repeats:    {REPEATS}")
    print(f" Steps:      {TOTAL_STEPS}")
    print(f" Output:     {detail_csv}")
    print(f"{'='*60}\n")

    # ---- Run Experiment ----
    all_results = []

    for run_id in tqdm(range(REPEATS), desc="Runs"):
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Run {run_id+1}/{REPEATS}", leave=False):
            sample_id = row['id']
            X = str(row['X'])
            y = str(row['y']).strip()

            if '[MASK]' not in X:
                continue

            # Token length check
            tokens = tokenizer.encode(X.replace('[MASK]', y), add_special_tokens=False)
            if len(tokens) > MAX_TOKENS:
                continue

            try:
                results = run_single_sample(
                    tokenizer, model, mask_token_id,
                    X, y, X.find("[MASK]"),
                    sample_id, run_id + 1
                )

                # Write results incrementally
                with open(detail_csv, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    for r in results:
                        writer.writerow(r)
                        all_results.append(r)

            except Exception as e:
                print(f"\n  Error on sample {sample_id}: {e}")
                continue

    # ---- Generate Summary ----
    if all_results:
        df_results = pd.DataFrame(all_results)

        summary_data = []
        for group_name in ['math_noise', 'smell_noise', 'control']:
            group_df = df_results[df_results['group'] == group_name]
            if group_df.empty:
                continue

            # Convert stabilization_step to numeric
            stab_steps = pd.to_numeric(group_df['stabilization_step'], errors='coerce')
            valid_stab = stab_steps[stab_steps >= 0]

            summary = {
                'group': group_name,
                'n_samples': len(group_df),
                'avg_stabilization_step': valid_stab.mean() if not valid_stab.empty else -1,
                'median_stabilization_step': valid_stab.median() if not valid_stab.empty else -1,
                'std_stabilization_step': valid_stab.std() if not valid_stab.empty else -1,
                'pct_changed': group_df['changed'].astype(bool).mean() * 100,
                'avg_confidence': pd.to_numeric(group_df['avg_confidence']).mean(),
                'avg_entropy': pd.to_numeric(group_df['avg_entropy']).mean(),
                'avg_final_confidence': pd.to_numeric(group_df['final_confidence']).mean(),
                'avg_final_entropy': pd.to_numeric(group_df['final_entropy']).mean(),
            }

            # Group-specific metrics
            if group_name == 'math_noise':
                summary['recovery_rate'] = group_df['recovered_gt'].astype(bool).mean() * 100
            elif group_name == 'smell_noise':
                summary['refactoring_rate'] = group_df['changed'].astype(bool).mean() * 100

            summary_data.append(summary)

        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(summary_csv, index=False)

        # Print summary
        print(f"\n{'='*60}")
        print(" EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        for _, row in df_summary.iterrows():
            print(f"\n--- Group: {row['group']} ---")
            print(f"  Samples:                   {row['n_samples']}")
            print(f"  Avg Stabilization Step:    {row['avg_stabilization_step']:.2f}")
            print(f"  Median Stabilization Step: {row['median_stabilization_step']:.2f}")
            print(f"  % Changed:                 {row['pct_changed']:.1f}%")
            print(f"  Avg Confidence:            {row['avg_confidence']:.4f}")
            print(f"  Avg Entropy:               {row['avg_entropy']:.4f}")
            if 'recovery_rate' in row and pd.notna(row.get('recovery_rate')):
                print(f"  Recovery Rate (→ GT):      {row['recovery_rate']:.1f}%")
            if 'refactoring_rate' in row and pd.notna(row.get('refactoring_rate')):
                print(f"  Refactoring Rate:          {row['refactoring_rate']:.1f}%")

        print(f"\n{'='*60}")
        print(f" Detail results:  {detail_csv}")
        print(f" Summary results: {summary_csv}")
        print(f"{'='*60}")
    else:
        print("\nNo results generated.")

    # Cleanup
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()


if __name__ == "__main__":
    main()
