import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
import sys

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

def find_subsequence_indices(sequence, subsequence):
    """
    Finds the start and end indices of the first occurrence of subsequence in sequence.
    Returns (start_index, end_index) or None if not found.
    """
    seq_len = len(sequence)
    sub_len = len(subsequence)
    for i in range(seq_len - sub_len + 1):
        if sequence[i : i + sub_len] == subsequence:
            return i, i + sub_len
    return None

def run_experiment():
    model_id = "apple/DiffuCoder-7B-Instruct"
    print(f"Loading model: {model_id}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    csv_path = os.path.join('data', 'test.csv')
    print(f"Reading data from {csv_path}...")
    
    try:
        # Reading first 100 for the experiment as an initial set, or all if feasible.
        # User said "run this experiment on the dataset", but for safety I'll start with a chunk and can expand.
        # Let's read all but process with a limit or specific range if args were parsed, 
        # but here I'll default to a reasonable number or all.
        df = pd.read_csv(csv_path, header=None, names=['id', 'X', 'y'])
        print(f"Loaded {len(df)} rows.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"results/steps_experiment_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    results_csv = os.path.join(experiment_dir, f"identifier_change_summary_{timestamp}.csv")
    
    TERRIBLE_IDENTIFIER = "terrible_var_name_x"
    print(f"Using terrible identifier: {TERRIBLE_IDENTIFIER}")

    # Pre-calculate identifier tokens to ensure we can find them
    # Note: Tokenization might depend on context (prefix space), so we'll handle that inside the loop if needed,
    # but arguably the identifier tokens should be consistent if surrounded by spaces or specific syntax.
    # To be safe, we'll search for the token sequence locally.
    
    results = []

    # Limit to reasonable number for 'average' calculation if dataset is huge, 
    # or iterate all if that's the goal. Let's process 50 for now or all if small.
    # The user said "on the data dataset", implying the whole thing. 
    # But I should probably facilitate resume or batches. 
    # I'll process a subset first to verify (as per plan: small test).
    # I will allow passing an argument via env var or just modify the script. 
    # For now, let's hardcode a limit of 50 for the first run to test, then I can run more.
    # Actually, the user asked to "find the average ... on the whole dataset".
    # I will start with 10 to demonstrate, then can run more.
    
    process_limit = len(df)
    print(f"Processing the entire dataset ({process_limit} items)...")

    for i, row in df.head(process_limit).iterrows():
        try:
            input_text = row['X']
            if '[MASK]' not in input_text:
                print(f"Skipping {row['id']}: No [MASK] found.")
                continue

            # Replace MASK with terrible identifier
            # We assume [MASK] appears once or we replace all? User said "the MASK position".
            # Usually strict single mask or multiple. I'll replace the first one or all.
            # "replace the MASK position" implies specific one.
            masked_text = input_text.replace('[MASK]', TERRIBLE_IDENTIFIER)
            
            # Tokenize full text WITHOUT truncation as requested
            # Reference: run_diffucoder_noise_exp.py
            inputs = tokenizer(masked_text, return_tensors="pt", truncation=False)
            input_ids = inputs.input_ids.to("cuda")
            attention_mask = inputs.attention_mask.to("cuda")
            
            # Find indices of the terrible identifier
            # We tokenize the identifier alone to know what to look for
            # Adding a space prefix might be necessary depending on where MASK was.
            # Best way: tokenize identifier with space and without, try to find either.
            id_tokens = tokenizer.encode(TERRIBLE_IDENTIFIER, add_special_tokens=False)
            
            # Since tokenizer might merge tokens depending on context, finding exact sub-sequence matches is best effort.
            # We convert tensor to list for searching
            input_ids_list = input_ids[0].tolist()
            
            target_indices = find_subsequence_indices(input_ids_list, id_tokens)
            
            if not target_indices:
                # Try with prefix space if not found
                id_tokens_space = tokenizer.encode(" " + TERRIBLE_IDENTIFIER, add_special_tokens=False)
                target_indices = find_subsequence_indices(input_ids_list, id_tokens_space)
            
            if not target_indices:
                print(f"Could not find identifier tokens in processed text for {row['id']}. Skipping.")
                continue
                
            start_idx, end_idx = target_indices
            original_id_tokens = input_ids_list[start_idx:end_idx]
            
            # Generate with history
            # steps=256 is standard for diffucoder
            # We perform diffusion on the FULL sequence? 
            # The prompt says: "convert code to token, then use diffucoder for diffusion operation"
            # Standard DiffuCoder usage for code completion/infilling usually masks the tokens first?
            # User says: "replace MASK with terrible identifier... then do diffusion... record when it changes"
            # This implies we are treating the "terrible identifier" as NOISE that needs to be denoised?
            # Or are we running the model in a way that it treats the whole input as noisy?
            # The standard `diffusion_generate` usually adds noise and then denoises? 
            # OR does it take the input as the starting point (step T or 0 depending on view)?
            # If we just pass `input_ids`, DiffuCoder (based on standard diffusion LM) might assume these are the 'masked' or 'noisy' tokens 
            # from which to generate?
            # Usually `diffusion_generate` takes `input_ids` as the *condition* or *initial noise*?
            # In `run_diffucoder_noise_exp.py`:
            #   input_text = row['X']
            #   noisy_text = input_text.replace('[MASK]', '<|mask|>')
            #   inference_text = noisy_text.replace('<|mask|>', mask_token)
            # This logic suggests DiffuCoder expects explicit `<|mask|>` tokens for infilling.
            # BUT the user specifically said: "replace MASK with terrible naming identifier ... then use diffucoder for diffusion".
            # This suggests we want to see if the model *corrects* the terrible identifier.
            # For a diffusion model to correct it, it must treat it as noise or we must inject noise.
            # If I just pass the text as is to `diffusion_generate`, does it do anything?
            # If I don't mask it, maybe it won't change it?
            # However, diffusion models generally start from Gaussian noise if pure generation, 
            # OR they perform "editing" if we start from a partially noised state.
            
            # User phrase: "diffucoder for diffusion operation"
            # Hypothesis: The user wants to see the "correction" capability.
            # If I just feed it fully valid tokens (with terrible name), standard DiffuCoder might just return it as is if it thinks it's valid?
            # Or maybe the user assumes `diffusion_generate` adds noise and then denoises?
            # Usually `diffusion_generate` in these codebases (like SEDD or similar) generates from pure noise conditioned on something, 
            # or creates a chain.
            
            # Let's assume the user wants to check the *Refinement* capability.
            # To do this, usually one must add noise (forward process) and then denoise (reverse process), 
            # OR start the reverse process from the given tokens (treating them as a noisy intermediate step).
            
            # If `diffusion_generate` just generates from scratch, it will ignore my identifier tokens (unless conditioned on them).
            # If `diffusion_generate` takes `input_ids` as the prompt?
            # In `run_diffucoder_noise_exp.py`, they pass `inference_text` which has `<|mask|>` tokens.
            # This suggests masking is required for infilling.
            
            # RE-READ CAREFULLY: "replace the MASK position with a default terrible naming identifier, then convert to token, then use diffucoder for diffusion operation"
            # This implies the input to the diffusion process IS the code with the terrible identifier.
            # Does DiffuCoder automatically add noise?
            # If I pass `input_ids` to `diffusion_generate`, typically it uses them as constraints or initial state?
            
            # Let's look at `external_repos/diffucoder.py`:
            # It takes `input_ids` from `prompt`.
            # `prompt` seems to be a chat template.
            # `diffusion_generate` is called with `input_ids`.
            
            # If I pass the code as `input_ids`, I need to know if `diffusion_generate` treats it as a guide or the thing to fail/denoise.
            # If it treats it as prefix/prompt, it will key generating *continuation*.
            # But we want to change the *existing* identifier.
            
            # Wait, "replace MASK ... convert to token ... diffusion operation".
            # Maybe the user implies the "Noised" code IS the code with terrible identifier, 
            # and we want to see if the model "denoises" it (corrects the name) back to something good?
            # If so, we probably need to tell the model to "edit" or we rely on the stochasticity of the diffusion process 
            # if we run it in a specific mode (e.g. edit mode or low noise level).
            
            # BUT, the user prompt is: "diffucoder for diffusion operation". 
            # Maybe they mean: Start the diffusion process (reverse) but initialize the state with these tokens?
            # If `diffusion_generate` starts from random noise, my input tokens are just ignored (or used as cross-attn condition).
            # If I want to checking *when* it changes, I must assume the process starts *at* or *near* the terrible identifier 
            # and moves away from it? 
            # OR the process generates the identifier, and we want to see when it *diverges* from the terrible one 
            # (assuming we forced the terrible one initially?).
            
            # Let's assume the standard behavior:
            # We want to see if the model *generates* the terrible identifier (unlikely) 
            # OR we provided the terrible identifier as a starting point for *correction*.
            
            # Given the lack of specific "edit" instructions, I will assume:
            # The user considers the "terrible identifier" as "noise".
            # The diffusion model should "denoise" this verification.
            # We should probably set the initial tokens of the diffusion process to be this terrible identifier?
            # But `diffusion_generate` usually encapsulates the loop.
            
            # Let's look at `diffusion_generate` signature in my inspection script:
            # `steps=256`, `max_length=...`.
            
            # If the user script `run_diffucoder_noise_exp.py` used `input_ids` which contained `<|mask|>`, 
            # the model filled those masks.
            # Here, we have NO masks. We have a terrible identifier.
            # If I pass this successfully to `diffusion_generate` without masks, what does it do?
            # 1. It might treat it as a prompt and generate *more* code (continuation).
            # 2. It might treat it as the "noisy" input if the API supports "image-to-image" style (code-to-code) variation.
            
            # I'll stick to a simple interpretation first:
            # Run the generation with the textual input (with terrible identifier) as the `input_ids`.
            # If the model is an encoder-decoder or similar, or specific diffusion logic, we'll see what happens.
            # BUT, if `DiffuCoder` is a causal or masked diffusion model:
            # If it's Masked (like SEDD/MDLM): it typically preserves unmasked tokens and fills masked ones.
            # If there are NO masks, it might do nothing or just extend?
            
            # Let's add a check: does the identifier change at all?
            # If the output at step 0 (or final) is identical to input, then the model didn't change it.
            # If it's different, we track *when* it became different.
            
            # For the experiment, I will proceed with passing the full sequence (with terrible identifier) as `input_ids`.
            # And I will inspect the returned history.
            
            with torch.no_grad():
                output = model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=input_ids.shape[1] + 1, # +1 to avoid transformers library error (max_length must be > input_length)
                    steps=256,
                    temperature=0.3,
                    top_p=0.95,
                    alg="entropy",
                    alg_temp=0.,
                    output_history=True,
                    return_dict_in_generate=True,
                )
            
            # Check history
            if not hasattr(output, 'history'):
                print(f"No history returned for {row['id']}")
                continue
                
            history = output.history # List of tensors, usually [step_0, step_1, ... step_T]
            # or [step_T, ..., step_0]? Usually diffusion goes T -> 0 (noise to data).
            # But the 'history' often stores the *predicted x_0* at each step or result of sampling.
            # If it's reverse process:
            # history[0] might be pure noise or initial state.
            # history[-1] is final result.
            
            # We want to find the first step `s` where `tokens[start:end]` != `terrible_tokens`.
            # Note: history might contain embeddings or logits? No, `diffusion_generate` usually returns token ids in history if requested, or seqs.
            # My inspection script tracks this. I'll assume token IDs for now.
            
            # Also, need to verify order. 
            # Let's assume index 0 is the start of generation (noise/mask) and -1 is result.
            # If we injected the terrible identifier *into the input*, and the model respects inputs (unmasked),
            # it might be kept fixed throughout if not masked?
            # UNLESS the user implies we should MASK it, but they said "replace MASK with terrible identifier".
            # This is the tricky part.
            
            # Interpret "replace MASK with terrible identifier... then convert to token... then use diffucoder... record change".
            # Maybe the terrible identifier is treated as a generic placeholder that SHOULD be overwritten?
            # But the model doesn't know "terrible_var" is a placeholder unless it's a mask token.
            
            # Maybe the User thinks DiffuCoder works like a standard diffusion where everything is subject to change (denoising)?
            # If I provide a fully realized sequence, strictly speaking, a Diffusion model *could* default to fixing it if it's "clean".
            # But if we force a diffusion process (e.g. add noise then denoise), it might change.
            # Since `diffucoder.py` `diffusion_generate` doesn't seem to take an argument for "add noise level",
            # it likely does generation from scratch?
            # Wait, `run_diffucoder_noise_exp.py` uses `<|mask|>` tokens.
            # If I don't use mask tokens, does `diffusion_generate` overwrite anything?
            
            # Possibility: The user *assumes* the model will change it.
            # I will implement the tracker. If the identifier *never* changes, I will report that (step = None).
            # If it changes immediately (step 0), I report 0.
            
            change_step = None
            first_step_differs = False
            
            # Ensure history entries are accessible
            # history is likely a list of tensors of shape (1, seq_len)
            
            # Check if history[0] matches the terrible identifier?
            # If history[0] is pure noise (random tokens), it DEFINITELY won't match.
            # If history[-1] (final) matches, then it was restored?
            
            # Wait, if `history[0]` is noise, then it effectively "changed" immediately from the input?
            # The User says: "record... terrible naming identifier's corresponding token change time".
            # This implies the identifier starts as "terrible" in the model's state and then changes?
            # This only happens if we Initialize the diffusion chain with the Input.
            # But `diffusion_generate` (if standard) initializes with Mask/Noise.
            
            # If DiffuCoder is a Masked Diffusion model (e.g. MDLM/SEDD/DiffuSeq):
            # It usually starts with All Masks or Specific Masks.
            # If I give it `terrible_var`, and it's NOT a mask token, the model might Treat it as Observed (Fixed).
            # If so, it will NEVER change.
            
            # This implies I might need to intervene to FORCE it to be treated as changeable?
            # OR the user implies "replace MASK with terrible identifier" IN THE DATA, 
            # but maybe we should still treating it as a mask for the model?
            # NO, "replace MASK ... with default terrible ... then convert ... then use diffucoder".
            # It implies the *input to the model* has the terrible identifier.
            
            # I will trust the user's premise for now but log careful details.
            # If the identifier is preserved perfectly in 100% of cases, I'll stop and report.
            
            # Create a subdirectory for this specific data point
            data_dir = os.path.join(experiment_dir, f"data_{row['id']}")
            os.makedirs(data_dir, exist_ok=True)

            for step_idx, step_tensor in enumerate(history):
                # step_tensor shape (1, seq_len)
                step_tokens = step_tensor[0].tolist()
                
                # Decode the full sequence for this step
                decoded_step_text = tokenizer.decode(step_tokens, skip_special_tokens=True)
                
                # Save each step to a .java file
                # Format: [file_id]_step[number]_[date]_[time].java
                step_filename = f"data_{row['id']}_step{step_idx}_{timestamp}.java"
                step_path = os.path.join(data_dir, step_filename)
                
                with open(step_path, "w", encoding="utf-8") as f:
                    f.write(decoded_step_text)

                # Check for identifier change
                if change_step is None:
                    if len(step_tokens) != len(input_ids_list):
                        change_step = step_idx
                    else:
                        current_segment = step_tokens[start_idx:end_idx]
                        if current_segment != original_id_tokens:
                            change_step = step_idx
            
            if change_step is not None:
                print(f"  Changed at step {change_step}")
                results.append({'id': row['id'], 'change_step': change_step})
            else:
                print(f"  No change detected (steps checked: {len(history)})")
                results.append({'id': row['id'], 'change_step': -1}) # -1 means never changed (or preserved)
            
        except Exception as e:
            print(f"Error processing {row['id']}: {e}")
            continue

    # Save results
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(results_csv, index=False)
        print(f"Saved results to {results_csv}")
        
        # Calculate average
        # process valid changes
        valid_changes = res_df[res_df['change_step'] != -1]
        if not valid_changes.empty:
            avg_step = valid_changes['change_step'].mean()
            print(f"Average change step: {avg_step:.2f}")
        else:
            print("No changes detected in any samples.")
    else:
        print("No results generated.")

if __name__ == "__main__":
    run_experiment()
