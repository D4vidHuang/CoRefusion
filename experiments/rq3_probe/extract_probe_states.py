#!/usr/bin/env python3
"""
RQ3 v2 — GPU hidden-state extraction for the linear-probe redesign.

This is the ONLY part of the new RQ3 that needs a GPU. It re-runs
DiffuCoder-7B-Base over the RefineID snippets and PERSISTS the residual stream
at the target identifier position so that all probing / statistics / plotting
can be done offline on CPU (train_probes.py, plot_rq3.py).

Two extractions, written to --out-dir:

  EXP1  (depth axis, single forward pass) -> exp1_states.npz + exp1_meta.csv
      For each snippet we clamp the target position to GOOD (developer name) or
      BAD (a length/sub-word-count-matched MISPLACED REAL identifier from a
      different snippet) and read all 29 residual-stream layers at the FIRST
      target sub-token. We also build a SCRAMBLED-CONTEXT twin of each (same
      clamped token, surrounding non-target tokens shuffled) so the trainer can
      report the double difference AUC_intact - AUC_scrambled = the *contextual*
      name-fit signal (token-intrinsic form cancels out). The old throw-away
      smell vocab (x/tmp/foo ...) is extracted too, but only as an APPENDIX
      "length-shortcut ceiling" contrast -- never the headline.

  EXP2  (denoising-step axis, real generation) -> exp2_states.npz + exp2_meta.csv
      Target masked, model fills over --steps denoising steps. At every step we
      save the hidden state at each target site AND a still_masked flag, the
      first-confident step, and whether the model's OWN final fill is EM-correct.
      The trainer evaluates a "will-succeed" probe ONLY on still-masked positions
      (anti-tautology) and overlays it with the commitment CDF -> the DCL metric.

Design rationale + the statistical protocol live in docs/rq3_redesign.md.
Verified data universe: data/test_filtered_1024.csv = 230 snippets / 1055 sites
/ 196 distinct good names / 64 package prefixes (the paper's "926" was a stale
position count and is retired).

Python 3.11 safe (no back-slashes inside f-string expressions).
"""

import os
import re
import sys
import json
import time
import hashlib
import random
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# DiffuCoder needs torchvision symbols present even on a text-only box.
# (Same shim the legacy experiment_a/b/c scripts use.)
# ---------------------------------------------------------------------------
class _MockModule:
    def __getattr__(self, name):
        return _MockModule()

    def __call__(self, *a, **k):
        return _MockModule()


for _m in ("torchvision", "torchvision.ops", "torchvision.transforms"):
    sys.modules.setdefault(_m, _MockModule())
if not hasattr(torch.ops, "torchvision"):
    class _DummyOps:
        def nms(self, *a, **k):
            return torch.tensor([])
    torch.ops.torchvision = _DummyOps()

from transformers import AutoTokenizer, AutoModel  # noqa: E402


# ===========================================================================
# helpers
# ===========================================================================
def split_subtokens(name):
    """camelCase / snake / digit aware identifier split -> list of word pieces."""
    name = str(name).strip()
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", name)
    return [p for p in parts if p] or [name]


def subword_count(name):
    return len(split_subtokens(name))


def package_prefix(code, levels=3):
    m = re.search(r"package\s+([\w\.]+)\s*;", str(code))
    if not m:
        return "<none>"
    return ".".join(m.group(1).split(".")[:levels])


def find_all_subseq(seq, sub):
    """All start indices where list `sub` occurs in list `seq`."""
    n, m = len(seq), len(sub)
    out = []
    if m == 0 or m > n:
        return out
    i = 0
    while i <= n - m:
        if seq[i:i + m] == sub:
            out.append(i)
            i += m
        else:
            i += 1
    return out


def locate_token(tokenizer, input_ids_list, name):
    """Return (all_start_indices, tok_len) for `name` in the id list.

    Tries the bare encoding and a leading-space encoding (BPE space marker)."""
    for variant in (name, " " + name):
        toks = tokenizer.encode(variant, add_special_tokens=False)
        starts = find_all_subseq(input_ids_list, toks)
        if starts:
            return starts, len(toks)
    return [], 0


def nontarget_hash(input_ids_list, target_positions):
    """Hash of all token ids EXCEPT the target-identifier token positions."""
    keep = [tid for i, tid in enumerate(input_ids_list) if i not in target_positions]
    return hashlib.md5(json.dumps(keep).encode()).hexdigest()[:12]


def scramble_context(input_ids_list, target_positions, rng):
    """Return a new id list with non-target positions permuted, target fixed."""
    target_positions = set(target_positions)
    movable = [i for i in range(len(input_ids_list)) if i not in target_positions]
    shuffled = movable[:]
    rng.shuffle(shuffled)
    out = list(input_ids_list)
    for src, dst in zip(movable, shuffled):
        out[dst] = input_ids_list[src]
    return out


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    out = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        out[i, : remainder[i]] += 1
    return out


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel


# old throw-away smell vocab — APPENDIX contrast only (the length-shortcut ceiling)
SMELL_VOCAB = ["x", "a", "n", "i", "b", "y",
               "tmp", "val", "foo", "res", "data", "var",
               "temp1", "val1", "item", "stuff", "obj1"]


def build_bad_pool(targets, seed=0):
    """For each snippet pick a length/sub-word-matched MISPLACED real identifier.

    Returns list aligned with `targets`: (bad_name, match_type)."""
    rng = random.Random(seed)
    by_k = {}
    by_len = {}
    for t in targets:
        by_k.setdefault(subword_count(t), []).append(t)
        by_len.setdefault(len(t), []).append(t)
    out = []
    generic_k = {1: "handler", 2: "requestHandler", 3: "defaultRequestHandler",
                 4: "defaultRequestHandlerFactory"}
    for idx, t in enumerate(targets):
        k = subword_count(t)
        # same sub-word count, different string (and ideally different snippet)
        cand = [c for c in by_k.get(k, []) if c != t]
        if cand:
            out.append((rng.choice(cand), "k"))
            continue
        cand = [c for c in by_len.get(len(t), []) if c != t]
        if cand:
            out.append((rng.choice(cand), "lenbucket"))
            continue
        out.append((generic_k.get(k, "requestHandler"), "synthetic"))
    return out


# ===========================================================================
# EXP1 — depth axis, single forward pass
# ===========================================================================
def run_exp1(model, tokenizer, df, mask_id, device, out_dir, seed, smell_appendix=True):
    rng = random.Random(seed)
    targets = [str(t).strip() for t in df["target"].tolist()]
    bad_pool = build_bad_pool(targets, seed=seed)

    rows = []          # metadata dicts
    vecs_first = []    # [29, d] per row  (first target sub-token)
    vecs_pool = []     # [29, d] per row  (mean over all sites) — robustness only

    n_layers_ref = [None]

    def one_variant(code_ids_list, target_positions, snippet_id, pkg, good_name,
                    bad_name, good_k, bad_k, site_count, first_start, nh,
                    cond, ctx, bad_source, match_type):
        ids = torch.tensor([code_ids_list], device=device)
        am = torch.ones_like(ids)
        with torch.no_grad():
            out = model(ids, attention_mask=am.bool(), output_hidden_states=True)
        hs = out.hidden_states  # tuple length L+1
        if n_layers_ref[0] is None:
            n_layers_ref[0] = len(hs)
        # first sub-token vector across layers
        first_idx = first_start
        v_first = np.stack([h[0, first_idx, :].float().cpu().numpy() for h in hs]).astype(np.float16)
        # mean over ALL target token positions across layers
        v_pool = np.stack([h[0, target_positions, :].float().mean(0).cpu().numpy()
                           for h in hs]).astype(np.float16)
        vecs_first.append(v_first)
        vecs_pool.append(v_pool)
        rows.append(dict(snippet_id=snippet_id, package=pkg, good_name=good_name,
                         bad_name=bad_name, good_k=good_k, bad_k=bad_k,
                         site_count=site_count, first_site_index=first_idx,
                         nontarget_hash=nh, cond=cond, ctx=ctx,
                         bad_source=bad_source, match_type=match_type))

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="EXP1"):
        try:
            snippet_id = row["id"]
            masked_code = str(row["masked_code"])
            good = str(row["target"]).strip()
            if "[MASK]" not in masked_code:
                continue
            bad, match_type = bad_pool[idx]
            pkg = package_prefix(masked_code)
            gk, bk = subword_count(good), subword_count(bad)

            good_code = masked_code.replace("[MASK]", good)
            bad_code = masked_code.replace("[MASK]", bad)
            good_ids = tokenizer(good_code, return_tensors="pt").input_ids[0].tolist()
            bad_ids = tokenizer(bad_code, return_tensors="pt").input_ids[0].tolist()

            g_starts, g_len = locate_token(tokenizer, good_ids, good)
            b_starts, b_len = locate_token(tokenizer, bad_ids, bad)
            if not g_starts or not b_starts:
                continue
            g_pos = [p for s in g_starts for p in range(s, s + g_len)]
            b_pos = [p for s in b_starts for p in range(s, s + b_len)]
            site_count = len(g_starts)
            g_hash = nontarget_hash(good_ids, set(g_pos))
            b_hash = nontarget_hash(bad_ids, set(b_pos))
            # byte-identical context holds iff k matched AND tokenizer aligned
            same_ctx = (g_hash == b_hash)

            # intact
            one_variant(good_ids, g_pos, snippet_id, pkg, good, bad, gk, bk,
                        site_count, g_starts[0], g_hash, "good", "intact", "na", match_type)
            one_variant(bad_ids, b_pos, snippet_id, pkg, good, bad, gk, bk,
                        site_count, b_starts[0], b_hash, "bad", "intact", "misplaced", match_type)
            # scrambled twin (same per-snippet permutation seed for comparability)
            perm_seed = (int(snippet_id) * 9973 + seed) & 0x7fffffff
            g_scr = scramble_context(good_ids, set(g_pos), random.Random(perm_seed))
            b_scr = scramble_context(bad_ids, set(b_pos), random.Random(perm_seed))
            one_variant(g_scr, g_pos, snippet_id, pkg, good, bad, gk, bk,
                        site_count, g_starts[0], g_hash, "good", "scrambled", "na", match_type)
            one_variant(b_scr, b_pos, snippet_id, pkg, good, bad, gk, bk,
                        site_count, b_starts[0], b_hash, "bad", "scrambled", "misplaced", match_type)

            # appendix: throw-away smell (intact only)
            if smell_appendix:
                smell = rng.choice([s for s in SMELL_VOCAB if s != good])
                s_code = masked_code.replace("[MASK]", smell)
                s_ids = tokenizer(s_code, return_tensors="pt").input_ids[0].tolist()
                s_starts, s_len = locate_token(tokenizer, s_ids, smell)
                if s_starts:
                    s_pos = [p for st in s_starts for p in range(st, st + s_len)]
                    one_variant(s_ids, s_pos, snippet_id, pkg, good, smell, gk,
                                subword_count(smell), len(s_starts), s_starts[0],
                                nontarget_hash(s_ids, set(s_pos)), "bad", "intact",
                                "smell", "smell")

            rows[-1]["same_ctx"] = same_ctx
        except Exception as e:  # noqa: BLE001
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                continue
            raise

    meta = pd.DataFrame(rows)
    X_first = np.stack(vecs_first)  # [N, L+1, d]
    X_pool = np.stack(vecs_pool)
    np.savez_compressed(os.path.join(out_dir, "exp1_states.npz"),
                        X_first=X_first, X_pool=X_pool)
    meta.to_csv(os.path.join(out_dir, "exp1_meta.csv"), index=False)
    print("[EXP1] rows=%d  layers=%d  d=%d  -> exp1_states.npz / exp1_meta.csv"
          % (X_first.shape[0], X_first.shape[1], X_first.shape[2]))
    return n_layers_ref[0]


# ===========================================================================
# EXP2 — denoising-step axis, real generation
# ===========================================================================
def run_exp2(model, tokenizer, df, mask_id, device, out_dir, steps, num_mask,
             exp2_layer, conf_thresh=0.8):
    rows = []
    traj = []  # [steps, d] per target position (first sub-token of each site)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="EXP2"):
        try:
            snippet_id = row["id"]
            masked_code = str(row["masked_code"])
            target = str(row["target"]).strip()
            if "[MASK]" not in masked_code:
                continue
            n_sites = masked_code.count("[MASK]")
            mask_str = ("<|mask|>" * num_mask)
            gen_code = masked_code.replace("[MASK]", mask_str)
            enc = tokenizer(gen_code, return_tensors="pt").to(device)
            x = enc.input_ids.clone()
            am = enc.attention_mask
            mask_positions = (x[0] == mask_id).nonzero(as_tuple=True)[0].tolist()
            if not mask_positions:
                continue
            # group consecutive mask positions into sites (num_mask each)
            sites = [mask_positions[i:i + num_mask]
                     for i in range(0, len(mask_positions), num_mask)]
            first_tok = [s[0] for s in sites]  # representative position per site

            num_transfer = get_num_transfer_tokens((x == mask_id).long()[:, :], steps)

            T = steps
            d = model.config.hidden_size if hasattr(model.config, "hidden_size") else None
            site_traj = {p: np.zeros((T, 0), dtype=np.float16) for p in first_tok}
            still_masked = {p: np.ones(T, dtype=bool) for p in first_tok}
            first_conf = {p: -1 for p in first_tok}
            flip_step = {p: -1 for p in first_tok}
            buf = {p: [] for p in first_tok}

            for t in range(T):
                with torch.no_grad():
                    cur_mask = (x == mask_id)
                    out = model(x, attention_mask=am.bool(), output_hidden_states=True)
                    logits = out.logits
                    h = out.hidden_states[exp2_layer][0]  # [seq, d]
                    for p in first_tok:
                        still_masked[p][t] = bool(cur_mask[0, p].item())
                        buf[p].append(h[p, :].float().cpu().numpy().astype(np.float16))
                    if not cur_mask.any():
                        # fill remaining steps with last hidden state, mark unmasked
                        for tt in range(t + 1, T):
                            for p in first_tok:
                                still_masked[p][tt] = False
                                buf[p].append(buf[p][-1])
                        break
                    noisy = add_gumbel_noise(logits, temperature=0.3)
                    x0 = torch.argmax(noisy, dim=-1)
                    p_all = F.softmax(logits.float(), dim=-1)
                    x0_p = torch.gather(p_all, -1, x0.unsqueeze(-1)).squeeze(-1)
                    for p in first_tok:
                        if cur_mask[0, p].item() and x0_p[0, p].item() > conf_thresh and first_conf[p] == -1:
                            first_conf[p] = t
                    conf = torch.where(cur_mask, x0_p, torch.tensor(-np.inf, device=device))
                    k = int(num_transfer[0, t].item()) if num_transfer.shape[1] > t else 0
                    k = min(k, int(cur_mask.sum().item()))
                    if k > 0:
                        _, sel = torch.topk(conf[0], k=k)
                        tindex = torch.zeros_like(x0, dtype=torch.bool)
                        tindex[0, sel] = True
                        x[tindex] = x0[tindex]
                        for j in sel.tolist():
                            if j in flip_step and flip_step[j] == -1:
                                flip_step[j] = t

            # decode final fill at the FIRST site, EM vs developer target
            first_site = sites[0]
            pred_ids = x[0, first_site[0]:first_site[0] + num_mask].tolist()
            pred = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
            em = int(pred == target)

            for p in first_tok:
                arr = np.stack(buf[p])  # [T, d]
                traj.append(arr)
                rows.append(dict(snippet_id=snippet_id, position=p,
                                 package=package_prefix(masked_code),
                                 em_correct=em, first_conf=first_conf[p],
                                 flip_step=flip_step[p], n_sites=n_sites,
                                 pred=pred, target=target,
                                 still_masked=json.dumps(still_masked[p].tolist())))
        except Exception as e:  # noqa: BLE001
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                continue
            raise

    meta = pd.DataFrame(rows)
    X = np.stack(traj)  # [Npos, T, d]
    np.savez_compressed(os.path.join(out_dir, "exp2_states.npz"), X=X)
    meta.to_csv(os.path.join(out_dir, "exp2_meta.csv"), index=False)
    print("[EXP2] positions=%d  steps=%d  d=%d  EM=%.3f  -> exp2_states.npz / exp2_meta.csv"
          % (X.shape[0], X.shape[1], X.shape[2], meta["em_correct"].mean()))


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="apple/DiffuCoder-7B-Base",
                    help="HF checkpoint (paper RQ3 uses Base; confirm matches RQ1/RQ2).")
    ap.add_argument("--data", default="data/test_filtered_1024.csv")
    ap.add_argument("--out-dir", default="results/rq3_probe")
    ap.add_argument("--max-snippets", type=int, default=None, help="smoke-test cap")
    ap.add_argument("--steps", type=int, default=32,
                    help="EXP2 denoising steps (paper RQ3=32; match RQ1/RQ2 gen config).")
    ap.add_argument("--num-mask", type=int, default=2, help="mask tokens per site (k=2 canvas).")
    ap.add_argument("--exp2-layer", type=int, default=-1,
                    help="layer index for EXP2 hidden state (-1=last; re-point to Exp1 chosen layer).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-exp1", action="store_true")
    ap.add_argument("--skip-exp2", action="store_true")
    ap.add_argument("--no-smell-appendix", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.data, header=None, names=["id", "masked_code", "target"])
    df = df.reset_index(drop=True)
    if args.max_snippets:
        df = df.head(args.max_snippets).reset_index(drop=True)
    n_sites = int(df["masked_code"].astype(str).str.count(r"\[MASK\]").sum())
    n_pkg = df["masked_code"].map(package_prefix).nunique()
    summary = dict(snippets=len(df), sites=n_sites,
                   distinct_names=df["target"].astype(str).str.strip().nunique(),
                   packages=n_pkg, model=args.model, steps=args.steps,
                   num_mask=args.num_mask, exp2_layer=args.exp2_layer,
                   timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    print("[universe] %s" % json.dumps(summary))
    with open(os.path.join(args.out_dir, "extract_meta.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[load] %s (bf16, trust_remote_code) ..." % args.model)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                      trust_remote_code=True).to(device).eval()
    mask_id = tok.convert_tokens_to_ids("<|mask|>")

    if not args.skip_exp1:
        run_exp1(model, tok, df, mask_id, device, args.out_dir, args.seed,
                 smell_appendix=not args.no_smell_appendix)
    if not args.skip_exp2:
        run_exp2(model, tok, df, mask_id, device, args.out_dir, args.steps,
                 args.num_mask, args.exp2_layer)

    print("[done] outputs in %s" % args.out_dir)


if __name__ == "__main__":
    main()
