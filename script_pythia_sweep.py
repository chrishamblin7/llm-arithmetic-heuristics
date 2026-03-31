"""
Pythia Heuristics Scaling Sweep
===============================
Runs the Nikankin heuristic analysis pipeline across Pythia model sizes
and/or training checkpoints to measure how arithmetic heuristic count
scales with compute and parameters.

Usage:
    python script_pythia_sweep.py --experiment both --output-dir ./results
"""

import argparse
import gc
import json
import logging
import os
import pickle
import random
import re
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

from component import Component
from eap.attr_patching import node_attribution_patching
from evaluation_utils import model_accuracy, model_accuracy_on_simple_prompts
from general_utils import (
    generate_activations,
    get_hook_dim,
    get_model_consts,
    get_neuron_importance_scores,
    load_model,
    safe_eval,
    set_deterministic,
)
from heuristics_analysis import (
    heuristic_class_knockout_experiment,
    prompt_knockout_experiment,
)
from heuristics_classification import (
    HeuristicAnalysisData,
    classify_heuristic_neurons,
    load_heuristic_classes,
)
from prompt_generation import (
    OPERATORS,
    OPERATOR_NAMES,
    _get_operand_range,
    _maximize_unique_answers,
    generate_all_prompts_for_operator,
    generate_prompts,
    separate_prompts_and_answers,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SEED = 42
MAX_OP = 300
OP_RANGES = {"+": (0, MAX_OP), "-": (0, MAX_OP), "*": (0, MAX_OP), "/": (1, MAX_OP)}
HEURISTIC_MATCH_THRESHOLD = 0.6
MIN_ACCURACY_FOR_ANALYSIS = 0.05

CHECKPOINT_SWEEP_MODELS = [
    f"pythia-6.9b-step{step}" for step in range(23000, 144000, 10000)
]

SIZE_SWEEP_MODELS = [
    "pythia-410m-step143000",
    "pythia-1b-step143000",
    "pythia-1.4b-step143000",
    "pythia-2.8b-step143000",
    "pythia-6.9b-step143000",
]

FOCUS_OPERATORS = [0, 3]  # addition (+) and division (/)


@dataclass
class ModelSweepResult:
    model_name: str
    n_params_approx: float
    training_step: int
    accuracy_per_op: Dict[str, float]
    heuristic_counts: Dict[str, Dict[str, int]]  # op -> {map_type -> count}
    heuristic_neuron_counts: Dict[str, Dict[str, int]]
    heuristic_type_breakdown: Dict[str, Dict[str, int]]  # op -> {type -> count}
    prompt_knockout_results: Dict[str, Dict]
    first_heuristics_layer: int
    wall_time_seconds: float


def estimate_params(model_name: str) -> float:
    """Rough parameter count from model name."""
    for tok, val in [
        ("12b", 12e9), ("6.9b", 6.9e9), ("2.8b", 2.8e9),
        ("1.4b", 1.4e9), ("1b", 1e9), ("410m", 410e6),
        ("160m", 160e6), ("70m", 70e6),
    ]:
        if tok in model_name:
            return val
    return 0.0


def parse_step(model_name: str) -> int:
    m = re.search(r"step(\d+)", model_name)
    return int(m.group(1)) if m else 0


def detect_first_heuristics_layer(
    model, model_name: str, model_path: str, output_dir: str
) -> int:
    """
    Detect the first layer where a linear probe on the residual stream
    can predict the arithmetic result above chance. Uses a fast heuristic:
    run attribution patching and find the first layer whose top neurons
    have substantial indirect effect (>10% of the max layer's effect).

    Falls back to n_layers // 2 if detection fails.
    """
    cache_path = os.path.join(output_dir, f"{model_name}_first_layer.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)["first_heuristics_layer"]

    n_layers = model.cfg.n_layers
    fallback = n_layers // 2

    try:
        scores_path = os.path.join(
            output_dir, "data", model_name,
            "addition_node_attribution_scores.pt",
        )
        if os.path.exists(scores_path):
            attr_scores = torch.load(scores_path, map_location="cpu")
        else:
            log.info("Running quick attribution patching for layer detection...")
            model_consts = get_model_consts(model_name)
            set_deterministic(SEED)
            prompts = generate_all_prompts_for_operator(
                "+", 0, min(MAX_OP, 100),
                (0, model_consts.max_single_token),
            )
            answers = [str(int(eval(p[:-1]))) for p in prompts]

            correct = []
            for i in range(0, len(prompts), 32):
                batch = prompts[i:i+32]
                batch_ans = answers[i:i+32]
                tokens = model.to_tokens(batch, prepend_bos=True)
                logits = model(tokens, return_type="logits")
                preds = logits[:, -1, :].argmax(-1)
                ans_tokens = model.to_tokens(batch_ans, prepend_bos=False).squeeze(1)
                for j in range(len(batch)):
                    if preds[j].item() == ans_tokens[j].item():
                        correct.append((batch[j], batch_ans[j]))
            if len(correct) < 10:
                log.warning("Too few correct prompts for layer detection, using fallback")
                return fallback

            correct = correct[:50]
            all_pa = correct.copy()
            random.shuffle(all_pa)
            corrupt = all_pa[:len(correct)]

            attr_scores = node_attribution_patching(
                model, correct, corrupt,
                attributed_hook_names=["mlp.hook_post"],
                metric="IE", batch_size=1, verbose=False,
            )

        layer_effects = []
        for layer in range(n_layers):
            key = f"blocks.{layer}.mlp.hook_post"
            if key in attr_scores:
                effect = attr_scores[key][:, -1].nan_to_num(0).abs().mean().item()
            else:
                effect = 0.0
            layer_effects.append(effect)

        max_effect = max(layer_effects) if layer_effects else 0
        if max_effect == 0:
            detected = fallback
        else:
            threshold = 0.10 * max_effect
            detected = fallback
            for i, e in enumerate(layer_effects):
                if e >= threshold:
                    detected = i
                    break

    except Exception as exc:
        log.warning(f"Layer auto-detection failed ({exc}), using fallback {fallback}")
        detected = fallback

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"first_heuristics_layer": detected, "layer_effects": layer_effects}, f)

    log.info(f"Detected first_heuristics_layer = {detected} for {model_name}")
    return detected


def compute_accuracy(model, model_name, model_consts, data_dir):
    """Compute per-operator accuracy and return dict."""
    acc_path = os.path.join(data_dir, "accuracy.pt")
    if os.path.exists(acc_path):
        return torch.load(acc_path, map_location="cpu")

    acc_dict = {}
    for operator in OPERATORS:
        min_op = 1 if operator == "/" else 0
        prompts = generate_all_prompts_for_operator(
            operator, min_op, MAX_OP,
            (0, model_consts.max_single_token),
        )
        answers = [str(int(eval(p[:-1]))) for p in prompts]
        acc = model_accuracy(model, prompts, answers, None, verbose=False)
        log.info(f"  {model_name} accuracy on '{operator}': {acc:.4f}")
        acc_dict[operator] = acc
    torch.save(acc_dict, acc_path)
    return acc_dict


def load_or_generate_prompts(model, model_consts, model_name, data_dir):
    """Generate correct prompts for analysis, caching to disk."""
    prompts_path = os.path.join(data_dir, f"large_prompts_and_answers_max_op={MAX_OP}.pkl")
    if os.path.exists(prompts_path):
        with open(prompts_path, "rb") as f:
            large_pa = pickle.load(f)
    else:
        large_pa = generate_prompts(
            model, operand_ranges=OP_RANGES, correct_prompts=True,
            num_prompts_per_operator=None,
            single_token_number_range=(0, model_consts.max_single_token),
        )
        with open(prompts_path, "wb") as f:
            pickle.dump(large_pa, f)

    set_deterministic(SEED)
    wanted_size = 50
    filtered = []
    for pa in large_pa:
        new_pa = []
        for p, a in pa:
            op1, op2 = tuple(map(int, re.findall(r"\d+", p)))[-2:]
            if op1 > 5 and op2 > 5 and int(a) > 2:
                new_pa.append((p, a))
        if len(new_pa) < wanted_size:
            new_pa = new_pa + random.sample(pa, k=max(0, wanted_size - len(new_pa)))
        filtered.append(new_pa)
    correct_pa = [_maximize_unique_answers(pa, k=wanted_size) for pa in filtered]
    return large_pa, correct_pa


def run_attribution_patching(model, model_name, operator_idx, correct_pa, data_dir):
    """Run attribution patching for a single operator and save scores."""
    scores_path = os.path.join(
        data_dir, f"{OPERATOR_NAMES[operator_idx]}_node_attribution_scores.pt"
    )
    if os.path.exists(scores_path):
        log.info(f"  Attribution scores already exist for {OPERATOR_NAMES[operator_idx]}")
        return

    set_deterministic(SEED)
    pa = correct_pa[operator_idx]
    corrupt_pa = random.sample(sum(correct_pa, []), k=len(pa))

    log.info(f"  Running attribution patching for {OPERATOR_NAMES[operator_idx]}...")
    scores = node_attribution_patching(
        model, pa, corrupt_pa,
        attributed_hook_names=["mlp.hook_post"],
        metric="IE", batch_size=1, verbose=True,
    )
    torch.save(scores, scores_path)


def run_heuristic_classification(
    model, model_name, model_consts, operator_idx,
    activation_map_type, first_heuristics_layer, data_dir,
):
    """Classify heuristic neurons for one operator and one map type."""
    results_path = os.path.join(
        data_dir,
        f"{OPERATOR_NAMES[operator_idx]}_heuristic_matches_dict_{activation_map_type}_maps.pt",
    )
    if os.path.exists(results_path):
        log.info(
            f"  Heuristic classification exists for "
            f"{OPERATOR_NAMES[operator_idx]} ({activation_map_type})"
        )
        return

    log.info(
        f"  Classifying heuristics for {OPERATOR_NAMES[operator_idx]} "
        f"({activation_map_type})..."
    )

    min_op = 1 if operator_idx == 3 else 0
    op1_op2_pairs = torch.tensor(sorted([
        (op1, op2)
        for op1 in range(min_op, MAX_OP)
        for op2 in _get_operand_range(
            OPERATORS[operator_idx], op1, min_op, MAX_OP,
            model_consts.max_single_token,
        )
    ]))
    prompts = [
        f"{op1}{OPERATORS[operator_idx]}{op2}=" for (op1, op2) in op1_op2_pairs
    ]

    neuron_importance_scores = get_neuron_importance_scores(
        model, model_name, operator_idx=operator_idx, pos=-1,
    )
    heuristic_neurons = []
    for layer in range(first_heuristics_layer, model.cfg.n_layers):
        top_indices = neuron_importance_scores[layer].topk(
            model_consts.topk_neurons_per_layer
        ).indices.tolist()
        heuristic_neurons += [(layer, n) for n in top_indices]

    k_activations = generate_activations(
        model, prompts,
        [Component("mlp_post", layer=l) for l in range(model.cfg.n_layers)],
        pos=-1, batch_size=32,
    )
    k_prompts_activations = {
        (layer, neuron): k_activations[layer][:, neuron]
        for (layer, neuron) in heuristic_neurons
    }

    kv_prompts_activations = {}
    results_for_all = [
        str(safe_eval(f"{op1}{OPERATORS[operator_idx]}{op2}"))
        for (op1, op2) in op1_op2_pairs
    ]
    results_labels = model.to_tokens(results_for_all, prepend_bos=False).view(-1)
    for (layer, neuron) in heuristic_neurons:
        v_logits = (
            model.blocks[layer].mlp.W_out[neuron].to(model.cfg.device)
            @ model.W_U.to(model.cfg.device)
        )
        logits_for_pairs = v_logits[results_labels.to(model.cfg.device)].cpu()
        kv_prompts_activations[(layer, neuron)] = (
            k_prompts_activations[(layer, neuron)] * logits_for_pairs
        )

    prompts_activations = (
        k_prompts_activations
        if activation_map_type == "K"
        else kv_prompts_activations
    )

    top_op_indices = {
        (l, n): op1_op2_pairs[
            prompts_activations[(l, n)].topk(len(prompts_activations[(l, n)])).indices.cpu().numpy()
        ].tolist()
        for (l, n) in heuristic_neurons
    }
    top_results = {}
    for (l, n) in top_op_indices:
        top_results[(l, n)] = [
            safe_eval(f"{op1}{OPERATORS[operator_idx]}{op2}")
            for (op1, op2) in top_op_indices[(l, n)]
        ]

    hdata = HeuristicAnalysisData()
    hdata.also_check_bottom_results = model_consts.mlp_activations_also_negative
    hdata.op1_op2_pairs = op1_op2_pairs
    hdata.top_op1_op2_indices = top_op_indices
    hdata.top_results = top_results
    hdata.max_op = MAX_OP
    hdata.max_single_token = model_consts.max_single_token
    hdata.operator_idx = operator_idx
    hdata.k_per_heuristic_cache = {}

    heuristic_classes = classify_heuristic_neurons(
        heuristic_neurons, hdata, verbose=True,
    )
    torch.save(heuristic_classes, results_path)

    del k_activations, k_prompts_activations, kv_prompts_activations
    torch.cuda.empty_cache()


def count_heuristics(data_dir, operator_idx, map_type, threshold=HEURISTIC_MATCH_THRESHOLD):
    """Count classified heuristics above threshold and break down by type."""
    hc = load_heuristic_classes(data_dir, operator_idx, map_type)
    total_heuristics = 0
    total_neurons = set()
    type_counts = {}

    for name, entries in hc.items():
        above = [(l, n, s) for (l, n, s) in entries if s >= threshold]
        if not above:
            continue
        total_heuristics += 1
        for l, n, s in above:
            total_neurons.add((l, n))

        if "mod" in name:
            htype = "modulo"
        elif "range" in name:
            htype = "range"
        elif "pattern" in name:
            htype = "pattern"
        elif "same_operand" in name:
            htype = "identical_operands"
        elif "multi_value" in name or "value" in name:
            htype = "value"
        else:
            htype = "other"
        type_counts[htype] = type_counts.get(htype, 0) + 1

    return total_heuristics, len(total_neurons), type_counts


def run_prompt_knockout(
    model, model_name, model_consts, operator_idx,
    correct_pa, data_dir, map_type="HYBRID",
):
    """Run the prompt knockout experiment for one operator."""
    results_path = os.path.join(
        data_dir,
        f"{OPERATOR_NAMES[operator_idx]}_prompt_ablation_results_thres={HEURISTIC_MATCH_THRESHOLD}_{map_type}_maps.pt",
    )
    if os.path.exists(results_path):
        log.info(f"  Prompt knockout exists for {OPERATOR_NAMES[operator_idx]}")
        loaded = torch.load(results_path, map_location="cpu")
        return {
            "neuron_limits": list(loaded[0]),
            "baseline": loaded[1].tolist(),
            "ablated": loaded[2].tolist(),
            "neuron_counts": loaded[3].tolist(),
            "control": loaded[4].tolist(),
        }

    log.info(f"  Running prompt knockout for {OPERATOR_NAMES[operator_idx]}...")
    set_deterministic(SEED)

    neuron_hard_limits = list(range(0, model_consts.topk_neurons_per_layer // 2 + 1, 5))
    baseline_results = torch.zeros(len(neuron_hard_limits))
    ablated_results = torch.zeros(len(neuron_hard_limits))
    ablated_neuron_counts = torch.zeros(len(neuron_hard_limits))
    control_results = torch.zeros(len(neuron_hard_limits))

    for idx, limit in enumerate(neuron_hard_limits):
        neuron_scores = get_neuron_importance_scores(
            model, model_name, operator_idx=operator_idx, pos=-1,
        )
        all_top_neurons = []
        for layer in range(model_consts.first_heuristics_layer, model.cfg.n_layers):
            top = neuron_scores[layer].topk(model_consts.topk_neurons_per_layer).indices.tolist()
            all_top_neurons += [(layer, n) for n in top]

        hc = load_heuristic_classes(data_dir, operator_idx, map_type)
        hc = {
            name: [(l, n, s) for (l, n, s) in lns if s >= HEURISTIC_MATCH_THRESHOLD]
            for name, lns in hc.items()
        }
        baseline, ablated, avg_count, control = prompt_knockout_experiment(
            hc, model, correct_pa[operator_idx],
            neuron_count_hard_limit_per_layer=limit,
            all_top_neurons=all_top_neurons,
            metric_fn=model_accuracy,
        )
        baseline_results[idx] = baseline
        ablated_results[idx] = ablated
        ablated_neuron_counts[idx] = avg_count
        control_results[idx] = control

    torch.save(
        (neuron_hard_limits, baseline_results, ablated_results,
         ablated_neuron_counts, control_results),
        results_path,
    )
    return {
        "neuron_limits": neuron_hard_limits,
        "baseline": baseline_results.tolist(),
        "ablated": ablated_results.tolist(),
        "neuron_counts": ablated_neuron_counts.tolist(),
        "control": control_results.tolist(),
    }


def analyze_single_model(
    model_name: str,
    model_path: str,
    output_dir: str,
    device: str = "cuda",
) -> ModelSweepResult:
    """Full analysis pipeline for one model."""
    t0 = time.time()
    log.info(f"=== Starting analysis: {model_name} ===")

    data_dir = os.path.join(output_dir, "data", model_name)
    os.makedirs(data_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "summaries", f"{model_name}.json")
    if os.path.exists(summary_path):
        log.info(f"  Summary already exists, loading from cache")
        with open(summary_path) as f:
            return ModelSweepResult(**json.load(f))

    set_deterministic(SEED)
    torch.set_grad_enabled(False)

    log.info(f"  Loading model {model_name}...")
    model = load_model(model_name, model_path, device, extra_hooks=False)

    first_layer = detect_first_heuristics_layer(model, model_name, model_path, output_dir)
    model_consts = get_model_consts(model_name, first_heuristics_layer=first_layer)
    log.info(
        f"  Model consts: first_layer={model_consts.first_heuristics_layer}, "
        f"topk={model_consts.topk_neurons_per_layer}"
    )

    log.info("  Computing accuracy...")
    acc_dict = compute_accuracy(model, model_name, model_consts, data_dir)

    log.info("  Loading/generating prompts...")
    large_pa, correct_pa = load_or_generate_prompts(model, model_consts, model_name, data_dir)

    heuristic_counts = {}
    heuristic_neuron_counts = {}
    type_breakdowns = {}
    knockout_results = {}

    for op_idx in FOCUS_OPERATORS:
        op_name = OPERATOR_NAMES[op_idx]
        op_sym = OPERATORS[op_idx]

        if acc_dict.get(op_sym, 0) < MIN_ACCURACY_FOR_ANALYSIS:
            log.info(f"  Skipping {op_name}: accuracy {acc_dict.get(op_sym, 0):.4f} < {MIN_ACCURACY_FOR_ANALYSIS}")
            heuristic_counts[op_name] = {"K": 0, "KV": 0}
            heuristic_neuron_counts[op_name] = {"K": 0, "KV": 0}
            type_breakdowns[op_name] = {}
            knockout_results[op_name] = {}
            continue

        run_attribution_patching(model, model_name, op_idx, correct_pa, data_dir)

        for map_type in ["K", "KV"]:
            run_heuristic_classification(
                model, model_name, model_consts, op_idx,
                map_type, first_layer, data_dir,
            )

        counts_k, neurons_k, types_k = count_heuristics(data_dir, op_idx, "K")
        counts_kv, neurons_kv, types_kv = count_heuristics(data_dir, op_idx, "KV")

        heuristic_counts[op_name] = {"K": counts_k, "KV": counts_kv}
        heuristic_neuron_counts[op_name] = {"K": neurons_k, "KV": neurons_kv}

        merged_types = {}
        for t in set(list(types_k.keys()) + list(types_kv.keys())):
            merged_types[t] = types_k.get(t, 0) + types_kv.get(t, 0)
        type_breakdowns[op_name] = merged_types

        try:
            ko = run_prompt_knockout(
                model, model_name, model_consts, op_idx,
                correct_pa, data_dir, map_type="HYBRID",
            )
            knockout_results[op_name] = ko
        except Exception as exc:
            log.warning(f"  Prompt knockout failed for {op_name}: {exc}")
            knockout_results[op_name] = {}

    elapsed = time.time() - t0
    result = ModelSweepResult(
        model_name=model_name,
        n_params_approx=estimate_params(model_name),
        training_step=parse_step(model_name),
        accuracy_per_op={k: float(v) for k, v in acc_dict.items()},
        heuristic_counts=heuristic_counts,
        heuristic_neuron_counts=heuristic_neuron_counts,
        heuristic_type_breakdown=type_breakdowns,
        prompt_knockout_results=knockout_results,
        first_heuristics_layer=first_layer,
        wall_time_seconds=elapsed,
    )

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(asdict(result), f, indent=2)

    log.info(f"=== Finished {model_name} in {elapsed:.0f}s ===")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result


def build_model_list(experiment: str) -> list:
    """Build the deduplicated model list for the requested experiment."""
    if experiment == "checkpoint":
        return list(CHECKPOINT_SWEEP_MODELS)
    elif experiment == "size":
        return list(SIZE_SWEEP_MODELS)
    elif experiment == "both":
        return CHECKPOINT_SWEEP_MODELS + [
            m for m in SIZE_SWEEP_MODELS if m not in CHECKPOINT_SWEEP_MODELS
        ]
    else:
        raise ValueError(f"Unknown experiment type: {experiment}")


def aggregate_results(output_dir: str, model_list: list):
    """Merge per-model summary JSONs into a single aggregated file."""
    all_results = []
    for model_name in model_list:
        summary_path = os.path.join(output_dir, "summaries", f"{model_name}.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                all_results.append(json.load(f))
        else:
            all_results.append({"model_name": model_name, "error": "summary not found"})

    agg_path = os.path.join(output_dir, "aggregated_results.json")
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Aggregated {len(all_results)} results -> {agg_path}")
    return agg_path


def run_sweep(experiment: str, output_dir: str, model_path: str, device: str,
              models_override: list = None):
    """Run the sweep across models (serial, single-GPU)."""
    model_list = models_override if models_override else build_model_list(experiment)

    log.info(f"Sweep: {experiment}, {len(model_list)} models, output -> {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    for model_name in model_list:
        try:
            analyze_single_model(model_name, model_path, output_dir, device)
        except Exception as exc:
            log.error(f"Failed on {model_name}: {exc}", exc_info=True)

    aggregate_results(output_dir, model_list)


def main():
    parser = argparse.ArgumentParser(description="Pythia Heuristics Scaling Sweep")
    parser.add_argument(
        "--experiment", choices=["checkpoint", "size", "both"], default="both",
        help="Which sweep to run",
    )
    parser.add_argument("--output-dir", default="./results", help="Output directory")
    parser.add_argument(
        "--model-path", default=None,
        help="Cache directory for HF model weights",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gpu-id", type=int, default=None,
        help="Pin to a specific GPU (sets CUDA_VISIBLE_DEVICES)",
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated model names to process (overrides --experiment list)",
    )
    args = parser.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = "cuda:0"
    else:
        device = args.device

    models_override = args.models.split(",") if args.models else None

    run_sweep(args.experiment, args.output_dir, args.model_path, device,
              models_override=models_override)


if __name__ == "__main__":
    main()
