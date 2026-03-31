"""
Generate figures and a markdown report from the Pythia heuristics sweep results.

Usage:
    python script_generate_report.py --results-dir ./results --output-dir ./report
"""

import argparse
import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: str) -> List[Dict]:
    path = os.path.join(results_dir, "aggregated_results.json")
    with open(path) as f:
        return json.load(f)


def split_checkpoint_and_size(results: List[Dict]):
    """Separate checkpoint-sweep (same arch, varying step) from size-sweep entries."""
    checkpoint_results = []
    size_results = []
    seen_sizes = set()

    for r in results:
        if "error" in r:
            continue
        name = r["model_name"]
        step = r["training_step"]
        params = r["n_params_approx"]

        if "6.9b" in name:
            checkpoint_results.append(r)

        key = f"{params:.0f}"
        if step == 143000 and key not in seen_sizes:
            size_results.append(r)
            seen_sizes.add(key)

    checkpoint_results.sort(key=lambda r: r["training_step"])
    size_results.sort(key=lambda r: r["n_params_approx"])
    return checkpoint_results, size_results


def fig_accuracy_vs_scale(checkpoint_results, size_results, output_dir):
    """Figure 1: Accuracy vs scale (log-log)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if checkpoint_results:
        ax = axes[0]
        steps = [r["training_step"] for r in checkpoint_results]
        for op in ["+", "/"]:
            accs = [r["accuracy_per_op"].get(op, 0) for r in checkpoint_results]
            label = "addition" if op == "+" else "division"
            ax.plot(steps, accs, "o-", label=label)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Accuracy")
        ax.set_title("Pythia-6.9B: Accuracy vs Training Step")
        ax.set_xscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if size_results:
        ax = axes[1]
        params = [r["n_params_approx"] for r in size_results]
        for op in ["+", "/"]:
            accs = [r["accuracy_per_op"].get(op, 0) for r in size_results]
            label = "addition" if op == "+" else "division"
            ax.plot(params, accs, "s-", label=label)
        ax.set_xlabel("Parameters")
        ax.set_ylabel("Accuracy")
        ax.set_title("Pythia Family: Accuracy vs Model Size")
        ax.set_xscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig1_accuracy_vs_scale.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_heuristic_count_vs_scale(checkpoint_results, size_results, output_dir):
    """Figure 2: Number of classified heuristic neurons vs scale."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for results, ax, x_key, x_label, title_suffix in [
        (checkpoint_results, axes[0], "training_step", "Training Step", "Pythia-6.9B"),
        (size_results, axes[1], "n_params_approx", "Parameters", "Pythia Family"),
    ]:
        if not results:
            continue
        xs = [r[x_key] for r in results]
        for op_name in ["addition", "division"]:
            counts = []
            for r in results:
                c = r.get("heuristic_counts", {}).get(op_name, {})
                counts.append(c.get("K", 0) + c.get("KV", 0))
            ax.plot(xs, counts, "o-", label=op_name)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Heuristic Count (K + KV)")
        ax.set_title(f"{title_suffix}: Heuristics vs Scale")
        ax.set_xscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig2_heuristic_count_vs_scale.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_heuristic_vs_accuracy(checkpoint_results, size_results, output_dir):
    """Figure 3: Heuristic count vs accuracy (the quanta relationship)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    all_results = checkpoint_results + size_results
    for op_name, op_sym, marker in [("addition", "+", "o"), ("division", "/", "s")]:
        accs = []
        counts = []
        labels = []
        for r in all_results:
            a = r["accuracy_per_op"].get(op_sym, 0)
            c = r.get("heuristic_counts", {}).get(op_name, {})
            total = c.get("K", 0) + c.get("KV", 0)
            if total > 0:
                accs.append(a)
                counts.append(total)
                labels.append(r["model_name"])

        if accs:
            ax.scatter(counts, accs, marker=marker, s=60, label=op_name, alpha=0.8)
            for i, lbl in enumerate(labels):
                short = lbl.replace("pythia-", "").replace("-step", "@")
                ax.annotate(short, (counts[i], accs[i]), fontsize=6, alpha=0.7)

    ax.set_xlabel("Heuristic Count")
    ax.set_ylabel("Accuracy")
    ax.set_title("Heuristic Count vs Accuracy (Quanta Relationship)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig3_heuristic_vs_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_type_breakdown(checkpoint_results, size_results, output_dir):
    """Figure 4: Breakdown of heuristic types by scale (stacked bar)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for results, ax, x_key, title_suffix in [
        (checkpoint_results, axes[0], "training_step", "Pythia-6.9B Checkpoints"),
        (size_results, axes[1], "n_params_approx", "Pythia Size Sweep"),
    ]:
        if not results:
            continue

        op_name = "addition"
        all_types = set()
        for r in results:
            bd = r.get("heuristic_type_breakdown", {}).get(op_name, {})
            all_types.update(bd.keys())
        all_types = sorted(all_types)

        xs = [str(r[x_key]) for r in results]
        bottom = np.zeros(len(results))

        colors = plt.cm.Set2(np.linspace(0, 1, max(len(all_types), 1)))
        for i, htype in enumerate(all_types):
            vals = np.array([
                r.get("heuristic_type_breakdown", {}).get(op_name, {}).get(htype, 0)
                for r in results
            ], dtype=float)
            ax.bar(xs, vals, bottom=bottom, label=htype, color=colors[i % len(colors)])
            bottom += vals

        ax.set_xlabel(x_key.replace("_", " ").title())
        ax.set_ylabel("Heuristic Count")
        ax.set_title(f"{title_suffix}: Heuristic Type Breakdown ({op_name})")
        ax.legend(fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig4_type_breakdown.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_knockout_across_scales(checkpoint_results, size_results, output_dir):
    """Figure 5: Prompt knockout results across scales."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for results, ax, x_key, title_suffix in [
        (checkpoint_results, axes[0], "training_step", "Pythia-6.9B"),
        (size_results, axes[1], "n_params_approx", "Pythia Family"),
    ]:
        if not results:
            continue

        op_name = "addition"
        xs = [r[x_key] for r in results]
        baseline_vals = []
        ablated_vals = []
        control_vals = []

        for r in results:
            ko = r.get("prompt_knockout_results", {}).get(op_name, {})
            bl = ko.get("baseline", [])
            ab = ko.get("ablated", [])
            ct = ko.get("control", [])
            baseline_vals.append(bl[-1] if bl else 0)
            ablated_vals.append(ab[-1] if ab else 0)
            control_vals.append(ct[-1] if ct else 0)

        ax.plot(xs, baseline_vals, "o-", label="Baseline", color="green")
        ax.plot(xs, ablated_vals, "s-", label="Heuristic Ablated", color="red")
        ax.plot(xs, control_vals, "^-", label="Control Ablated", color="blue")
        ax.set_xlabel(x_key.replace("_", " ").title())
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{title_suffix}: Prompt Knockout ({op_name})")
        ax.set_xscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig5_knockout_across_scales.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def write_markdown_report(
    results, checkpoint_results, size_results,
    fig_paths, output_dir,
):
    """Write a markdown summary report."""
    lines = [
        "# Pythia Heuristics Scaling Sweep — Report",
        "",
        "## Overview",
        "",
        f"Total models analyzed: {len([r for r in results if 'error' not in r])}",
        f"Failed models: {len([r for r in results if 'error' in r])}",
        "",
    ]

    if checkpoint_results:
        lines.append("## Checkpoint Sweep (Pythia-6.9B)")
        lines.append("")
        lines.append("| Step | Acc (+) | Acc (/) | Heuristics (+) | Heuristics (/) |")
        lines.append("|------|---------|---------|----------------|----------------|")
        for r in checkpoint_results:
            a_add = r["accuracy_per_op"].get("+", 0)
            a_div = r["accuracy_per_op"].get("/", 0)
            h_add = r.get("heuristic_counts", {}).get("addition", {})
            h_div = r.get("heuristic_counts", {}).get("division", {})
            h_add_total = h_add.get("K", 0) + h_add.get("KV", 0)
            h_div_total = h_div.get("K", 0) + h_div.get("KV", 0)
            lines.append(
                f"| {r['training_step']:,} | {a_add:.3f} | {a_div:.3f} "
                f"| {h_add_total} | {h_div_total} |"
            )
        lines.append("")

    if size_results:
        lines.append("## Size Sweep")
        lines.append("")
        lines.append("| Model | Params | Acc (+) | Acc (/) | Heuristics (+) | Heuristics (/) | First Layer |")
        lines.append("|-------|--------|---------|---------|----------------|----------------|-------------|")
        for r in size_results:
            p = r["n_params_approx"]
            p_str = f"{p/1e9:.1f}B" if p >= 1e9 else f"{p/1e6:.0f}M"
            a_add = r["accuracy_per_op"].get("+", 0)
            a_div = r["accuracy_per_op"].get("/", 0)
            h_add = r.get("heuristic_counts", {}).get("addition", {})
            h_div = r.get("heuristic_counts", {}).get("division", {})
            h_add_total = h_add.get("K", 0) + h_add.get("KV", 0)
            h_div_total = h_div.get("K", 0) + h_div.get("KV", 0)
            lines.append(
                f"| {r['model_name']} | {p_str} | {a_add:.3f} | {a_div:.3f} "
                f"| {h_add_total} | {h_div_total} | {r['first_heuristics_layer']} |"
            )
        lines.append("")

    lines.append("## Figures")
    lines.append("")
    for path in fig_paths:
        fname = os.path.basename(path)
        lines.append(f"![{fname}]({fname})")
        lines.append("")

    lines.append("## Key Findings")
    lines.append("")
    lines.append("_(To be filled in after reviewing the figures above.)_")
    lines.append("")

    report_path = os.path.join(output_dir, "REPORT.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Generate Pythia sweep report")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--output-dir", default="./report")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = load_results(args.results_dir)
    checkpoint_results, size_results = split_checkpoint_and_size(results)

    fig_paths = []
    fig_paths.append(fig_accuracy_vs_scale(checkpoint_results, size_results, args.output_dir))
    fig_paths.append(fig_heuristic_count_vs_scale(checkpoint_results, size_results, args.output_dir))
    fig_paths.append(fig_heuristic_vs_accuracy(checkpoint_results, size_results, args.output_dir))
    fig_paths.append(fig_type_breakdown(checkpoint_results, size_results, args.output_dir))
    fig_paths.append(fig_knockout_across_scales(checkpoint_results, size_results, args.output_dir))

    report_path = write_markdown_report(
        results, checkpoint_results, size_results, fig_paths, args.output_dir,
    )
    print(f"Report written to {report_path}")
    print(f"Figures: {fig_paths}")


if __name__ == "__main__":
    main()
