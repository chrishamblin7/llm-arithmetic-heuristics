"""
Parallel Pythia Heuristics Sweep — 8-GPU Orchestrator
=====================================================
Distributes model analysis across multiple GPUs using a work-stealing queue.
Each GPU worker runs script_pythia_sweep.py on one model at a time.

Usage:
    python script_parallel_sweep.py --num-gpus 8 --experiment both --output-dir ./results
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from filelock import FileLock

from script_pythia_sweep import build_model_list, estimate_params, aggregate_results


def predownload_models(model_list: list):
    """Download all unique model weights sequentially before workers start.

    Uses snapshot_download to cache weights without loading into memory,
    avoiding the disk I/O storm when 8 workers start simultaneously.
    """
    from huggingface_hub import snapshot_download
    seen_revisions = set()
    for model_name in model_list:
        name, step = model_name.split("-step")
        hf_name = f"EleutherAI/{name}"
        revision = f"step{step}"
        cache_key = (hf_name, revision)
        if cache_key in seen_revisions:
            continue
        seen_revisions.add(cache_key)
        print(f"[Pre-download] {hf_name} @ {revision}...")
        try:
            snapshot_download(hf_name, revision=revision)
        except Exception as e:
            print(f"[Pre-download] Warning: {hf_name} @ {revision} failed: {e}")
    print(f"[Pre-download] Done. {len(seen_revisions)} unique checkpoints cached.")


def build_sorted_queue(experiment: str) -> list:
    """Build model list sorted largest-first (LPT scheduling)."""
    models = build_model_list(experiment)
    return sorted(models, key=lambda m: estimate_params(m), reverse=True)


def claim_next_model(queue_path: str, lock_path: str, worker_id: int) -> str | None:
    """Atomically claim the next unclaimed model from the work queue."""
    lock = FileLock(lock_path, timeout=30)
    with lock:
        with open(queue_path) as f:
            queue = json.load(f)

        for entry in queue:
            if entry["status"] == "pending":
                entry["status"] = "running"
                entry["worker"] = worker_id
                entry["started_at"] = datetime.now().isoformat()
                with open(queue_path, "w") as f:
                    json.dump(queue, f, indent=2)
                return entry["model"]

    return None


def mark_complete(queue_path: str, lock_path: str, model_name: str, success: bool):
    """Mark a model as completed or failed in the queue."""
    lock = FileLock(lock_path, timeout=30)
    with lock:
        with open(queue_path) as f:
            queue = json.load(f)

        for entry in queue:
            if entry["model"] == model_name:
                entry["status"] = "done" if success else "failed"
                entry["finished_at"] = datetime.now().isoformat()
                break

        with open(queue_path, "w") as f:
            json.dump(queue, f, indent=2)


def worker_loop(worker_id: int, gpu_id: int, queue_path: str, lock_path: str,
                output_dir: str, experiment: str, model_path: str):
    """Worker process: claim models and run analysis until queue is empty."""
    log_path = os.path.join(output_dir, f"worker_{worker_id}_gpu{gpu_id}.log")

    while True:
        model_name = claim_next_model(queue_path, lock_path, worker_id)
        if model_name is None:
            with open(log_path, "a") as lf:
                lf.write(f"[{datetime.now().isoformat()}] Worker {worker_id}: Queue empty, exiting\n")
            break

        with open(log_path, "a") as lf:
            lf.write(f"[{datetime.now().isoformat()}] Worker {worker_id}: Starting {model_name} on GPU {gpu_id}\n")

        cmd = [
            sys.executable, "script_pythia_sweep.py",
            "--experiment", experiment,
            "--output-dir", output_dir,
            "--gpu-id", str(gpu_id),
            "--models", model_name,
        ]
        if model_path:
            cmd.extend(["--model-path", model_path])

        try:
            result = subprocess.run(
                cmd,
                stdout=open(os.path.join(output_dir, f"model_{model_name}.stdout.log"), "w"),
                stderr=subprocess.STDOUT,
                timeout=7200,  # 2 hour timeout per model
            )
            success = result.returncode == 0
        except subprocess.TimeoutExpired:
            success = False
        except Exception:
            success = False

        mark_complete(queue_path, lock_path, model_name, success)

        status = "DONE" if success else "FAILED"
        with open(log_path, "a") as lf:
            lf.write(f"[{datetime.now().isoformat()}] Worker {worker_id}: {model_name} -> {status}\n")


def print_progress(queue_path: str, start_time: float):
    """Print a progress summary table."""
    with open(queue_path) as f:
        queue = json.load(f)

    total = len(queue)
    done = sum(1 for e in queue if e["status"] == "done")
    failed = sum(1 for e in queue if e["status"] == "failed")
    running = sum(1 for e in queue if e["status"] == "running")
    pending = sum(1 for e in queue if e["status"] == "pending")
    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"  Progress: {done}/{total} done, {running} running, "
          f"{pending} pending, {failed} failed")
    print(f"  Elapsed: {timedelta(seconds=int(elapsed))}")
    if done > 0:
        est_remaining = (elapsed / done) * (total - done - failed) / max(running, 1)
        print(f"  Est. remaining: ~{timedelta(seconds=int(est_remaining))}")
    print(f"{'='*60}")

    for entry in queue:
        status_sym = {"pending": "  ", "running": ">>", "done": "OK", "failed": "XX"}
        gpu_info = f"GPU {entry.get('worker', '?')}" if entry["status"] != "pending" else ""
        print(f"  [{status_sym.get(entry['status'], '??')}] {entry['model']:40s} {gpu_info}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Parallel Pythia Heuristics Sweep")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--experiment", choices=["checkpoint", "size", "both"], default="both")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--poll-interval", type=int, default=30, help="Progress poll interval in seconds")
    parser.add_argument("--skip-predownload", action="store_true", help="Skip model pre-download (use if models already cached)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_list = build_sorted_queue(args.experiment)
    queue_path = os.path.join(args.output_dir, "work_queue.json")
    lock_path = queue_path + ".lock"

    # Initialize queue (preserving any previously completed entries for crash recovery)
    if os.path.exists(queue_path):
        with open(queue_path) as f:
            existing = {e["model"]: e for e in json.load(f)}
    else:
        existing = {}

    queue = []
    for model in model_list:
        if model in existing and existing[model]["status"] == "done":
            queue.append(existing[model])
        else:
            queue.append({"model": model, "status": "pending", "worker": None,
                          "started_at": None, "finished_at": None})

    with open(queue_path, "w") as f:
        json.dump(queue, f, indent=2)

    already_done = sum(1 for e in queue if e["status"] == "done")
    remaining = len(queue) - already_done
    print(f"[Launcher] {len(queue)} models total, {already_done} already done, "
          f"{remaining} to process across {args.num_gpus} GPUs")

    if remaining == 0:
        print("[Launcher] All models already complete, skipping to aggregation")
    else:
        if not args.skip_predownload:
            models_to_download = [e["model"] for e in queue if e["status"] == "pending"]
            predownload_models(models_to_download)
        else:
            print("[Launcher] Skipping pre-download (--skip-predownload)")
        # Spawn worker processes with staggered starts
        import multiprocessing
        workers = []
        stagger_delay = 10  # seconds between starts (models already cached)
        for i in range(min(args.num_gpus, remaining)):
            p = multiprocessing.Process(
                target=worker_loop,
                args=(i, i, queue_path, lock_path, args.output_dir,
                      args.experiment, args.model_path),
                daemon=False,
            )
            p.start()
            workers.append(p)
            print(f"[Launcher] Started worker {i} -> GPU {i} (pid {p.pid})")
            if i < min(args.num_gpus, remaining) - 1:
                time.sleep(stagger_delay)

        # Monitor loop
        start_time = time.time()
        while any(w.is_alive() for w in workers):
            time.sleep(args.poll_interval)
            print_progress(queue_path, start_time)

            # Sync intermediate results to cloud (best-effort)
            subprocess.run(
                ["rsync", "-a", f"{args.output_dir}/",
                 "/cloud/misc/chris_heuristics_sweep/results/"],
                capture_output=True, timeout=60,
            )

        # Final status
        for w in workers:
            w.join()
        print_progress(queue_path, start_time)

    # Aggregate results
    print("[Launcher] Aggregating results...")
    all_models = build_model_list(args.experiment)
    agg_path = aggregate_results(args.output_dir, all_models)
    print(f"[Launcher] Aggregated results: {agg_path}")

    # Generate report
    print("[Launcher] Generating report...")
    subprocess.run([
        sys.executable, "script_generate_report.py",
        "--results-dir", args.output_dir,
        "--output-dir", os.path.join(os.path.dirname(args.output_dir), "report"),
    ])

    print("[Launcher] All done.")


if __name__ == "__main__":
    main()
