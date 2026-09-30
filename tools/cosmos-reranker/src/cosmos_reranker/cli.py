# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the selected Cosmos Nano reranker from one command interface."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "89b1fdd89441964868a8767ee964e1e41a85ea68"
STAGES = {
    "stage0": ("train_stage0_to_step2000_replay.toml", 2000),
    "stage1": ("train_stage2000_to_step8000_replay.toml", 8000),
}


def run(*args, **kwargs):
    subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def environment(root, *, runtime=False):
    env = os.environ.copy()
    paths = [str(root / "src")]
    if runtime:
        dependency = root / ".runtime/cosmos-rl"
        if not (dependency / "cosmos_rl").is_dir():
            raise ValueError("Run `cosmos-reranker runtime` before training")
        paths.insert(0, str(dependency))
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        TRANSFORMERS_AUTO_SWITCH="0",
    )
    return env


def runtime(args):
    """Create an isolated, pinned framework checkout and apply the recipe patch."""
    destination = args.root / ".runtime/cosmos-rl"
    if destination.exists():
        raise ValueError(f"Runtime already exists; inspect it before recreating: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    run("git", "init", destination)
    run("git", "-C", destination, "remote", "add", "origin", args.source)
    run("git", "-C", destination, "fetch", "--depth=1", "origin", BASE_COMMIT)
    run("git", "-C", destination, "checkout", "--detach", "FETCH_HEAD")
    run("git", "-C", destination, "apply", "--check", args.root / "runtime/cosmos_rl.patch")
    run("git", "-C", destination, "apply", args.root / "runtime/cosmos_rl.patch")
    print(f"Prepared {destination} at {BASE_COMMIT} with the recipe runtime patch")
    print(
        f"Install framework dependencies in the training environment: {sys.executable} -m pip install -e {destination}"
    )


def configure(args):
    """Render both stages while preserving the source optimizer and LR horizon."""
    import toml

    if args.gpus not in (8, 16, 32):
        raise ValueError("Use 8, 16, or 32 data-parallel GPUs (global candidate batch remains 640)")
    batch = 640 // args.gpus
    for stage, (name, _) in STAGES.items():
        config = tomllib.loads((args.root / "configs" / name).read_text())
        output = args.run_root.resolve() / stage
        config["results_dir"] = str(output)
        config["redis"] = str(args.redis_port + (1 if stage == "stage0" else 0))
        config["train"]["output_dir"] = str(output)
        config["train"]["timestamp"] = f"cosmosreranker{stage}dp{args.gpus}"
        config["train"]["resume"] = (
            False
            if stage == "stage0"
            else str(args.run_root.resolve() / "stage0/checkpoints/step_2000/policy")
        )
        config["train"]["train_batch_per_replica"] = batch
        config["train"]["train_policy"]["dataloader_batch_size"] = batch
        config["policy"]["parallelism"]["dp_replicate_size"] = args.gpus
        config["policy"]["model_name_or_path"] = str(args.model.resolve())
        custom = config["custom"]
        custom["train_dataset"]["annotation_path"] = [
            str(args.data_root.resolve() / filename)
            for filename in (
                "train_deployed_top20_k20_trainonly750k_seed880821.json",
                "train_deployed_top20_k20_remaining_after750k_seed880822.json",
            )
        ]
        custom["train_dataset"]["media_path"] = str(args.pas_root.resolve() / "images")
        attributes = custom["structured_attribute"]
        attributes["train_records_path"] = str(
            args.data_root.resolve() / "attribute_replay/train_records.json"
        )
        attributes["validation_records_path"] = str(
            args.data_root.resolve() / "attribute_replay/val_records.json"
        )
        attributes["attribute_vocab_path"] = str(args.pas_root.resolve() / "attribute_vocab.json")
        cache = args.cache_root.resolve()
        custom["visual_cache"]["manifests"] = [
            str(cache / directory / f"manifest_rank{rank:02d}.json")
            for directory in (
                "pas_cr3_block24_prefix_step03906_trainonly750k_star",
                "pas_cr3_block24_prefix_step03906_top50_supplement4474_clean",
            )
            for rank in range(8)
        ]
        destination = args.run_root / "configs" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(toml.dumps(config))
        print(destination)


def prepare(args):
    env = environment(args.root)
    env.update(
        PAS_DATA_ROOT=str(args.pas_root.resolve()),
        RETRIEVER_CHECKPOINT=str(args.retriever.resolve()),
        SIGLIP_TOKENIZER=str(args.tokenizer.resolve()),
        VAL_CAPTION_DENYLIST=str(args.validation_subset.resolve()),
        CR3_BASE_MODEL=str(args.model.resolve()),
        PAS_DATA_OUTPUT=str(args.data_root.resolve()),
        PAS_EMBED_OUTPUT=str(args.embedding_root.resolve()),
        PAS_CACHE_OUTPUT=str(args.cache_root.resolve()),
        PYTHON_BIN=sys.executable,
        TORCHRUN_BIN=str(Path(sys.executable).parent / "torchrun"),
    )
    if args.gallery_embeddings:
        env["TRAIN_IMAGE_EMBEDDINGS"] = str(args.gallery_embeddings.resolve())
    run("bash", args.root / "src/cosmos_reranker/prepare.sh", args.stage, env=env)


def export_adapter(root, run_root):
    adapter = run_root / "stage1/safetensors/step_8000"
    run(
        sys.executable,
        "-m",
        "cosmos_reranker.wait_adapter_export",
        "--adapter-dir",
        adapter,
        env=environment(root),
    )
    destination = run_root / "deploy"
    if destination.exists():
        raise ValueError(f"Export destination exists; inspect it before exporting: {destination}")
    run(
        sys.executable,
        "-m",
        "cosmos_reranker.split_joint_lora_adapter",
        "--adapter",
        adapter,
        "--language-output",
        destination / "language",
        "--visual-output",
        destination / "visual",
        env=environment(root),
    )


def train(args):
    name, target = STAGES[args.stage]
    config = args.run_root / "configs" / name
    if not config.is_file():
        raise ValueError(f"Run configure first: {config}")
    settings = tomllib.loads(config.read_text())
    output = Path(settings["train"]["output_dir"])
    marker = output / f"checkpoints/step_{target}/policy/.rank_0_complete"
    if args.stage == "stage1":
        prior = Path(settings["train"]["resume"]) / ".rank_0_complete"
        if not prior.is_file():
            raise ValueError(f"Stage 0 must finish first: {prior}")
    if marker.is_file():
        print(f"Stage already completed: {marker}")
        return
    env = environment(args.root, runtime=True)
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("COSMOS_MASTER_PORT", str(int(settings["redis"]) + 20000))
    env.update(COSMOS_LOCAL_STATIC_RDZV="1", COSMOS_SHUTDOWN_ON_NO_POLICY_REPLICAS="1")
    command = [sys.executable, "-m", "cosmos_rl.launcher.launch_all", "--config", str(config)]
    if args.launch_args:
        command.extend(shlex.split(args.launch_args))
    command.append(str(args.root / "src/cosmos_reranker/training.py"))
    output.mkdir(parents=True, exist_ok=True)
    launcher = subprocess.Popen(command, env=env)
    print(f"Started PID {launcher.pid}; target checkpoint: {marker}", flush=True)
    try:
        while launcher.poll() is None:
            if marker.is_file():
                # Do not stop while asynchronous safetensors export is incomplete.
                run(
                    sys.executable,
                    "-m",
                    "cosmos_reranker.wait_adapter_export",
                    "--adapter-dir",
                    output / f"safetensors/step_{target}",
                    env=environment(args.root),
                )
                launcher.send_signal(signal.SIGUSR1)
                break
            time.sleep(5)
        status = launcher.wait()
    except BaseException:
        if launcher.poll() is None:
            launcher.send_signal(signal.SIGUSR1)
            launcher.wait()
        raise
    if not marker.is_file():
        raise RuntimeError(
            f"Training exited {status} without the complete target checkpoint: {marker}"
        )
    print(f"Completed {args.stage} at step {target}")


def materialize(args):
    run(sys.executable, args.root / "checkpoints/step8000/materialize.py")


def merge(args):
    run(
        sys.executable,
        "-m",
        "cosmos_reranker.merge_adapter",
        "--base-model",
        args.model,
        "--adapter",
        args.adapter / "visual",
        "--output",
        args.output,
        "--merge-dtype",
        "float32",
        "--save-dtype",
        "float16",
        env=environment(args.root),
    )


def evaluate(args):
    """Launch one evaluator per GPU and merge only a complete set of shards."""
    gpu_ids = args.gpus.split(",")
    if (
        not gpu_ids
        or any(not item.strip() for item in gpu_ids)
        or len(set(gpu_ids)) != len(gpu_ids)
    ):
        raise ValueError("--gpus must list distinct CUDA device IDs")
    if args.adapter and not args.model:
        raise ValueError(
            "Use --model with the visually merged base when evaluating a language adapter"
        )
    env = environment(args.root)
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    args.output.mkdir(parents=True, exist_ok=True)
    shards = []
    processes = []
    handles = []
    try:
        for rank, gpu in enumerate(gpu_ids):
            shard = args.output / f"shard_{rank}_of_{len(gpu_ids)}"
            if shard.exists():
                raise ValueError(f"Shard output exists; select a new output directory: {shard}")
            shard.mkdir()
            shards.append(shard)
            command = [
                sys.executable,
                "-m",
                "cosmos_reranker.evaluation.evaluate",
                "--pairs-file",
                str(args.pairs),
                "--image-root",
                str(args.images),
                "--image-embeddings",
                str(args.image_embeddings),
                "--text-embeddings",
                str(args.text_embeddings),
                "--output-dir",
                str(shard),
                "--run-name",
                f"cosmos_shard_{rank}",
                "--modes",
                "scalar_plus_accessories",
                "--query-types",
                "easy",
                "medium",
                "hard",
                "--k",
                "5",
                "--rerank-depth",
                "20",
                "--reranker",
                args.reranker,
                "--reranker-batch-size",
                "16",
                "--query-batch-size",
                "64",
                "--reranker-score-chunk-size",
                str(args.score_chunk),
                "--save-query-rankings",
                "--shard-count",
                str(len(gpu_ids)),
                "--shard-index",
                str(rank),
                "--gpu-memory-utilization",
                str(args.gpu_memory),
                "--max-model-len",
                str(args.max_model_len),
            ]
            if args.validation_subset:
                command.extend(["--query-subset-file", str(args.validation_subset)])
            else:
                command.append("--deduplicate-scalar-plus-accessories")
            if args.limit_pairs:
                command.extend(["--limit-pairs", str(args.limit_pairs)])
            if args.model:
                command.extend(["--model-id", str(args.model)])
            if args.adapter:
                command.extend(
                    ["--lora-path", str(args.adapter / "language"), "--max-lora-rank", "32"]
                )
            shard_env = dict(env, CUDA_VISIBLE_DEVICES=gpu.strip())
            handle = (shard / "run.log").open("w")
            handles.append(handle)
            processes.append(
                subprocess.Popen(command, env=shard_env, stdout=handle, stderr=subprocess.STDOUT)
            )
            print(f"Shard {rank} on GPU {gpu}: {shard / 'run.log'}", flush=True)
        statuses = [process.wait() for process in processes]
        if any(statuses):
            raise RuntimeError(
                f"Evaluation failed (shard exit codes {statuses}); inspect shard run.log files"
            )
        run(
            sys.executable,
            "-m",
            "cosmos_reranker.evaluation.merge_results",
            "--output-dir",
            args.output,
            "--run-name",
            "cosmos",
            "--k",
            "5",
            *shards,
            env=env,
        )
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        for handle in handles:
            handle.close()


def rank(args):
    from cosmos_reranker.evaluation.config import RerankerConfig
    from cosmos_reranker.evaluation.data import Segment
    from cosmos_reranker.evaluation.rerankers import Candidate, build_reranker

    options = {
        "model_id": args.model,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory,
    }
    if args.adapter:
        options.update(lora_path=str(args.adapter / "language"), max_lora_rank=32)
    scorer = build_reranker(
        RerankerConfig(name="cosmos_reason", rerank_depth=len(args.images), options=options)
    )
    candidates = [
        Candidate(Segment(index, str(path), None, None), -float(index), image_path=path)
        for index, path in enumerate(args.images)
    ]
    for candidate in scorer.rerank(args.query, candidates):
        if candidate.rerank_score is None:
            raise RuntimeError(f"CR3 did not return yes/no scores: {candidate.rerank_meta}")
        print(json.dumps({"image": str(candidate.image_path), "score": candidate.rerank_score}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Standalone package checkout")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("runtime", help="Prepare the isolated, pinned Cosmos RL dependency")
    p.add_argument("--source", default="https://github.com/jiayin-nvidia/cosmos-rl.git")
    p.set_defaults(func=runtime)
    p = commands.add_parser("configure", help="Render the exact two-stage fine-tuning recipe")
    for name in ("model", "pas-root", "data-root", "cache-root", "run-root"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--gpus", type=int, choices=(8, 16, 32), default=8)
    p.add_argument("--redis-port", type=int, default=13220)
    p.set_defaults(func=configure)
    p = commands.add_parser("prepare", help="Mine K20 groups and build the frozen vision cache")
    p.add_argument("stage", choices=("indices", "embeddings", "mine", "attributes", "cache", "all"))
    for name in (
        "model",
        "pas-root",
        "retriever",
        "tokenizer",
        "validation-subset",
        "data-root",
        "embedding-root",
        "cache-root",
    ):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--gallery-embeddings", type=Path)
    p.set_defaults(func=prepare)
    p = commands.add_parser("train", help="Train to step 2000 or resume to step 8000")
    p.add_argument("stage", choices=tuple(STAGES))
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument(
        "--launch-args", help="Additional Cosmos launcher flags as a shell-quoted string"
    )
    p.set_defaults(func=train)
    p = commands.add_parser(
        "export", help="Split a newly trained joint LoRA into deployable adapters"
    )
    p.add_argument("--run-root", type=Path, required=True)
    p.set_defaults(func=lambda a: export_adapter(a.root, a.run_root))
    p = commands.add_parser(
        "checkpoint", help="Reassemble and verify the supplied step-8000 adapters"
    )
    p.set_defaults(func=materialize)
    p = commands.add_parser("merge-vision", help="Merge the visual adapter into an inference base")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=merge)
    p = commands.add_parser("rank", help="Rank candidate images by Cosmos yes-minus-no score")
    p.add_argument("--query", required=True)
    p.add_argument("--images", nargs="+", type=Path, required=True)
    p.add_argument("--model", default="nvidia/Cosmos3-Nano")
    p.add_argument("--adapter", type=Path)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-memory", type=float, default=0.85)
    p.set_defaults(func=rank)
    p = commands.add_parser("evaluate", help="Run the deduplicated test or fixed Val999 evaluation")
    for name in ("pairs", "images", "image-embeddings", "text-embeddings", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--validation-subset", type=Path)
    p.add_argument("--reranker", choices=("cosmos_reason", "identity"), default="cosmos_reason")
    p.add_argument("--model")
    p.add_argument("--adapter", type=Path)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--score-chunk", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-memory", type=float, default=0.85)
    p.add_argument(
        "--limit-pairs", type=int, default=0, help="Debug only; never use for table results"
    )
    p.set_defaults(func=evaluate)
    p = commands.add_parser("fusion", help="Fit fusion on validation and apply it to test")
    for name in ("validation-rankings", "test-rankings", "output-dir"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--calibration-summary", type=Path)

    def fusion(args):
        arguments = [
            "--validation-rankings",
            args.validation_rankings,
            "--test-rankings",
            args.test_rankings,
            "--output-dir",
            args.output_dir,
        ]
        if args.calibration_summary:
            arguments.extend(["--calibration-summary", args.calibration_summary])
        run(
            sys.executable,
            "-m",
            "cosmos_reranker.fit_val_apply_test_fusion",
            *arguments,
            env=environment(args.root),
        )

    p.set_defaults(func=fusion)
    args = parser.parse_args()
    args.root = args.root.resolve()
    try:
        args.func(args)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"cosmos-reranker: {error}\n")


if __name__ == "__main__":
    main()
