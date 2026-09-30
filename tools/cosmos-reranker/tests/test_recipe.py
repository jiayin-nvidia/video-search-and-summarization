# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protect the checkpoint's loss, cache identity, and grouped batch contract."""

import math
import sys
import tomllib
from pathlib import Path

import pytest
import torch

from cosmos_reranker import cli
from cosmos_reranker.loss import BinaryResponseSpec, RankPointLoss, extract_binary_scores
from cosmos_reranker.visual_cache import visual_cache_key

SPEC = BinaryResponseSpec((1, 2, 3), (1, 4, 3), 1, 2, 4)


@pytest.mark.parametrize("positive_count", [1, 3, 19])
def test_equal_scores_preserve_positive_probability_mass(positive_count):
    labels = torch.tensor([1.0] * positive_count + [0.0] * (20 - positive_count))
    scores = torch.zeros(20, requires_grad=True)
    value = RankPointLoss(SPEC).from_scores(scores, labels)
    assert value.item() == pytest.approx(math.log(20 / positive_count) + math.log(2))
    value.backward()
    assert torch.all(scores.grad[:positive_count] < 0)
    assert torch.all(scores.grad[positive_count:] > 0)


def test_binary_projection_uses_state_before_answer_and_fp32():
    output = torch.zeros(2, 6, 8, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.tensor([[-100, -100, 1, 2, 3, -100], [-100, -100, 1, 4, 3, -100]])
    weight = torch.zeros(10, 8, dtype=torch.bfloat16)
    weight[2, 0] = 2
    weight[4, 0] = -1
    with torch.no_grad():
        output[:, 2, 0] = torch.tensor([1.0, -1.0])
    scores, binary = extract_binary_scores(output, labels, SPEC, projection_weight=weight)
    assert scores.dtype == torch.float32
    torch.testing.assert_close(scores, torch.tensor([3.0, -3.0]))
    torch.testing.assert_close(binary, torch.tensor([1.0, 0.0]))
    scores.sum().backward()
    assert output.grad[:, :2].count_nonzero() == 0
    assert output.grad[:, 3:].count_nonzero() == 0
    assert output.grad[:, 2].count_nonzero() == 2


@pytest.mark.parametrize("labels", [torch.ones(20), torch.zeros(20), torch.zeros(19)])
def test_invalid_query_groups_fail(labels):
    with pytest.raises(ValueError):
        RankPointLoss(SPEC).from_scores(torch.zeros_like(labels), labels)


def test_inference_readout_requires_both_requested_tokens():
    from cosmos_reranker.evaluation.rerankers.vlm.common import read_logits

    score, _ = read_logits("margin", {2: {"logprob": -0.25}, 4: {"logprob": -2.5}}, [2, 4], {}, "")
    assert score == 2.25
    score, metadata = read_logits("margin", {2: {"logprob": -0.25}}, [2, 4], {}, "")
    assert score is None
    assert metadata["error"] == "missing token logprob"


def test_cache_identity_resolves_symlink(tmp_path):
    image = tmp_path / "image.jpg"
    image.touch()
    alias = tmp_path / "alias.jpg"
    alias.symlink_to(image)
    assert visual_cache_key(alias) == visual_cache_key(image)
    assert visual_cache_key(image) != visual_cache_key(tmp_path / "other.jpg")


def test_rendered_32_gpu_recipe_retains_schedule_and_resumption(tmp_path, monkeypatch):
    package = Path(__file__).resolve().parents[1]
    argv = ["cosmos-reranker", "--root", str(package), "configure", "--gpus", "32"]
    for name in ("model", "pas-root", "data-root", "cache-root", "run-root"):
        argv.extend([f"--{name}", str(tmp_path / name)])
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    configs = tmp_path / "run-root/configs"
    first = tomllib.loads((configs / cli.STAGES["stage0"][0]).read_text())
    second = tomllib.loads((configs / cli.STAGES["stage1"][0]).read_text())
    for config in (first, second):
        assert config["train"]["max_num_steps"] == 200000
        assert config["train"]["optm_warmup_steps"] == 500
        assert (
            config["train"]["train_batch_per_replica"]
            * config["policy"]["parallelism"]["dp_replicate_size"]
            == 640
        )
        assert config["train"]["train_policy"]["mini_batch"] == 20
        assert len(config["custom"]["train_dataset"]["annotation_path"]) == 2
        assert len(config["custom"]["visual_cache"]["manifests"]) == 16
        assert not config["validation"]["enable"]
    assert first["train"]["resume"] is False
    assert second["train"]["resume"] == str(
        tmp_path / "run-root/stage0/checkpoints/step_2000/policy"
    )


def test_distributed_sampler_preserves_complete_groups():
    from cosmos_reranker.training import DistributedGroupBatchSampler

    assigned = []
    for rank in range(8):
        sampler = DistributedGroupBatchSampler(
            range(640),
            group_size=20,
            batch_size=80,
            num_replicas=8,
            rank=rank,
            shuffle=True,
            seed=160901,
        )
        batches = list(sampler)
        assert len(batches) == 1
        for start in range(0, 80, 20):
            group = batches[0][start : start + 20]
            assert group == list(range(group[0], group[0] + 20))
            assert group[0] % 20 == 0
        assigned.extend(batches[0])
    assert sorted(assigned) == list(range(640))
