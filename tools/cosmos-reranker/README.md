# Cosmos Nano reranker and fine-tuning recipe

This folder contains the Cosmos/CR3 Nano baseline reranker, the selected PAS
step-8000 language and visual LoRA adapters, and the recipe that produced them.
It is independent of the VSS application and can be run from this folder.

The fine-tuning implementation is here. Cosmos RL is a pinned training runtime
in the ignored `.runtime/` directory; installing it does not modify another
Cosmos RL checkout. The baseline reranker and evaluation do not require Cosmos RL.

## Contents

| Component | Purpose |
| --- | --- |
| `src/cosmos_reranker/cli.py` | Single command interface for preparation, training, export, and evaluation. |
| `training.py`, `loss.py`, `structured_attribute_aux.py` | The selected binary ranking and attribute-choice training objective. |
| Data preparation and visual cache modules | Rebuild the query selections, SigLIP2 K20 candidates, attribute targets, and frozen vision features. |
| `src/cosmos_reranker/evaluation/` | Cosmos Nano scoring, SigLIP2 checkpoint loading, PAS metrics, and shard merging. |
| `configs/` | Both stages' source settings, with portable path placeholders. Use `configure` to render runnable TOMLs. |
| `checkpoints/step8000/` | Supplied language and visual adapters, configuration, checksum verification, and model license. |
| `recipe/` | Exact cache image lists, expected artifact counts, mining manifests, and source checksums. |
| `runtime/cosmos_rl.patch` | Framework changes required for grouped batches, cached vision, joint LoRA, FP32 masters, and resume. |
| `results/` | Reference metrics and the completed independent reproduction evidence. |

The Qwen3 VL data packer is required by the Cosmos Nano architecture. There are
no Gemma, Qwen reranker experiments, alternate training objectives, teacher
models, or experimental attribute-probe heads in this package.

## Environment

Use Linux, Python 3.12, NVIDIA GPUs, and a CUDA toolkit compatible with PyTorch.
Keep training and vLLM evaluation in separate environments because they use
different Transformers versions.

The verified training environment was the
`nvcr.io/nvidia/tao/tao-toolkit:7.1.0-cosmos-rl` image with PyTorch
`2.11.0a0+nv26.3`, Transformers `4.57.5`, FlashAttention `2.7.4`, PEFT `0.17.1`,
Safetensors `0.8.0`, and W&B `0.28.1`. Start in that image or install matching
PyTorch and FlashAttention builds for your CUDA environment first.

```bash
cd /workspace/jiayin/video-search-and-summarization/tools/cosmos-reranker
python -m venv --system-site-packages .venv-train
.venv-train/bin/python -m pip install -e '.[train]'
.venv-train/bin/cosmos-reranker runtime
.venv-train/bin/python -m pip install -e .runtime/cosmos-rl

python3.12 -m venv .venv-eval
.venv-eval/bin/python -m pip install -e '.[eval]'
```

`runtime` fetches Cosmos RL commit
`89b1fdd89441964868a8767ee964e1e41a85ea68` and applies the included patch. For an
offline installation, pass `runtime --source /path/to/a/cosmos-rl/git-repository`
containing that commit. The runtime is created inside this folder.

The verified evaluation environment uses vLLM `0.23.0`, Transformers `5.14.1`,
and PyTorch `2.11.0+cu130`. The training runtime uses Redis, supplied by the
training image. Eight A100 80 GB GPUs run the original recipe. Training can use
16 or 32 GPUs while keeping the same global candidate batch of 640.

On the existing CUDA 13.2 host, use these runtime workarounds:

```bash
export CUDA_HOME=/usr/local/cuda-13.2
export LD_LIBRARY_PATH="$CUDA_HOME/compat/lib.real${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PAS_DISABLE_CUDNN_CONV=1
```

`PAS_DISABLE_CUDNN_CONV=1` selects PyTorch's native CUDA patch projection on
hosts where cuDNN cannot select the BF16 Conv3d engine. It is not needed on
every host. W&B defaults to offline mode in the training command.

## Required external inputs

The PAS data and base models are external inputs. The selected adapters are
included. Set these paths once; the values below locate the original assets
on this workspace:

```bash
export PAS_ROOT=/workspace/data/PAS_Datasets/NVIDIA_PAS_Filtered_06152026_V3.1_tao_ft_metadata_base_plus_val_test_medium_attr_matched_aug_val20_accessory_v2.1
export CR3_BASE=/workspace/jiayin/models/Cosmos3-Nano-VLM
export RETRIEVER=/workspace/jiayin/models/model_epoch_001_step_03906.pth
export SIGLIP_TOKENIZER=/workspace/jiayin/models/siglip2-so400m-patch16-256-tokenizer
export VAL999=/workspace/jiayin/rerank_experiment/artifacts/pas_v31_val_query_subset_scalar_plus_accessories_balanced_999/query_subset.json
export DATA_ROOT="$PWD/reproduction/data/step03906"
export EMBEDDING_ROOT="$PWD/reproduction/embeddings"
export CACHE_ROOT="$PWD/reproduction/cache"
export RUN_ROOT="$PWD/reproduction/train"
```

`PAS_ROOT` must contain `train_pairs.json`, `val_pairs.json`,
`attribute_vocab.json`, and `images/`. Keep the fixed Val999 subset: its captions
are excluded from training query mining. `CR3_BASE` is the clean HF VLM
export used to train the supplied adapter. Check input hashes against
`recipe/source_checksums.json`; another base or PAS revision defines a
different experiment.

## 1. Mine the data and create the cache

```bash
.venv-train/bin/cosmos-reranker prepare all \
  --pas-root "$PAS_ROOT" --model "$CR3_BASE" \
  --retriever "$RETRIEVER" --tokenizer "$SIGLIP_TOKENIZER" \
  --validation-subset "$VAL999" \
  --data-root "$DATA_ROOT" --embedding-root "$EMBEDDING_ROOT" \
  --cache-root "$CACHE_ROOT"
```

Preparation uses eight GPUs to preserve the recorded embedding and cache shard
layout. Each step can also be run separately by replacing `all` with `indices`,
`embeddings`, `mine`, `attributes`, or `cache`. Completed output files and their
manifests are reused. Use new output directories when changing inputs.

### Data mining

1. Read the 1,999,953 training pair rows. Exclude captions in the fixed Val999
   subset. Reservoir-sample 250,000 rows per difficulty with seed `880821`.
   Select the remaining eligible, disjoint query rows with seed `880822`.
   The expected selections contain 750,000 and 1,131,764 rows.
2. Use the step-03906 SigLIP2 retriever to encode 222,217 first-seen training
   images and the two text selections. Embeddings are normalized, 1,152
   dimensional, and stored in row-aligned NumPy arrays and eight text shards.
   Pass `--gallery-embeddings /path/image_embeddings.npy` to reuse a verified
   gallery array.
3. Search each query's dataset gallery by dot product and take the top 20.
   Generate yes/no targets from PAS scalar attributes plus accessories.
   Preserve candidate order, and retain groups containing both classes.
   The first selection uses the saved deployed-top-K record schema; the
   remaining selection uses the source miner's rank-spanning schema.
   The expected outputs are 669,871 and 1,017,346 K20 groups, respectively.
   Retriever scores determine candidates; they do not supervise the loss.
4. Extract structured image-attribute records from the training and validation
   pairs. Expected counts are 222,217 train and 8,138 validation records.
   The validation records are preserved as recipe artifacts; training
   validation is disabled for this checkpoint.

### Frozen visual cache

The exact primary and supplement image lists are in `recipe/cache_*_images.txt`:
217,743 primary images and 4,474 supplement images, covering all training
images without overlap. The cache builder preprocesses each image using patch
size 16, spatial merge size 2, and at most 81,920 pixels. It runs the clean
Cosmos vision prefix through block 23 and saves BF16 inputs to block 24,
earlier deep-stack features, token offsets, and `grid_thw` in safetensors
shards. Each GPU takes every eighth entry in the ordered image list.

The trainer reads these frozen tensors and runs blocks 24–26 and the merger
with trainable visual LoRA. The cache key is SHA-1 of the canonical absolute
image path. If image-root paths or symlink targets change, regenerate the
cache. Manifests record preprocessing settings, shard locations, and tensor
layout; the data packer checks those settings and token grids when loading.

| Output | Location under configured roots |
| --- | --- |
| Query selections | `DATA_ROOT/query_indices_trainonly_*.json` and manifests |
| K20 annotations | `DATA_ROOT/train_deployed_top20_k20_*.json` and manifests |
| Attribute targets | `DATA_ROOT/attribute_replay/{train,val}_records.json` |
| Retrieval arrays | `EMBEDDING_ROOT/trainonly_*/` |
| Vision tensors | `CACHE_ROOT/pas_cr3_block24_prefix_step03906_*/` |

Source hashes and per-shard expected counts are in `recipe/`. CUDA top-K ties
can change a small number of mined candidates across hardware. The completed
regeneration reproduced the remaining-selection annotations byte for byte;
the first selection had four candidate changes. All 6,952 cache tensors
(approximately 163.5 GB) matched the source tensors byte for byte.

## 2. Reproduce the fine-tuned checkpoint

```bash
.venv-train/bin/cosmos-reranker configure \
  --model "$CR3_BASE" --pas-root "$PAS_ROOT" \
  --data-root "$DATA_ROOT" --cache-root "$CACHE_ROOT" \
  --run-root "$RUN_ROOT" --gpus 8

.venv-train/bin/cosmos-reranker train stage0 --run-root "$RUN_ROOT"
.venv-train/bin/cosmos-reranker train stage1 --run-root "$RUN_ROOT"
.venv-train/bin/cosmos-reranker export --run-root "$RUN_ROOT"
```

Stage 0 saves the full step-2000 training checkpoint. Stage 1 resumes its LoRA,
optimizer, scheduler, and training state, then runs to step 8000. Both stages
use both annotation shards. The command waits for the target checkpoint and
its safetensors export, then requests the framework's graceful stop.

**Do not change `max_num_steps=200000` to 2000 or 8000.** That setting is part
of the source learning-rate schedule. The stage command controls when to stop.
Rendered configurations are in `RUN_ROOT/configs/`. Outputs are in
`RUN_ROOT/{stage0,stage1}/`; deployable adapters are in
`RUN_ROOT/deploy/{language,visual}/`.

### Exact objective and settings

For each candidate, the training score is the FP32 projection of the causal
hidden state through the frozen yes/no vocabulary rows:
`s = logit(yes) - logit(no)`. For each complete query group:

```text
rank_loss  = logsumexp(all K20 scores) - logsumexp(positive scores)
point_loss = (mean(BCE on positives) + mean(BCE on negatives)) / 2
total_loss = mean(rank_loss) + mean(point_loss) + 0.25 * attribute_choice_CE
```

The attribute suffix follows the binary answer, so it cannot change the causal
context used to predict yes/no. Each field's full choice vocabulary is shuffled
deterministically by `(seed, sample ID, field)`. Only isolated A–N answer-letter
positions receive the auxiliary loss. Missing labels, option text, markers,
and punctuation are excluded. There is no separate trainable attribute head.

| Setting | Selected checkpoint recipe |
| --- | --- |
| LoRA | Rank 32, alpha 64, dropout 0, standard scaling |
| Language targets | `q/k/v/o_proj`, `up/down/gate_proj`; no LM-head LoRA |
| Vision targets | Attention and MLP in blocks 24–26, plus the merger |
| Precision | FP32 trainable masters; BF16 frozen parameters and forward computation |
| Batch | 640 global candidates; K20 microbatches; 80 candidates per rank on 8 GPUs |
| Optimizer | Fused AdamW; LR groups `[2e-5, 5e-6, 5e-6, 0]`; betas `[0.9,0.999]`; epsilon `1e-8`; weight decay `0.01` |
| Schedule | 500-step warmup; cosine decay; minimum LR factor `0.1`; source 200,000-step cap |
| Other | Seed `160901`; gradient clip `1`; gradient checkpointing; no sequence packing; validation disabled |

### Use 16 or 32 GPUs

Render with `--gpus 16` or `--gpus 32` and a separate run root. Per-rank batches
become 40 or 20; K20 microbatches and the global batch stay unchanged. The
completed independent reproduction used 32 A100 GPUs across four workers.

For Lepton, place the checkout, environments, runtime, inputs, and output root
on the shared mount. Submit the rendered stage with the framework launcher:

```bash
PYTHONPATH="$PWD/.runtime/cosmos-rl:$PWD/src" \
  .venv-train/bin/python -m cosmos_rl.launcher.launch_all \
  --config "$RUN_ROOT/configs/train_stage0_to_step2000_replay.toml" \
  --num-workers 4 --lepton-mode --lepton-job-name cosmos-reranker-stage0 \
  --lepton-container-image nvcr.io/nvidia/tao/tao-toolkit:7.1.0-cosmos-rl \
  --lepton-resource-shape gpu.8xa100-80gb \
  --lepton-node-group "$LEPTON_NODE_GROUP" \
  --lepton-image-pull-secrets "$LEPTON_IMAGE_PULL_SECRET" \
  --lepton-mount "$LEPTON_MOUNT" \
  --lepton-shared-memory-size 65536 \
  --lepton-env "PYTHONPATH=$PWD/.runtime/cosmos-rl:$PWD/src" \
  --lepton-env WANDB_MODE=offline \
  --lepton-env PAS_DISABLE_CUDNN_CONV=1 \
  --lepton-env COSMOS_LOCAL_STATIC_RDZV=1 \
  --lepton-env COSMOS_SHUTDOWN_ON_NO_POLICY_REPLICAS=1 \
  --lepton-env CUDA_HOME=/usr/local/cuda-13.2 \
  --lepton-env LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat/lib.real \
  "$PWD/src/cosmos_reranker/training.py"
```

Stop the returned job ID once
`stage0/checkpoints/step_2000/policy/.rank_0_complete` exists and its step-2000
safetensors export is complete. Submit stage 1 with the second rendered TOML
and stop after the complete step-8000 export. Then run `export`. The full
optimizer checkpoint is required to resume; the deployable adapter alone
does not reproduce the continuation.

## 3. Load the supplied checkpoint or the new adapter

```bash
.venv-train/bin/cosmos-reranker checkpoint
export ADAPTER="$PWD/checkpoints/step8000"
# To evaluate your own reproduction instead:
# export ADAPTER="$RUN_ROOT/deploy"

.venv-train/bin/cosmos-reranker merge-vision \
  --model "$CR3_BASE" --adapter "$ADAPTER" \
  --output "$PWD/reproduction/merged-vision"
```

The supplied language adapter is split into two Git files. `checkpoint`
reassembles it and verifies both adapters:

| Adapter | SHA-256 |
| --- | --- |
| Language | `c6c994316eb1a727f349aa6ad0b6a84c52a88c9858fb952f95b8a9d6d341e750` |
| Visual | `a5e4724d5212d5a1a52df0f957d650e7988b7b23bfe4d3540fde1db402a0caaa` |

Visual LoRA is merged into the base in FP32 and saved in FP16, matching the
recorded evaluator. Language LoRA is loaded dynamically by vLLM. A new
training run should produce numerically close results; different GPU counts
and kernels do not guarantee identical adapter bytes.

### Minimal reranking example

```bash
# Baseline, without fine-tuning:
.venv-eval/bin/cosmos-reranker rank \
  --model nvidia/Cosmos3-Nano --query 'a person wearing a red shirt' \
  --images /path/candidate1.jpg /path/candidate2.jpg

# Selected checkpoint:
.venv-eval/bin/cosmos-reranker rank \
  --model "$PWD/reproduction/merged-vision" --adapter "$ADAPTER" \
  --query 'a person wearing a red shirt' \
  --images /path/candidate1.jpg /path/candidate2.jpg
```

The output is ranked JSON lines containing image paths and yes-minus-no
scores. Higher scores indicate stronger matches. The prompt ends at
`<answer>` and requests the yes/no token log probabilities. Their shared
normalization cancels in the score difference.

## 4. Reproduce the checkpoint's evaluation rows

The test protocol uses 69,633 deduplicated queries: 22,190 easy, 23,559 medium,
and 23,884 hard across six datasets. Deduplication keys are
`(dataset, query_type, caption)` and matching ground-truth sets are unioned.
Rerank the SigLIP2 top 20, preserve the remaining gallery order, and report
full-gallery AP, Rank-1, and Rank-5. `K=5` is the metric cutoff.

Use the recorded test export and step-03906 retrieval caches:

```bash
export TEST_ROOT=/workspace/jiayin/rerank_experiment/artifacts/pas_v31_test_tao
export TEST_CACHE=/workspace/jiayin/rerank_experiment/artifacts/pas_v31_step03906_cache
export VAL_CACHE=/workspace/jiayin/rerank_experiment/artifacts/pas_v31_val_step03906_tao

.venv-eval/bin/cosmos-reranker evaluate \
  --pairs "$TEST_ROOT/test_pairs.json" --images "$TEST_ROOT/images" \
  --image-embeddings "$TEST_CACHE/test_pairs_source_image_embeddings.pkl" \
  --text-embeddings "$TEST_CACHE/test_pairs_text_embeddings_lower.pkl" \
  --model "$PWD/reproduction/merged-vision" --adapter "$ADAPTER" \
  --gpus 0,1,2,3,4,5,6,7 --output reproduction/evaluation/test

.venv-eval/bin/cosmos-reranker evaluate \
  --pairs "$PAS_ROOT/val_pairs.json" --images "$PAS_ROOT/images" \
  --image-embeddings "$VAL_CACHE/val_pairs_source_image_embeddings.pkl" \
  --text-embeddings "$VAL_CACHE/val_pairs_text_embeddings_lower.pkl" \
  --validation-subset "$VAL999" \
  --model "$PWD/reproduction/merged-vision" --adapter "$ADAPTER" \
  --gpus 0,1,2,3,4,5,6,7 --output reproduction/evaluation/val999
```

To run zero-shot CR3 on the same retrieval candidates, omit `--adapter`, set
`--model nvidia/Cosmos3-Nano`, and use `--max-model-len 32768 --score-chunk 256`.
To measure the retrieval baseline, use `--reranker identity`. For public
SigLIP2 baseline rows, supply the public embedding cache instead. Evaluation
accepts any number of comma-separated GPU IDs, assigning one query shard per GPU.
Merged metrics and `query_rankings.jsonl` are under
`OUTPUT/scalar_plus_accessories/`; each shard has a `run.log`.

To rebuild the fine-tuned retrieval cache from pairs, use the included encoder
and converter in the training environment:

```bash
.venv-train/bin/python -m cosmos_reranker.evaluation.generate_embeddings \
  --pairs-file "$TEST_ROOT/test_pairs.json" --image-root "$TEST_ROOT/images" \
  --checkpoint "$RETRIEVER" --tokenizer-dir "$SIGLIP_TOKENIZER" \
  --output-dir reproduction/evaluation/test_arrays
.venv-train/bin/python -m cosmos_reranker.evaluation.convert_embeddings \
  --pairs-file "$TEST_ROOT/test_pairs.json" \
  --image-embeddings reproduction/evaluation/test_arrays/image_embeddings.npy \
  --text-embeddings reproduction/evaluation/test_arrays/text_embeddings.npy \
  --output-dir reproduction/evaluation/test_cache
```

Repeat for `val_pairs.json`, selecting the `val_pairs_*` output names supported
by the converter. The gallery must stay complete even when evaluating Val999.

### Validation-selected fusion

```bash
.venv-eval/bin/cosmos-reranker fusion \
  --validation-rankings reproduction/evaluation/val999/scalar_plus_accessories/query_rankings.jsonl \
  --test-rankings reproduction/evaluation/test/scalar_plus_accessories/query_rankings.jsonl \
  --output-dir reproduction/evaluation/fusion
```

Fit global score means and standard deviations and choose alpha on **Val999
overall mAP only**, then apply those frozen values to test:
`alpha * z(SigLIP2) + (1-alpha) * z(CR3)`. To replay the original table's
calibration, add
`--calibration-summary results/pas_cr3_step8000_fusion_replay.json`.
That frozen calibration uses alpha `0.31788`. It is a separate comparison
from refitting calibration for a newly trained checkpoint.

## Reproduction evidence and practical limits

The previous full reproduction rebuilt mining and caches, trained an independent
checkpoint to step 8000 on 32 GPUs, and evaluated the full test and Val999.

| Overall result | mAP | Rank-1 | Rank-5 |
| --- | --- | --- | --- |
| Reference selected reranker | 80.63% | 81.35% | 97.29% |
| Independent step-8000 reranker | 80.62% | 81.37% | 97.30% |
| Reference validation-fused model | 82.99% | 85.09% | 98.12% |
| Independent model, original frozen calibration | 82.96% | 85.01% | 98.10% |
| Independent model, refitted Val999 calibration | 82.65% | 84.41% | 98.02% |

The independent reranker differs by at most 0.07 percentage points across the
12 difficulty/overall metric cells. Original-calibration fusion differs by
at most 0.24 points. Refitting Val999 chose alpha `0.20035` and differed by
up to 1.01 points; that calibration difference remains. The supplied checkpoint
was also evaluated afresh, with selected reranking/fusion rows within 0.07 points.
Detailed metrics and checkpoint comparisons are in `results/`.

The standalone cleanup is checked against the original objective, BF16
projection, attribute loss, annotation messages, group ordering, and a short
distributed training/export run. See `results/standalone_validation.json`.

## Development and licenses

```bash
.venv-train/bin/python -m pip install -e '.[dev]'
.venv-train/bin/python -m pytest
.venv-train/bin/ruff check src tests
.venv-train/bin/ruff format --check src tests
```

Code is Apache-2.0; component origins are recorded in `NOTICE` and
`3rdParty_Licenses.md`. The Cosmos model and derived adapters use the model's
[OpenMDW-1.1 license](https://openmdw.ai/license/1-1/), identified in the
[NVIDIA model card](https://huggingface.co/nvidia/Cosmos3-Nano).
The base model and PAS data retain their own terms and are external inputs.
