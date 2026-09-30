# Component origins and licenses

| Files | Origin | License |
| --- | --- | --- |
| `training.py`, `loss.py`, `structured_attribute_aux.py`, data preparation, cache, adapter and fusion modules | PAS reproduction, `jiayin-nvidia/cosmos-rl@4a5105c3cd6815dc3333df2de75771f050c5fc8d` | Apache-2.0 |
| `src/cosmos_reranker/evaluation/` | NVIDIA `rerank_experiment` sources captured in the same immutable reproduction commit | Apache-2.0 |
| `runtime/cosmos_rl.patch` | Changes to NVIDIA Cosmos RL base `89b1fdd89441964868a8767ee964e1e41a85ea68` | Apache-2.0 |
| `checkpoints/step8000/` model adapters | Cosmos3-Nano VLM, fine-tuned on PAS using the included recipe | OpenMDW-1.1; see `LICENSE.model` |

Python dependencies and the separately fetched Cosmos RL runtime retain their
upstream licenses. Base model weights, retrieval checkpoints, and PAS data are
external inputs. This component does not grant rights to those external inputs.
