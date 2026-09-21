# Spatial attention training results

The spatial-attention system and CNN/GRU comparator share the renderer, training engine, and search interfaces.

## Implemented components

- Spatial patch Transformer encoder with learned two-dimensional positions.
- Causal program decoder with visual cross attention and tied token/output embeddings.
- Shared AdamW training, warmup/cosine scheduling, label smoothing, token-weighted validation, and gradient clipping.
- Atomic, dataset/configuration-bound latest/best snapshots with optimizer and RNG restoration.
- Batched likelihood beam search, completion archival, visual-memory reuse, and geometry-aware token filtering.
- Paired architecture/search experiments with raw generations and primitive/raster metrics.

The architecture, equations, state transitions, complexity, and limitations are documented in [attention-and-search.md](attention-and-search.md).

## Reproduce the bounded CPU experiment

Install the optional training dependencies, then run:

```bash
python -m pixel_to_program dataset --directory artifacts/attention-data --train 16 --dev 4 --test 4 --seed 1729
python -m pixel_to_program ablation --data-dir artifacts/attention-data --run-dir artifacts/attention-runs --epochs 2 --d-model 16 --seeds 17 23 --beam-sizes 1 4 --output gallery/attention-ablation.json
```

The example is a tiny execution/ablation run, not a useful-accuracy claim. Checkpoint artifacts remain outside Git. The committed [ablation report](../gallery/attention-ablation.json) contains actual paired outputs and dataset identity.

## Resume an attention training run

```bash
python -m pixel_to_program train-system --data-dir artifacts/attention-data --checkpoint artifacts/run.pt --epochs 2 --d-model 16 --heads 2 --encoder-layers 1 --decoder-layers 1 --feedforward 32 --patch-size 8 --stop-after-epochs 1
python -m pixel_to_program train-system --data-dir artifacts/attention-data --checkpoint artifacts/run.pt --epochs 2 --d-model 16 --heads 2 --encoder-layers 1 --decoder-layers 1 --feedforward 32 --patch-size 8 --resume
python -m pixel_to_program evaluate-system --data-dir artifacts/attention-data --checkpoint artifacts/run.pt --beam-size 4
```

`run.pt` contains both latest-resume and best-development inference weights. Evaluation defaults to embedded best selection. No separately updated best file is required.

## Executed CPU evidence

- Full local suite: **54 tests passed** with the optional CPU Torch dependency installed.
- Tests check causal masking, visual gradients, loss normalization, exact dropout/optimizer/RNG resume, embedded best/latest state, and exhaustive beam likelihood cases.
- Independent cross-review reproduced completion eviction in beam search; completion archival and a width-one-greedy regression now cover that case.
- Paired experiment: 16 training scenes, 4 development scenes, and 4 test scenes; 2 training seeds; 2 epochs/8 updates per architecture and seed; 16 architecture/search/constraint variants.

| Architecture | Parameters | Dev NLL seed17 /23 | Greedy raw validity | Greedy constrained validity | Constrained exact-raster matches |
| --- | ---: | --- | ---: | ---: | ---: |
| cnn_gru | 43,035 | 3.7327 / 3.7366 | 0/8 | 8/8 | 0/8 |
| patch_transformer | 7,888 | 3.9421 / 4.1315 | 0/8 | 8/8 | 0/8 |

The denominator of eight above counts four shared test scenes evaluated after two independent training seeds. These short runs show grammar validity without useful image reconstruction. The models have different parameter counts and regularization, and the Transformer has worse development NLL in this tiny run.

Dataset SHA-256: `8265cf8e7b5d02047c35d9839e25e5cb98b572ee53d82f2b000b780a6ab42cdd`.

## Evidence and remaining work

Behavioral tests cover future-token isolation, visual gradients, padding invariance, exact CPU resume with dropout, frozen best-weight snapshots, exhaustive small beam cases, grammar completion, and a previously reproduced lost-completion bug. The standalone verifier additionally installs a wheel outside the checkout and executes the default commands and `search-demo` without Torch.

Longer training, natural-image transfer, useful compositional accuracy, parameter-matched architecture comparisons, distributed training, and GPU performance remain unverified. The measured short runs expose errors rather than treating grammatical validity as image reconstruction success.
