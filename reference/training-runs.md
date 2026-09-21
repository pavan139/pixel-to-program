# Scene and training experiment notes

Generate the split before training. Record the seed and manifest hash. Training omits green rectangles and blue ellipses; development introduces green rectangles; test introduces blue ellipses. The designated held-out primitive must remain visible, and raster/program duplicates cannot cross splits.

The default `evaluate` is a codec/renderer check using known programs. It does not estimate image-to-program accuracy. Learned evaluation must feed image tensors through an actual saved model, retain invalid generations, and score their failure explicitly.

For paired architecture runs:

```bash
python -m pixel_to_program dataset --directory artifacts/attention-data --train 16 --dev 4 --test 4 --seed 1729
python -m pixel_to_program ablation --data-dir artifacts/attention-data --run-dir artifacts/attention-runs --epochs 2 --d-model 16 --seeds 17 23 --beam-sizes 1 4 --output artifacts/attention-ablation.json
```

The stored `gallery/attention-ablation.json` is that short experiment. CNN/GRU and patch-attention models use the shared data/optimization/search engine, but their parameter counts and regularization are different. It is not a parameter-matched causal comparison. Report token-weighted development NLL, raw and constrained generation validity, exact raster match, foreground IoU, RGB error, and primitive/held-out recall.

Four test scenes evaluated under two training seeds produce eight scene-seed observations, not eight unique scenes. Both architectures had grammar-valid constrained outputs without exact reconstruction success in the recorded run. Longer training or a different decoder setting needs a separately identified report.

For resume tests, compare an uninterrupted run with one stopped and resumed under the same dataset, configuration, and CPU environment. Include dropout so RNG restoration matters. Do not extrapolate that deterministic CPU check to arbitrary GPU kernels.
