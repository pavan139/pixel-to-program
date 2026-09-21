# Technical review

## Reproduced defects and fixes

- Manifest rows that were arrays, omitted fields, or had wrong types could produce incidental attribute/key/type errors. Rows now have an exact typed contract, line budget, duplicate-key rejection, and explicit validation errors.
- Absolute paths inside the data root were accepted, and output-directory symlinks could direct generated files outside the selected directory. Image paths are now relative, exclude parent traversal and NULs, and are checked against escaping symlinks on reads and writes. Manifest symlink escapes and missing/non-file images are rejected.
- Manifest loading checked heldout pair membership without checking whether that primitive was visible. A blue ellipse covered by a red rectangle could pass with a perfectly consistent program/image pair. Loaded heldout scenes now require exactly one reserved primitive that changes at least half its intrinsic foreground pixels. Generation uses the same visibility contract and bounded deduplication attempts.
- Decoder postprocessing silently removed internal BOS/padding and accepted streams without EOS. Those streams now remain explicit failures, preserving raw IDs, EOS status, and validation errors. No reference program replaces failed model output.
- Development selection averaged per-batch losses equally, biasing results when batch lengths differed. Training/development reports now use summed non-padding token loss divided by the actual target-token denominator. Checkpoint architecture types, weight compatibility, and file size are validated.

## Added technical capability

`--constrained` masks actual CNN/GRU decoder logits with a token grammar. It tracks scene boundaries, fields, shape count, coordinate-dependent width/height limits, and EOS placement. It receives no target labels. A short token budget still yields an incomplete failure; valid syntax does not establish a correct reconstruction.

Evaluation now includes multiset exact-primitive extraction and exact recall of the reserved heldout primitive. Matching includes kind, color, x/y, width, and height; duplicate predictions do not gain extra credit. These metrics distinguish wrong color/geometry from superficially similar foreground masks. Raster scores remain separate because drawing order and occlusion affect image equivalence.

## Executed verification

Using Python 3.12 and CPU PyTorch 2.6:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m pixel_to_program demo --output artifacts/review-demo.json
python -m pixel_to_program evaluate --output artifacts/review-evaluation.json
python tools/probe_rendering_contract.py
```

The complete suite passes **37 tests**. Three optional integration tests execute real neural code: a four-example/one-epoch training-save-load-evaluation check; controlled decoder logits proving actual masking, bounded completion, and raw failure handling; and malformed checkpoint validation. A regression assertion disables teacher-forced `forward` during evaluation to verify that heldout predictions come from autoregressive generation. Another independently recomputes development token loss from the selected checkpoint.

Adversarial tests cover malformed/duplicate manifest keys, wrong field types, missing/absolute/parent/NUL paths, read/write symlink escapes, occluded reserved primitives, finite deduplication attempts, grammar field/geometry budgets, EOS rules, and primitive matching. All tests are offline. Optional torch tests skip explicitly without the dependency.

Three targeted mutations were killed in temporary copies: removing heldout visibility enforcement, disabling neural grammar masking, and using unshifted teacher-forcing labels. [mutation-results.json](mutation-results.json) retains failure evidence. This is a focused spot check, not an exhaustive mutation score.

Ruff formatting and configured E9/F checks pass. The core demo and 36-case reference CLI evaluation were executed. Those reference results remain codec/renderer checks with `learned_model_evaluated: false`.

## Remaining limits

No useful learned accuracy, broad visual generalization, or inference-speed claim follows from a one-epoch smoke test or controlled-logit fixture. No full training or GPU evaluation was run. The dataset is synthetic, 32×32, and limited to two primitives and three colors. Some non-reserved objects can still be occluded, and different programs can produce equivalent rasters; exact extraction and raster metrics must therefore be interpreted together. External checkpoints are loaded with `weights_only=True` but still consume local resources. Large/adversarial filesystem races and production-scale datasets are outside this review.
