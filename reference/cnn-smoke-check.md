# Original CNN pipeline smoke check

The following checks were executed during implementation on Python 3.12 with CPU-only PyTorch 2.6.0:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m unittest discover -s tests -v
```

Result after the technical review: **37 tests passed**, including `test_train_save_load_predict_and_evaluate`. That integration test generates four training, two development, and two test scenes; trains one epoch with hidden width 16; saves and reloads the selected checkpoint; performs prediction; and evaluates both test scenes. The checkpoint and generated data are temporary and are removed by the test. Two further optional tests execute controlled neural grammar masking and malformed-checkpoint checks.

This check establishes that training, checkpoint loading, prediction, and evaluation execute together. The tiny dataset and single epoch do not establish useful reconstruction accuracy or compositional generalization. Those questions need a separately reported held-out experiment.

Additional executed checks:

- Core `demo` generated a valid scene program, SVG, PPM, and JSON report.
- Core `evaluate` checked 36 oracle codec/renderer cases; that report explicitly marks learned-model evaluation false.
- Ruff `--select E9,F` and Python compilation passed.
- Split regression tests confirm that designated heldout primitives remain visible, rendered images are globally unique, and manifest images match their programs.

The suite can run without PyTorch; in that case the three optional integration tests are explicitly skipped rather than reported as executed. Other tests remain independent of ML packages. See [the technical review](rendering-defects.md) for adversarial and targeted mutation evidence.
