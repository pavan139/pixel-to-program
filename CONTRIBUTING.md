# Working on scenes, training, or search

Changes to the scene language begin with the parser and renderer. Add a boundary image/program pair, check geometry and overlay order, then confirm that manifest validation still rejects hidden held-out primitives and duplicate rasters across splits. The generator's known program is a label, never a fallback prediction.

For neural changes, retain causal-prefix isolation, visual gradients, non-padding loss denominators, and exact CPU resume checks. A checkpoint must keep dataset/configuration identity, optimizer/schedule progress, and RNG state together. Test data must not choose best weights.

For search changes, use tiny exhaustive score tables. Width one must remain greedy, EOS must be scored, and a finished hypothesis cannot disappear merely because later prefixes expand. Grammar validity and image reconstruction are separate assertions.

Run `python -m unittest discover -s tests -v`; with `.[train]` installed, use `RUN_ML_SMOKE=1` for all neural checks. Run `python tools/check_vision.py --require-lint --wheel` before proposing the change. Paired ablations in `gallery/` should retain raw incorrect generations, parameter counts, regularization differences, seeds, and the same scene IDs.
