# Sharing scene experiments and checkpoints

Publish source, the scene format, dataset-generation settings, and raw evaluation before linking a trained checkpoint. Run `python tools/check_vision.py --require-lint --wheel` and inspect `reports/verification.json` for the current source/check record.

A checkpoint result needs the manifest hash, architecture and optimizer configuration, selection criterion, seeds, and generation/search settings. State whether evaluation loaded best-development weights or the latest resume state. Keep full optimizer snapshots outside Git and provide an integrity digest with any separately hosted artifact.

Images placed in `gallery/` need a label: ground-truth rendering, model prediction, or reconstruction comparison. The checked-in `scene.svg` is ground truth. Do not present grammar-valid wrong scenes as successful image reconstruction. The small ablation report deliberately retains zero exact-raster successes.

This directory can be pushed to an independent remote with its own workflow and package metadata. The standard-library renderer checks run without Torch; optional neural checks require the training extra. Preserve that split when preparing a release so installing the parser never downloads a model.
