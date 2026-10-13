# Vision package check history

[reports/verification.json](../reports/verification.json) contains the current source digest, test counts, optional skips, CLI results, and wheel-install checks. Refresh it with `python tools/check_vision.py --require-lint --wheel`.

The package has three different evidence types: dependency-free renderer/manifest checks, small actual neural integration tests, and the short paired training report in `gallery/attention-ablation.json`. None substitutes for the others. The [early CNN smoke note](cnn-smoke-check.md) preserves its original execution scope, and [spatial-attention-results.md](spatial-attention-results.md) describes the later paired run.

Historical mutation traces retain the paths at which those tests originally executed. Current helper commands use `tools/`; the source and numerical evidence are preserved.
