# External Assets Manifest

This directory is the authoritative registry for files intentionally absent from the Git repository.

The source inventory is the repository-pruning report supplied during finalization. The registry distinguishes four cases:

- **PUBLISH_EXTERNAL** — canonical heavy assets intended for public distribution outside Git history, currently through GitHub Releases.
- **LOCAL_ONLY** — useful forensic, intermediate, or non-canonical files retained only in the local project archive.
- **NOT_DISTRIBUTED** — CelebA-derived image data or archives intentionally absent from the public repository; public redistribution is not assumed.
- **EXCLUDED_LEGACY** — obsolete helper/link files superseded by the centralized registry.

## Canonical public checkpoints

The following three canonical checkpoints are published in GitHub Release [`v1.0.0`](https://github.com/DrAndreyBardin/DLS_FR_Spring_2026/releases/tag/v1.0.0):

| Asset | Release filename | Source filename | Size | SHA-256 |
|---|---|---|---:|---|
| Double 2-stack Hourglass | `double_hourglass_2stack_best.pt` | `best.pt` | 75.4 MB | `bbb6f5e27d5527029081bc2603ffb9d29803442816ea8390e893cb886e57a152` |
| Cross-Entropy ResNet18 | `ce_resnet18_best.pt` | `best.pt` | 44 MB | `6f9d679d49ad4df299c0aa5f2c091ee8d7d6635e5e03e975e2b7217bab7235ec` |
| ArcFace ResNet18 | `arcface_resnet18_best.pt` | `best.pt` | 44 MB | `dea25ee62641011ef560d508aac8ccbca573b91739776d3b508ad05a26c2d1e0` |

The corresponding direct release-asset URLs are recorded in `manifest.csv`, which remains the authoritative registry for externally distributed assets.

## Notes

- `latest.pt` and reference checkpoints are not public canonical assets and remain local.
- The ArcFace epoch-15 reference checkpoint is preserved only as an internal control; the canonical ArcFace checkpoint remains the original `best.pt`.
- Large CelebA-derived image trees and archives are intentionally not assigned GitHub Release storage in this manifest.
- Large deterministic diagnostic/intermediate tables remain local when compact summaries or remastered manifests already exist in Git.
- The source report contains one apparent unit inconsistency for `6.dls_fr_task3_output_v32_500`: `810.5/267.5KB` is interpreted as approximately 810.5 KB before pruning because the omitted `aligned_faces/` is 542.9 KB and the retained directory is 267.5 KB.
- `7.ce_vs_arcface_tpr_fpr_results` has no omitted files in the supplied report and therefore has no row in `manifest.csv`; the result bundle is retained in Git as a complete canonical artifact.

## Update rule

`manifest.csv` is the single source of truth for heavy or intentionally omitted assets. Avoid creating parallel per-directory link files unless a future usability requirement specifically justifies them.
