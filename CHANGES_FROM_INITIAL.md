# Original implementation and revision additions

Updated 2026-11-14. Baseline: https://github.com/Yoyiming/SpatialTIP/tree/d2d076f0f2aa5195df926bbf916b83976a22d529

All 46 original public files remain. The four original fine-tuning/inference scripts
retain only the previously added checkpoint prefix compatibility. The original
training, input masking, model, evaluation and clustering computation is unchanged.
The clustering notebook's evaluation cell additionally calls spatial_metrics.py;
all other notebook cells are unchanged. README includes archive and revision usage.

The authorized revision additions comprise independent EDM preparation, actual EDM
fine-tuning/inference and recovery evaluation dependencies; MAGIC, MIST, stLearn,
GraphST/STAGATE/SEDR adapters and existing-prediction evaluation; reference-method
adapters; original-zero proxy/marker evaluation; unknown-organ tissue handling and
optional HD image resizing. No dedicated ablation, raw-control, sensitivity or
gene-sparsity experiment runners are included.

The EDM scientific implementations were taken from the actual research scripts.
Release adaptations remove an unused scFoundation wildcard import, expose the
preparation source paths and training-only gene-effect inputs, use project-relative
baseline paths, remove the ablation CLI, provide annotation assembly from original
DLPFC labels and optional prediction-path manifests. The proxy driver retains the
reported raw-score comparisons and omits exploratory threshold-calibration branches.
The original-zero analysis still requires matching raw reference counts.

Validation: all Python and notebook code parses; nine EDM seed/ratio combinations
match the original dependent_mask function exactly; a real H5AD masking roundtrip
passes; recovery/spatial metric fixtures pass; shared EDM utility definitions are
AST-identical to the original utility functions. Only the notebook evaluation cell
changed. Full CUDA training, third-party baseline execution, real checkpoint loading
and historical result reproduction have not been rerun. The documented historical
corrected healthy checkpoint is not silently replaced with the archived original one.

This update contains the author-selected revision experiment code and spatial-smoothing alignment.

Author-authorized spatial-smoothing alignment: all SpatialTIP output paths now share spatial_smoothing.py. The original evaluator and notebook now pool synchronously from the fixed completed matrix; the notebook assembly now preserves observed nonzeros and fills zeros. Radius/KNN graph is undirected and includes self. This supersedes the unchanged-evaluation statements above for these specific steps. Historical metrics were not rerun.
