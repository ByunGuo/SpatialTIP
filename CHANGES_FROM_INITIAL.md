# Original-logic release

Updated 2026-11-14 following the author instruction to preserve the original implementation.

Baseline: https://github.com/Yoyiming/SpatialTIP/tree/d2d076f0f2aa5195df926bbf916b83976a22d529

The previous reorganized implementation has been withdrawn. All 46 original public files are restored, including separate ordinary and masked fine-tuning/inference entries, original vocabulary/tissue mappings, annotations and the clustering notebook. Sampling, data splitting, batching, validation RNG, learning-rate schedules, input masking, expression merging and spatial processing follow the baseline.

The only executable changes are checkpoint key-prefix compatibility in finetune.py, finetune_masked.py, impute.py and impute_masked.py. Research checkpoints use stGPT.; the original public model uses spatialtip. Loading strips module. and translates that decoder prefix, without changing tensor values or original strict-loading settings.

README adds the processed-data DOI and exact-mapping guidance. Existing Git ignore/attributes and source attribution records are retained.

Validation: all restored Python files parse. The four loading-key transformations preserve tensor-object identity and preserve original public keys. Full-size GPU inference/training and historical metric reproduction have not been rerun.
