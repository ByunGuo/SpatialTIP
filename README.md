# SpatialTIP: a multimodal pretrained framework for imputing missing values in spatial transcritomics

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
  - [Pretraining](#1-pretraining)
  - [Fine-tuning](#2-fine-tuning)
  - [Gene Imputation](#3-gene-imputation)
  - [Spatial Clustering](#4-spatial-clustering)
- [Project Structure](#project-structure)
- [Citation](#citation)

## Overview

SpatialTIP leverages a ViT-L vision encoder (UNI pretrained) cross-attended to a BERT-style gene expression decoder. 

### Key Features

- 🔬 **Gene Imputation**: Predict missing/unexpressed gene values
- 🗺️ **Spatial Clustering**: Identify spatial domains from imputed expression
- 🖼️ **Vision-Gene Integration**: Combines histology images with transcriptomics

## Installation

### Requirements

- Python 3.9
- CUDA 11.8
- R with mclust package (for clustering)

### Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate spatialtip

# Set R environment variables (Linux)
export R_HOME=/root/miniconda3/envs/spatialtip/lib/R
export R_USER=/root/miniconda3/envs/spatialtip/lib/python3.9/site-packages/rpy2
```

### Download Pretrained Weights

Download the UNI ViT-L pretrained weights and place them at:
```
uni/checkpoints/pytorch_model.bin
```

## Data Preparation

### Data Format

SpatialTIP expects the following data structure:

```
hest_data/
├── st/                          # Raw spatial transcriptomics data (.h5ad)
│   ├── MISC1.h5ad
│   ├── MISC2.h5ad
│   └── ...
├── patches/                     # Image patches (.h5)
│   ├── MISC1.h5
│   ├── MISC2.h5
│   └── ...
├── vocabulary_healthy_3k.json  # Gene vocabulary
└── tissue_dict.json             # Tissue type mapping
```

### Data Processing Pipeline

**Step 1: Download HEST Dataset** (Optional)

```bash
python hest_download.py
```

**Step 2: Preprocess Data**

This step:
1. Selects highly variable genes (HVGs) across samples
2. Creates gene vocabulary (`vocabulary_*.json`)
3. Creates tissue type mapping (`tissue_dict*.json`)
4. Filters and normalizes expression data

## Model Architecture

### Components

| Component | File | Description |
|-----------|------|-------------|
| Core BERT | `spatialtip.py` | BERT-style encoder-decoder with cross-attention |
| Model Wrapper | `spatialtip_model.py` | Combines vision encoder with gene decoder |
| Initialization | `spatialtip_init.py` | Model initialization utilities |
| Dataset | `dataset.py` | Data loading utilities |


### Configuration Parameters

**Important:** Model configuration in `spatialtip_init.py` depends on the training dataset. Key parameters that vary:

| Parameter | Description | Varies By |
|-----------|-------------|-----------|
| `vocab_size` | Gene vocabulary size | Number of HVGs in dataset |
| `tissue_type` | Number of tissue types | Tissue categories in dataset |
| `max_seq_len` | Maximum sequence length | Training stage (see below) |

**Sequence Length Configuration:**
- **Pretraining**: `max_seq_len = 3001` (shorter for efficiency across multiple samples)
- **Fine-tuning**: `max_seq_len = 6001` (longer for sample-specific genes)

When using a different dataset, update these values in `spatialtip_init.py` accordingly.

## Usage

### 1. Pretraining

Train on multiple samples for general representation learning:

```bash
torchrun --nproc_per_node=4 train.py \
    --batch_size 32 \
    --epochs 80 \
    --learning_rate 1e-4 \
    --mask_prob 0.15 \
    --world_size 4 \
    --save_dir spatialtip_model
```

**Key Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | 32 | Batch size per GPU |
| `--epochs` | 80 | Number of training epochs |
| `--learning_rate` | 1e-4 | Learning rate |
| `--mask_prob` | 0.15 | Probability of masking non-zero genes |
| `--world_size` | 4 | Number of GPUs |
| `--save_dir` | spatialtip_model | Model save directory |

### 2. Fine-tuning

There are two fine-tuning modes depending on your goal:

#### Standard Fine-tuning (`finetune.py`)

Fine-tune on **original unmasked data** for general adaptation to a specific sample:

```bash
torchrun --nproc_per_node=4 finetune.py \
    --finetune_sample MISC1 \
    --epochs 60 \
    --train_mask_prob 0.6 \
    --batch_size 32
```

#### Masked Fine-tuning (`finetune_masked.py`)

Fine-tune on **artificially masked data** for imputation evaluation. This mode:
1. Randomly masks a portion of expressed genes (specified by `--test_mask_prob`)
2. Saves the mask for later evaluation
3. Trains the model to predict masked values

```bash
torchrun --nproc_per_node=4 finetune_masked.py \
    --finetune_sample MISC1 \
    --test_mask_prob 0.2 \
    --epochs 30
```

| Mode | Script | Data | Use Case |
|------|--------|------|----------|
| Standard | `finetune.py` | Original data | General adaptation |
| Masked | `finetune_masked.py` | Artificially masked data | Imputation evaluation |

### 3. Gene Imputation

#### General Imputation (`impute.py`)

Predict expression for **unexpressed genes** (genes with zero counts). This masks all non-zero values and asks the model to predict what the zero values should be. Works with any model weights (pretrained or fine-tuned):

```bash
python impute.py \
    --test_sample MISC1 \
    --model_dir spatialtip_model \
    --model_name spatialtip_hest_healthy_3k.pt
```

**Output:** `{output_dir}/{sample}_imputed.h5ad`

| Argument | Default | Description |
|----------|---------|-------------|
| `--test_sample` | MISC1 | Sample to impute |
| `--model_dir` | spatialtip_model | Model directory |
| `--model_name` | spatialtip_hest_healthy_3k.pt | Model filename |
| `--data_dir` | hest_data | Data directory |
| `--data_subdir` | st_healthy_6k_v1 | Data subdirectory |
| `--patches_dir` | hest_data/patches | Patches directory |
| `--vocabulary_path` | hest_data/vocabulary_healthy_3k.json | Vocabulary file |
| `--output_dir` | Result/st_healthy_6k | Output directory |
| `--device` | cuda:0 | Device to use |
| `--batch_size` | 64 | Batch size |

#### Masked Imputation (`impute_masked.py`)

Evaluate imputation accuracy on **artificially masked data**. This uses data where some expressed genes were masked out, and compares predictions against ground truth. Requires fine-tuned model on the masked data:

```bash
python impute_masked.py \
    --test_sample MISC1 \
    --test_mask_prob 0.2 \
    --model_dir spatialtip_model
```

**Output:** `{output_dir}/{sample}_{mask_prob}.h5ad`

| Argument | Default | Description |
|----------|---------|-------------|
| `--test_sample` | MISC2 | Sample to evaluate |
| `--test_mask_prob` | 1.0 | Masking probability used in data |
| `--model_dir` | spatialtip_model | Model directory |
| `--model_name` | spatialtip_hest_healthy_{sample}_finetune.pt | Model filename (auto-set if not specified) |
| `--data_dir` | hest_data | Data directory |
| `--masked_data_subdir` | st_healthy_6k_masked_{seed} | Masked data subdirectory |
| `--original_data_subdir` | st_healthy_6k_v1 | Original data subdirectory |
| `--masks_dir` | hest_data/masks_healthy_6k_{seed} | Masks directory |
| `--output_dir` | Result/predicted_healthy_6k_masked_{seed} | Output directory |
| `--device` | cuda:0 | Device to use |
| `--batch_size` | 32 | Batch size |

| Script | Input Data | Use Case |
|--------|------------|----------|
| `impute.py` | Original data | General imputation of unexpressed genes |
| `impute_masked.py` | Pre-masked data | Evaluate imputation accuracy with ground truth |

### 4. Spatial Clustering

Perform spatial domain identification using the imputed gene expression. See `spatial_domain_clustering.ipynb` for a step-by-step tutorial.

**Key Steps:**
1. Load original and imputed data
2. Combine imputed values with original non-zero values
3. Apply spatial neighbor smoothing
4. Dimensionality reduction with PCA
5. Clustering with mclust (GMM-based)
6. Optional label refinement

**Input Files:**
- Original adata: `hest_data/st/{sample}.h5ad`
- Imputed adata: `Results/{sample}/spatialtip_imputed.h5ad`

**Output:**
- Clustering visualization: `figures/clustering/{sample}/clustering_result.png`
- Clustered adata: `Results/{sample}/spatialtip_clustered.h5ad`

### Evaluation Metrics

Calculate evaluation metrics for imputation results:

```bash
python evaluate.py \
    --sample MISC1 \
    --mask_prob 0.2
```

**Output:** `{output_dir}/{sample}_{mask_prob}_evaluation_metrics.csv`

**Metrics:**
- RMSE / MAE
- Spot-wise Pearson Correlation Coefficient (PCC)
- Spot-wise Cosine Similarity
- Gene-wise PCC
- Gene-wise Cosine Similarity

| Argument | Default | Description |
|----------|---------|-------------|
| `--sample` | MISC3 | Sample name |
| `--mask_prob` | 0.2 | Mask probability |
| `--seed` | 2024 | Random seed |
| `--data_dir` | hest_data | Data directory |
| `--masked_data_subdir` | st_healthy_6k_masked_{seed} | Masked data subdirectory |
| `--original_data_subdir` | st_healthy_6k_v1 | Original data subdirectory |
| `--imputed_data_dir` | Result/predicted_healthy_6k_masked_{seed} | Imputed data directory |
| `--masks_dir` | hest_data/masks_healthy_6k_{seed} | Masks directory |
| `--output_dir` | Metrices/healthy_6k_{seed} | Output directory |
| `--no_spatial_smoothing` | False | Disable spatial neighbor smoothing |
| `--rad_cutoff` | 150 | Radius cutoff for spatial network |

## Project Structure

```
SpatialTIP/
├── spatialtip.py              # Core BERT architecture
├── spatialtip_model.py        # Main model class
├── spatialtip_init.py         # Model initialization
├── dataset.py                 # Dataset classes
├── utils.py                   # Utility functions
│
├── train.py                   # Pretraining script
├── finetune.py                # Fine-tuning script
├── finetune_masked.py          # Fine-tuning on masked data
│
├── impute.py                  # General imputation
├── impute_masked.py           # Imputation on masked data
├── evaluate.py                # Evaluation metrics
├── spatial_domain_clustering.ipynb  # Spatial clustering tutorial
│
├── preprocess_hest_healthy_union.py  # Healthy data preprocessing
├── preprocess_hest_cancer_union.py   # Cancer data preprocessing
├── hest_download.py           # HEST dataset downloader
│
├── environment.yml            # Conda environment
└── README.md                  # This file
```

## Sample ID Mapping

DLPFC samples map to standard dataset IDs:

| Sample | Dataset ID | Sample | Dataset ID |
|--------|-----------|--------|-----------|
| MISC1 | 151676 | MISC7 | 151670 |
| MISC2 | 151675 | MISC8 | 151669 |
| MISC3 | 151674 | MISC9 | 151510 |
| MISC4 | 151673 | MISC10 | 151509 |
| MISC5 | 151672 | MISC11 | 151508 |
| MISC6 | 151671 | MISC12 | 151507 |

## Dependencies

| Package | Version |
|---------|---------|
| PyTorch | 2.4.0 |
| CUDA | 11.8 |
| flash-attn | 2.6.3 |
| timm | 1.0.9 |
| transformers | 4.44.2 |
| scanpy | 1.10.2 |
| anndata | 0.10.9 |
| rpy2 | 3.5.11 |

## License

This project is licensed under the Apache 2.0 License.


## Archived analysis resources

Processed datasets: https://doi.org/10.5281/zenodo.24190756

Use each SpatialTIP checkpoint with its exact original gene-to-ID vocabulary and tissue mapping. The archived research checkpoints use the `stGPT.` decoder prefix; the loading statements in the four fine-tuning/inference entries translate it to the original public `spatialtip.` prefix. The original entry points retain the original public implementation. Independent revision experiment entries are described below.


## Revision experiment code

The original training, inference and clustering pipeline remains unchanged. The additional
EDM entry points retain the actual revision implementation independently of the original
uniform-mask experiment. Run commands from the repository root. GPU training/inference
requires the original CUDA environment and FlashAttention; CPU checks do not validate GPU runs.

### EDM preparation and recovery

`experiments/edm/prepare.py` generates a boolean mask (`True = held-out observed nonzero`),
`input.h5ad`, and its configuration. Intrinsic zeros are never selected. Supply frozen
training-derived gene effects, or fit them using only the exact training panels belonging
to the selected pretrained model. Target expression must not enter that fit.

```bash
python -m experiments.edm.prepare --sample MISC1 \
  --expression hest_data/st_healthy_opus/MISC1.h5ad \
  --patches hest_data/st/patches/MISC1.h5 \
  --training-inputs TRAINING_PANEL_1.h5ad TRAINING_PANEL_2.h5ad \
  --ratio 0.4 --seed 42
```

Replace the two illustrative training paths with the **complete actual training panel list**.
For subsequent targets use `--gene-effects PATH_TO_GENERATED_GENE_EFFECT_JSON` instead.
Masking defaults to `experiments/edm/healthy_expression_smoke/expression_gene/42/MISC1_0.4/`;
this historical directory name is retained for the baseline adapters. It does not imply
that the preparation or a run is a smoke test. Other samples/ratios use the matching path.
To reproduce existing results, the source gene/spot axes, normalization and frozen training
effects must match the experimental input version; a different processed panel is a new run.

```bash
torchrun --standalone --nproc_per_node=4 finetune_edm.py \
  --run_mode formal --finetune_sample MISC1 --seed 42 --test_mask_prob 0.4 \
  --adata_path MASK_FOLDER/input.h5ad --heldout_path MASK_FOLDER/heldout.npz \
  --patches_path hest_data/st/patches/MISC1.h5 \
  --vocabulary_path MATCHING_VOCABULARY.json --checkpoint_path MATCHING_CHECKPOINT.pt \
  --save_dir MODEL_OUTPUT
python impute_edm.py --run_mode formal --test_sample MISC1 \
  --adata_path MASK_FOLDER/input.h5ad --patches_path hest_data/st/patches/MISC1.h5 \
  --vocabulary_path MATCHING_VOCABULARY.json --checkpoint_path SELECTED_CHECKPOINT.pt \
  --output_path PREDICTION_FOLDER/prediction.h5ad
python evaluate_edm.py --sample MISC1 --seed 42 --mask_prob 0.4 \
  --adata_path MASK_FOLDER/input.h5ad --truth_path hest_data/st_healthy_opus/MISC1.h5ad \
  --prediction_path PREDICTION_FOLDER/prediction.h5ad \
  --heldout_path MASK_FOLDER/heldout.npz --output_dir METRICS_FOLDER
```

Replace uppercase paths with actual files. `--run_mode formal` is required for a full run;
the retained experimental default is a limited smoke run. Pass the actual model's
`--vocab_size`, `--tissue_type` and `--max_seq_len` where they differ from healthy defaults.
NoFT uses the pretrained checkpoint directly at inference. Full checkpoints must match
their exact gene/tissue mappings. The historical EDM healthy scripts used the `corrected`
checkpoint; the originally released healthy checkpoint is not asserted to reproduce that
parameter state. This source update does not substitute one for the other.

### Cross-tissue inputs

Use the same EDM entry points. For HD patches add `--resize_images` to both training and
inference. For the actual unknown-heart-organ policy (`tissue=-1`), substitute
`experiments/cross_tissue/run_organ_entry.py train` or `experiments/cross_tissue/run_organ_entry.py infer`
for the corresponding script name, retaining all remaining arguments. This wrapper uses
a zero tissue feature for unknown organs, as in the experiment. Known tissue features
are unchanged. No separate cross-tissue training pipeline is introduced.

### Baseline comparisons

The adapters below retain their experimental algorithm calls and settings. Install each
method in its own compatible environment. For source-based dependencies, place the
original method code in `../baselines/{GraphST,STAGATE,SEDR,MIST,stlearn}`. These third-party
sources, model weights and environments are not bundled here.

```bash
python -m experiments.edm.run_magic --samples MISC1 --ratio 0.4
python -m experiments.edm.run_uploaded_graph_baselines --methods GraphST --samples MISC1 --ratio 0.4
python -m experiments.edm.run_mist --samples MISC1 --ratio 0.4
python -m experiments.edm.run_stlearn --samples MISC1 --ratio 0.4
python -m experiments.edm.evaluate_uploaded_reference_predictions \
  --method BLEEP --samples MISC1 --ratio 0.4 \
  --prediction-dir BASELINE_PREDICTIONS --output-dir BASELINE_METRICS
```

GraphST/STAGATE/SEDR use the corresponding `--methods` value. BLEEP and OmiCLIP retained
reference-generated predictions are evaluated through the last command; their training
is not replaced by a new implementation. Baseline adapters expect normalized truth at
`hest_data/st_healthy_opus/<sample>.h5ad`, patches at `hest_data/st/patches/<sample>.h5`,
and the frozen masks at the preparation paths above. Baseline predictions must be from
the intended experimental protocol, with matching gene/barcode identities.

ALRA/scImpute/ZINB-WaVE reference adapters are provided in `run_reference_pilot.R`.
Use `python -m experiments.edm.export_reference_inputs --sample MISC1 --output REF_INPUT`,
then `Rscript experiments/edm/run_reference_pilot.R . REF_INPUT REF_OUTPUT ALRA formal`
(or `scImpute`/`ZINB_WaVE`). These reference/proxy adapters require matching raw integer
counts at `hest_data/st/<sample>.h5ad`; normalized expression is not a substitute.
They are separate from the EDM recovery baseline table.

### Original-zero proxy analysis

```bash
python -m experiments.edm.evaluate_cross_section_zero_proxy
python -m experiments.edm.evaluate_layer_marker_distortion
python -m experiments.edm.evaluate_baseline_false_positive --methods SpatialTIP
```

The five target sections, same-patient reference sections, label rules and numerical
scoring follow the actual revision analysis. These are operational proxy labels, not
molecular ground truth. DLPFC annotations are assembled from the original repository's
`DLPFC/*/*_truth.txt`. The analysis additionally requires the reference sections' raw
counts with coverage of target genes, the matching normalized target panels, EDM masks,
and saved predictions. The processed-data archive alone does not supply all these inputs.

Historical prediction locations are retained. Optionally set `SPATIALTIP_PREDICTION_MANIFEST`
to a JSON file mapping method then sample to the prediction file, for example
`{"SpatialTIP": {"MISC1": "/path/MISC1/prediction.h5ad"}}`, with all five samples present.
SpatialTIP files must be named `prediction.h5ad` for the label-building scripts.
Use the final evaluator's `--methods` to select available methods; each comparison uses
its common gene universe. Outputs from a different method set are not automatically
identical to a previously reported comparison.

### Spatial-domain metrics

Section 8 of `spatial_domain_clustering.ipynb` now also calls `spatial_metrics.py` for NMI,
boundary-edge F1, boundary accuracy and spatial continuity. The preceding clustering
cells are unchanged. Edges use a radius of 1.25 times the median nearest-neighbor distance;
boundary metrics use edges with annotated endpoints. Continuity uses all spatial edges.

Reference adapter package versions used in the research environment were ALRA 0.0.0.9000, scImpute 0.0.9 and ZINB-WaVE 1.24.0. The graph/SME baseline adapters require their original method dependencies; they are not part of the base SpatialTIP environment.


### Shared spatial smoothing

All SpatialTIP output-smoothing entries use `spatial_smoothing.py`. Observed nonzero
input values are retained and predictions fill input zeros before pooling. A single
average is computed from that fixed assembled matrix over self and the undirected
neighbor graph. DLPFC uses radius 150. Other platforms can set `--n_neighbors` in
`evaluate.py` or `evaluate_edm.py` to the experiment's 6--12 neighbors, excluding self.
KNN edges are symmetrized, so the final undirected neighborhood can exceed k.
This replaces sequential in-place updates in the original evaluation/notebook;
historical numerical results have not been recomputed. Original-zero marker metrics
still restore original observed nonzeros after smoothing solely for that endpoint,
as required to isolate effects at originally zero entries. Proxy-label construction
and baseline methods' own spatial operations are unchanged.
