# SpatialTIP

SpatialTIP combines paired histology patches and gene-expression tokens for spatial transcriptomics imputation. This source release provides the core data preparation, pretraining, target fine-tuning, inference, recovery evaluation, spatial clustering, and attention-association export workflows.

This is a standalone cleanup of the current research implementation, compared with the [initial public repository](https://github.com/Yoyiming/SpatialTIP/tree/d2d076f0f2aa5195df926bbf916b83976a22d529). Supplementary experiment runners, baseline implementations, ablations, sensitivity sweeps, plotting scripts, temporary scripts, manuscripts, logs, and generated results are excluded.

A categorized comparison with the initial release, including behavior changes and compatibility notes, is available in [CHANGES_FROM_INITIAL.md](CHANGES_FROM_INITIAL.md) (Chinese).

## Installation

Linux with an NVIDIA GPU is recommended for full-size training. The configuration uses UNI ViT-L and a six-layer gene decoder; complete training requires substantial GPU memory. The explicit CPU attention path supports small checks, but is not a practical recommendation for full-size training.

Create an isolated Python environment from the repository root:

```bash
conda env create -f environment.yml
conda activate spatialtip
```

Install one matching PyTorch/torchvision pair, then the remaining dependencies. For CUDA 11.8:

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

For CPU checks, replace the first command's index URL with `https://download.pytorch.org/whl/cpu`. These wheel pairs follow the [official PyTorch installation table](https://pytorch.org/get-started/previous-versions/). The Conda file intentionally does not contain machine-specific build strings or an absolute environment prefix.

FlashAttention is optional. With a compatible Linux/CUDA installation, `flash-attn==2.6.3` can be installed separately. `attention_backend: "auto"` uses it for compatible CUDA half-precision tensors when available, otherwise PyTorch scaled-dot-product attention. Attention-logit extraction explicitly computes attention scores. `"flash"` requires the compatible extension; `"sdpa"` or `"math"` selects the other paths explicitly. Backend changes may produce small numerical differences.

Spatial clustering also requires `Rscript` on `PATH` and R's `mclust` package:

```r
install.packages("mclust")
```

No local R installation paths are set by the Python code. Use `clustering.py --rscript /path/to/Rscript` if necessary. Other workflows do not require R or rpy2.

## Required resources

Datasets and model weights are not bundled. Obtain UNI weights under the provider's applicable access and license terms. New pretraining requires `--vision-checkpoint /path/to/UNI/pytorch_model.bin`. A complete SpatialTIP checkpoint already contains its vision encoder, so inference/fine-tuning requires only that complete checkpoint.

A legacy SpatialTIP checkpoint must be used with **its exact original gene-to-ID vocabulary and tissue-to-ID mapping**. Matching gene counts does not prove ID compatibility. This cleanup does not replace missing current vocabularies with the old public repository's JSON files, and does not provide a new weight download URL. Record IDs and mappings alongside your checkpoints. New checkpoints store the vocabulary file's SHA256 and architecture configuration; the loader checks them.

`configs/model.example.json` contains the current healthy-model architecture defaults, including 20,938 token IDs and five tissue IDs. It is an editable example, not a declaration that an arbitrary checkpoint matches. Set `vocab_size` to the vocabulary length including `CLS=0`, `tissue_type` to the tissue mapping length, and `max_seq_len` to the intended gene-panel length plus one. Use the appropriate configuration and mappings for a disease model. Changing vocabulary IDs is incompatible with an existing checkpoint even if strict tensor loading succeeds.

Processed expression H5AD files must have:

- Unique spot and gene identifiers; all genes present in the supplied vocabulary.
- Finite, nonnegative normalized/log-transformed expression in `X`.
- Integer `obs['tissue']` IDs from the checkpoint's tissue mapping.
- `obsm['spatial']` coordinates for spatial clustering or spatial recovery evaluation.

Patch HDF5 files must contain `img` with shape `(N, H, W, 3)` and dtype `uint8`, and a `barcode` dataset. Patches are matched by barcode, not by row position. Default input patches are 224 × 224 RGB; `--resize-images` explicitly permits bilinear resizing. Dataset image handles are opened independently in each loader worker.

## Data preparation

Optional, explicitly selected HEST downloads:

```bash
python hest_download.py --local-dir data/hest --sample-id MISC1
```

Use Hugging Face's normal user authentication or environment configuration when access requires it. The downloader contains no embedded credentials, does not log in on import, and requires `--all` for a full-dataset download.

The example CSV manifests contain illustrative paths and sample identifiers; edit them for your data. Required columns are `sample,tissue,h5ad,patches`. Relative paths resolve against the CSV's directory. They are not the paper's historical training/test split.

For a **new model**, fit new deterministic gene and tissue mappings using only the samples you explicitly select:

```bash
python preprocess.py fit \
  --manifest examples/train_samples.csv \
  --output-dir data/pretraining \
  --vocabulary resources/vocabulary.json \
  --tissue-map resources/tissue_map.json \
  --n-top-genes 3000
```

The pipeline selects Seurat-v3 HVGs on raw counts, normalizes total counts using Scanpy's default target, applies `log1p`, retains selected genes and spots with paired image patches, and writes numeric tissue IDs. New vocabulary IDs use a stable gene ordering. This deliberately does not reproduce the nondeterministic IDs of older set-based vocabulary generation.

Prepare a target using the frozen mappings:

```bash
python preprocess.py transform \
  --manifest examples/target_samples.csv \
  --output-dir data/target \
  --vocabulary resources/vocabulary.json \
  --tissue-map resources/tissue_map.json \
  --n-top-genes 6000
```

Target HVG candidates are restricted to the frozen vocabulary; library-size normalization still uses the full original count matrix before the selected genes are retained. The output may contain fewer than the requested number of genes if the intersection is smaller. No vocabulary IDs are added or reordered during `transform`. The prepared `manifest.csv` records the actual output paths and dimensions.

For historical checkpoints, prefer the original processed H5AD panels and original mappings. Reprocessing current raw data is not proof that a historical gene panel, normalization, or result has been reproduced. Preprocessing refuses to overwrite inputs and requires explicit `--overwrite` to replace generated outputs.

## Pretraining and fine-tuning

Update the example model configuration to match the fitted mappings before training:

```bash
python train.py \
  --manifest data/pretraining/manifest.csv \
  --config configs/model.example.json \
  --vocabulary resources/vocabulary.json \
  --vision-checkpoint checkpoints/UNI/pytorch_model.bin \
  --output-dir runs/pretrain \
  --epochs 80 --batch-size 32 --mask-prob 0.15
```

For multiple GPUs, replace `python train.py` with `torchrun --standalone --nproc_per_node=4 train.py`. The single-process command also works without distributed initialization. Different sections are loaded separately so a batch never mixes incompatible gene panels. The default spot split is 90/10 for pretraining and 80/20 for target fine-tuning (`--train-fraction`); this is an internal optimization split, not an independent-donor validation claim. DDP samplers may pad subsets with repeated spots to keep rank batch sizes equal; the run metadata records this.

Fine-tune a complete SpatialTIP checkpoint on one processed target:

```bash
python finetune.py \
  --expression data/target/TARGET_01.h5ad \
  --patches data/patches/TARGET_01.h5 \
  --config configs/model.example.json \
  --vocabulary resources/vocabulary.json \
  --checkpoint runs/pretrain/best.pt \
  --output-dir runs/target \
  --epochs 60 --batch-size 32 --mask-prob 0.6
```

Training samples observed zeros/nonzeros separately for MLM (default probabilities 0.05 and 0.15 for pretraining, with 0.6 for fine-tuning nonzeros). The mask token is `-1`. Reconstruction preserves the original MSE reduction over the complete batch × gene matrix after unselected entries are zeroed, rather than dividing only by the number of selected entries. The CLS-based bidirectional contrastive loss retains coefficient 0.5 and the original cross-rank gather convention. The first 18 UNI transformer blocks remain frozen; the final six blocks are trainable.

Training retains linear warmup and periodic cosine restarts: defaults are cycle length 40 epochs and warmup 5 for pretraining, cycle length 30 and warmup 2 for fine-tuning, minimum learning rate 0.00005, and peak decay 0.98 per cycle. These are explicit arguments. The independently expressed schedule does not import the scBERT scheduler. This release uses a shared trainer, valid single-process/DDP handling, complete batches and evaluation without gradient tracking. It does **not** claim bitwise reproduction of every historical batch-dropping, stochastic validation or stopping decision. The default patience is 10; `--patience 0` disables early stopping. A supplied SpatialTIP checkpoint initializes model weights; optimizer/scheduler state is not resumed. `best.pt`, `last.pt`, `training.csv`, and `run.json` are written only to a new or empty output directory.

## Imputation

```bash
python impute.py \
  --expression data/target/TARGET_01.h5ad \
  --patches data/patches/TARGET_01.h5 \
  --config configs/model.example.json \
  --vocabulary resources/vocabulary.json \
  --checkpoint runs/target/best.pt \
  --output outputs/TARGET_01_prediction.h5ad \
  --device cuda:0 --batch-size 1
```

The default `--input-mask zeros` converts zero input entries to `-1` and retains observed nonzero expression as context. `--input-mask observed` reproduces the older top-level imputation script's different condition of hiding nonzeros; `none` leaves the input unchanged. These conditions should not be mixed in comparisons. The output contains dense model predictions for **all** input spots and genes in their original order, without a CLS column, observed-value replacement, or spatial smoothing. The default batch size prioritizes memory use; adjust it for your hardware.

The loader accepts legacy bare state dictionaries with leading `module.` prefixes and the original public `spatialtip.*` decoder prefix, normalizing those to the retained internal `stGPT.*` keys. Tensor loading is strict; incompatible keys/shapes are errors. New release checkpoints additionally validate the vocabulary SHA256 and architecture metadata. Legacy weight files lack that binding, so the original vocabulary remains the user's responsibility.

## Held-out recovery evaluation

Uniform held-out evaluation is separate from training-time MLM. To prepare it:

```bash
python mask_expression.py \
  --truth data/target/TARGET_01.h5ad \
  --output data/masked/TARGET_01.h5ad \
  --heldout data/masked/TARGET_01_heldout.npz \
  --probability 0.2 --seed 2024
```

The sparse boolean NPZ uses **True = held out** in the original H5AD axis order. Start from a pretraining checkpoint that has not been adapted on the unmasked target. In particular, do not reuse `runs/target/best.pt` from the unmasked fine-tuning example above. Fine-tune and predict in separate masked-data output directories:

```bash
python finetune.py \
  --expression data/masked/TARGET_01.h5ad \
  --patches data/patches/TARGET_01.h5 \
  --heldout data/masked/TARGET_01_heldout.npz \
  --config configs/model.example.json \
  --vocabulary resources/vocabulary.json \
  --checkpoint runs/pretrain/best.pt \
  --output-dir runs/target_masked

python impute.py \
  --expression data/masked/TARGET_01.h5ad \
  --patches data/patches/TARGET_01.h5 \
  --config configs/model.example.json \
  --vocabulary resources/vocabulary.json \
  --checkpoint runs/target_masked/best.pt \
  --output outputs/TARGET_01_masked_prediction.h5ad \
  --device cuda:0
```

Held-out entries remain unknown inputs and are excluded from the training loss. Evaluate the masked-input prediction against the original processed truth:

```bash
python evaluate.py \
  --truth data/target/TARGET_01.h5ad \
  --imputed outputs/TARGET_01_masked_prediction.h5ad \
  --heldout data/masked/TARGET_01_heldout.npz \
  --output outputs/TARGET_01_metrics.csv
```

Use a prediction produced from the masked input, not a prediction from the unmasked target. RMSE/MAE pool only the held-out entries. Spot-wise/gene-wise PCC and cosine similarity average valid vectors, with valid-vector counts retained. Undefined values remain missing. To additionally evaluate observed-value completion and historical sequential spatial smoothing, supply `--input data/masked/TARGET_01.h5ad --smoothing radius --radius 150`; the output CSV labels each stage separately. No supplementary expression-dependent masking, proxy-label or baseline experiment runners are included.

## Spatial clustering

```bash
python clustering.py \
  --original data/target/TARGET_01.h5ad \
  --imputed outputs/TARGET_01_prediction.h5ad \
  --output outputs/TARGET_01_clustered.h5ad \
  --n-clusters 7 --pca-components 20 \
  --smoothing radius --radius 150 --refine-neighbors 30
```

This entry point retains original nonzero expression, fills original zeros with predictions, applies the historical **sequential** spatial mean including the center spot, runs PCA and mclust EEE, then applies simultaneous nearest-neighbor label refinement. Radius is measured in the units of `obsm['spatial']`; choose a suitable setting for a different platform. Sequential smoothing reads previously updated rows and therefore depends on spot order. It is not the synchronous kNN mean used by some sensitivity scripts. Those supplementary scripts are excluded.

`--smoothing none` disables expression smoothing; `--refine-neighbors 0` disables label refinement. For a matrix already containing the intended completed/smoothed output, `--already-integrated` bypasses both the observed-value merge and expression smoothing, avoiding duplicate processing. Refinement vote ties follow the first label in the ordered nearest-neighbor list; exact distance ties are ordered by spot index for deterministic behavior in this release. This tie handling is not a promise of bitwise reproduction of historical distance-library behavior.

`--n-clusters` is explicit and is never inferred from reference annotations. Reference labels are optional: use `--label-key ground_truth` only when that column exists and reference ARI/NMI is desired. A dataset without domain ground truth can still be clustered, but reference-label agreement metrics are unavailable. The output records the matrix, PCA coordinates, initial/refined labels, and processing parameters.

## Attention-derived associations

```bash
python output_attentions.py \
  --expression data/target/TARGET_01.h5ad \
  --patches data/patches/TARGET_01.h5 \
  --config configs/model.example.json \
  --vocabulary resources/vocabulary.json \
  --checkpoint runs/target/best.pt \
  --output outputs/TARGET_01_attention.npz --device cuda:0

python attention_network.py \
  --attention outputs/TARGET_01_attention.npz \
  --output outputs/TARGET_01_associations.csv --top-k 20
```

The extractor averages returned gene self-attention **pre-softmax logits** over heads, layers and all input spots, excluding CLS. Rows are query genes and columns are key genes; scores include the additive constants of the explicit attention mask. These are model-internal associations, not normalized probabilities or evidence of causal regulation. `--max-spots` explicitly selects a leading subset and records its size; the default processes all spots. Gene × gene attention extraction can use substantial memory. `attention_network.py --query GENE_A GENE_B` optionally ranks keys for selected query genes without altering the scores.

## Source layout and release scope

```text
SpatialTIP/
  spatialtip.py             # Gene decoder and auto-discretization embedding
  spatialtip_init.py        # UNI/BERT architecture initialization
  spatialtip_model.py       # Vision/gene model and training objectives
  runtime.py                # Configuration and strict checkpoint loading
  dataset.py                # Barcode-aligned H5AD/HDF5 dataset
  utils.py                  # Tokenization, masking and distributed helpers
  preprocess.py             # Explicit new/frozen vocabulary data preparation
  hest_download.py          # Optional, explicit dataset download
  train.py                  # Shared training engine and pretraining CLI
  finetune.py               # Single-target fine-tuning CLI
  impute.py                 # Dense prediction export
  mask_expression.py        # Uniform nonzero holdout preparation
  evaluate.py               # Held-out recovery metrics
  clustering.py             # Completion/smoothing/PCA/mclust/refinement
  output_attentions.py      # Gene self-attention logit extraction
  attention_network.py      # Ranked query/key association export
  configs/model.example.json
  examples/                 # Illustrative CSV formats, no dataset payloads
  environment.yml
  requirements.txt
  LICENSE
  THIRD_PARTY_NOTICES.md
  source_provenance.json
```

No entire `scBERT` or `scFoundation` distribution is bundled, and no unused loader or GPL scBERT scheduler is imported. Provenance and third-party attribution are retained in the listed files. Inputs, weights and generated outputs are ignored by Git by default.

## Validation and limits

The source was reorganized independently of the original research directory. Checks used an isolated Windows/Python 3.11 environment with the pinned CPU dependencies. They covered real Seurat-v3 preprocessing and H5AD/HDF5 data loading, the gene decoder's numerical agreement with the original explicit-attention implementation, synthetic checkpoint loading, single/two-process CPU training management, recovery metrics, and full clustering with R 4.4.2/mclust 6.1.1. Training-management fixtures use a small substitute model; decoder/wrapper checks use small tensors and a substitute vision encoder. Details are recorded in `source_provenance.json`.

These checks establish code behavior and file contracts, not the scientific accuracy of a trained model. Full-size GPU training, multi-GPU execution, real-checkpoint restoration with the currently unavailable local resources, and paper-result reproduction are not established by these checks. No trained weights, historical evaluation datasets or current vocabulary/tissue mappings were exported as part of this code-only cleanup.

## License

SpatialTIP source is distributed under Apache-2.0. See `LICENSE` and `THIRD_PARTY_NOTICES.md` for incorporated code attribution. External datasets and UNI weights retain their own terms.
