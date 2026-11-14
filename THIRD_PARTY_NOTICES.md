# Current original-logic release

The original scBERT dependency tree and its LICENSE are restored. The notice below records the earlier reorganized release and is retained as historical attribution.

# Third-party notices

The SpatialTIP source release retains the Apache License 2.0 in [LICENSE](LICENSE). The following adapted components retain their upstream attribution. External model weights are not included in this source release.

## Hugging Face Transformers: BERT implementation

The BERT classes used by SpatialTIP are adapted from the Hugging Face Transformers BERT implementation, with SpatialTIP-specific gene/expression embeddings, multimodal cross-attention, and prediction behavior.

- Upstream source: [Transformers v4.44.2, modeling_bert.py](https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/bert/modeling_bert.py).
- License: Apache License 2.0; see [upstream LICENSE](https://github.com/huggingface/transformers/blob/v4.44.2/LICENSE) and this release's [LICENSE](LICENSE).
- Upstream copyright notices:

  ```text
  Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
  Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
  ```

The upstream reference identifies the implementation family and license; it does not assert that the original SpatialTIP adaptation was copied from this exact Transformers release.

## scFoundation: auto-discretization embedding

SpatialTIP's `AutoDiscretizationEmbedding2` adapts the continuous-expression embedding in scFoundation's `model/pretrainmodels/mae_autobin.py`. SpatialTIP adds its own defaults and normalization and does not include the scFoundation model loader or pretrained model weights.

- Upstream source: [scFoundation, commit 397631c495eddf9ad6644fc00c6ea8139e651245](https://github.com/biomap-research/scFoundation/blob/397631c495eddf9ad6644fc00c6ea8139e651245/model/pretrainmodels/mae_autobin.py).
- License: Apache License 2.0; see [upstream LICENSE](https://github.com/biomap-research/scFoundation/blob/397631c495eddf9ad6644fc00c6ea8139e651245/LICENSE) and this release's [LICENSE](LICENSE).
- Upstream copyright notice:

  ```text
  Copyright 2023 BioMap (Beijing) Intelligence Technology Limited
  ```

The scFoundation upstream documentation distinguishes its source-code license from its model-weights license. This release distributes only the adapted embedding source described above.

## External dependencies and vision weights

PyTorch, Transformers, timm, and other packages are installed as dependencies rather than vendored as source in this release. Their licenses remain those of their respective projects. UNI vision weights must be obtained separately from [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI), subject to their own access and use terms; this repository's Apache license does not grant rights to those weights.

## scBERT

The scBERT source tree and model weights are not included in this release. This release does not copy `scBERT/utils.py` or rely on importing `scBERT` for its learning-rate schedule. The scBERT code included in the older public SpatialTIP tree carried a separate GPLv3 license and is not covered by SpatialTIP's Apache license.
