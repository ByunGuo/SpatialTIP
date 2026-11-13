"""Architecture initialization; no weight downloads or filesystem work at import."""

import torch
from torch import nn
from transformers import BertConfig

from spatialtip import BertLMHeadModel


class SpatialTIPBase(nn.Module):
    def init_vision_encoder(self, model_name="vit_large_patch16_224", img_size=224,
                           patch_size=16, init_values=1e-5, dynamic_img_size=True,
                           num_classes=0, vision_checkpoint=None):
        import timm

        if model_name != "vit_large_patch16_224":
            raise ValueError("The released architecture requires the 24-block UNI ViT-L encoder.")
        model = timm.create_model(
            model_name, pretrained=False, img_size=img_size, patch_size=patch_size,
            init_values=init_values, num_classes=num_classes,
            dynamic_img_size=dynamic_img_size,
        )
        if vision_checkpoint is not None:
            state = torch.load(vision_checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
        return model

    def init_stGPT(self, vocab_size=20938, max_seq_len=6001, tissue_type=5,
                   hidden_size=512, num_hidden_layers=6, num_attention_heads=8,
                   intermediate_size=1024, attention_backend="auto"):
        config = BertConfig(
            vocab_size=vocab_size, hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act="gelu", add_cross_attention=True,
        )
        config.encoder_width = hidden_size
        config.cross_attention_freq = 1
        config.query_length = 0
        config.max_seq_len = max_seq_len
        config.tissue_type = tissue_type
        config.mask_token_id = -1
        config.use_tissue = True
        config.use_cls = True
        config.attention_backend = attention_backend
        return BertLMHeadModel(config)


def disabled_train(self, mode=True):
    return self
