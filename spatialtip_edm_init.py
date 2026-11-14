import os
import torch
from torchvision import transforms
import timm
import torch.nn as nn
from spatialtip_edm import BertLMHeadModel
from transformers import BertConfig


class stGPTBase(nn.Module):
    def __init__(self):
        super(stGPTBase, self).__init__()

    def init_vision_encoder(self, model_name="vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, dynamic_img_size=True, num_classes=0, vision_checkpoint="uni/checkpoints/pytorch_model.bin"):
        # Load the img model
        model = timm.create_model(model_name, img_size=img_size, patch_size=patch_size, init_values=init_values,
                                  num_classes=num_classes, dynamic_img_size=dynamic_img_size)
        # A complete SpatialTIP checkpoint supplies these weights during strict loading.
        if vision_checkpoint is not None:
            model.load_state_dict(torch.load(vision_checkpoint, map_location='cpu'), strict=True)

        return model  
    
    def init_stGPT(self, vocab_size=20938, max_seq_len=6001, tissue_type=5):
        encoder_config = BertConfig()
        encoder_config.encoder_width = 512
        encoder_config.add_cross_attention = True
        encoder_config.hidden_act = "gelu"
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = 0
        encoder_config.num_hidden_layers = 6
        encoder_config.num_attention_heads = 8
        encoder_config.hidden_size = 512
        encoder_config.intermediate_size = 1024
        encoder_config.max_seq_len = max_seq_len
        encoder_config.vocab_size = vocab_size
        encoder_config.tissue_type = tissue_type
        encoder_config.mask_token_id = -1
        encoder_config.use_tissue = True
        encoder_config.use_cls = True
        
        print("vocab_size:", vocab_size)
        print("tissue_type:", tissue_type)
        print("max_seq_len:", max_seq_len)

        stGPT = BertLMHeadModel(encoder_config)
        return stGPT


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self
    