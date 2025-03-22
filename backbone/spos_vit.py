import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import sys
import os
sys.path.append(os.getcwd())
from backbone.vit_block import PatchEmbed, Block, Sub_Block
from backbone.utils import trunc_normal_, _no_grad_trunc_normal
from util.dist_init import random_choice

__all__ = [
    "vit_small_patch16_224",
    "vit_base_patch32_224",
    "vit_base_patch16_224",
    "vit_base_patch14_224",
    "vit_large_patch32_224",
    "vit_large_patch16_224",
    "vit_large_patch14_224",
    "vit_huge_patch32_224",
    "vit_huge_patch16_224",
    "vit_huge_patch14_224"
]

class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, qkv_bias=False, 
                 qk_scale=None, drop_rate=0., attn_drop_rate=0.,norm_layer=nn.LayerNorm, pretrain=False, drop_path_rate=0, 
                 classifier='token', num_heads_list=None, mlp_ratio_list=None, depth_list=None):
        super().__init__()
        self.classifier = classifier
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.depth_list = depth_list

        self.depth_blocks = nn.ModuleList([])
        for depth in depth_list:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
            blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, num_heads_list=num_heads_list,mlp_ratio_list=mlp_ratio_list)
                for i in range(depth)])
            self.depth_blocks.append(blocks)
        self.norm = norm_layer(embed_dim)

        self.pretrain = pretrain
        self.repr = nn.Sequential(
            nn.Linear(embed_dim, self.embed_dim),
            nn.GELU()
        )

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        trunc_normal_(self.pos_embed, std=.02)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                trunc_normal_(m.bias, std=1e-6)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes)
        # self.head.weight.data.normal_(mean=0.0, std=0.01)
        self.head.weight.data.zero_()
        self.head.bias.data.zero_()

    def forward_features(self, x, return_feat=False, depth_idx=np.random.randint(3), attn_idx=np.random.randint(4, size=14), mlp_idx=np.random.randint(3, size=14)):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.depth_blocks[depth_idx]):
            x = blk(x, attn_idx[i], mlp_idx[i])

        x = self.norm(x)
        if return_feat:
            return x

        if self.classifier == 'mean':
            return x[:, 1:].mean(dim=1)
        return x[:, 0]
    
    def freeze_backbone(self):
        for name, param in self.named_parameters():
            if not 'head' in name:
                param.requires_grad = False

    def forward(self, 
                x, 
                depth_idx=np.random.randint(3), 
                attn_idx=np.random.randint(4, size=14), 
                mlp_idx=np.random.randint(3, size=14),
                return_feat=False):
        """ Indices contain the index of candidate w.r.t block depth, attn_heads, mlp_ratio
        """
        x = self.forward_features(x, return_feat, depth_idx, attn_idx[:self.depth_list[depth_idx]], mlp_idx[:self.depth_list[depth_idx]])
        if return_feat:
            return x
        if self.pretrain:
            x = self.repr(x)
        x = self.head(x)
        return x

class Sub_VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, qkv_bias=False, 
                 qk_scale=None, drop_rate=0., attn_drop_rate=0.,norm_layer=nn.LayerNorm, pretrain=False, drop_path_rate=0, 
                 classifier='token', depth_idx=None, num_heads_idx=None, mlp_ratio_idx=None):
        super().__init__()
        self.classifier = classifier
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.depth_list = [12,13,14]
        self.num_heads_list = [3,4,6,8]
        self.mlp_ratio_list = [3,4,5]

        # self.depth_blocks = nn.ModuleList([])
        # for depth in depth_list:
        depth = self.depth_list[depth_idx]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Sub_Block(
                dim=embed_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, num_heads=self.num_heads_list[num_heads_idx[i]],mlp_ratio=self.mlp_ratio_list[mlp_ratio_idx[i]])
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.pretrain = pretrain
        self.repr = nn.Sequential(
            nn.Linear(embed_dim, self.embed_dim),
            nn.GELU()
        )

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        trunc_normal_(self.pos_embed, std=.02)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                trunc_normal_(m.bias, std=1e-6)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes)
        # self.head.weight.data.normal_(mean=0.0, std=0.01)
        self.head.weight.data.zero_()
        self.head.bias.data.zero_()

    def forward_features(self, x, return_feat=False):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        if return_feat:
            return x

        if self.classifier == 'mean':
            return x[:, 1:].mean(dim=1)
        return x[:, 0]
    
    def freeze_backbone(self):
        for name, param in self.named_parameters():
            if not 'head' in name:
                param.requires_grad = False

    def forward(self, 
                x, 
                return_feat=False):
        """ Indices contain the index of candidate w.r.t block depth, attn_heads, mlp_ratio
        """
        x = self.forward_features(x, return_feat)
        if return_feat:
            return x
        if self.pretrain:
            x = self.repr(x)
        x = self.head(x)
        return x


def scale_positional_embedding(posemb, new_posemb):
    ntok_new = new_posemb.size(1)
    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = nn.functional.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bicubic')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    return torch.cat([posemb_tok, posemb_grid], dim=1)



def vit_small_patch16_224(img_size=224, patch_size=16, embed_dim=384, num_classes=1000, depth_list=[12,13,14], num_heads_list=[3,4,6,8], mlp_ratio_list=[3,4,5], **kwargs):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_classes=num_classes,
        depth_list=depth_list,
        num_heads_list=num_heads_list,
        mlp_ratio_list=mlp_ratio_list,
        **kwargs
    )
    return model

def sub_vit_small_patch16_224(img_size=224, patch_size=16, embed_dim=384, num_classes=1000, depth_idx=0, num_heads_idx=[0,1,2,3], mlp_ratio_idx=[0,1,2,3], **kwargs):
    model = Sub_VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_classes=num_classes,
        depth_idx=depth_idx,
        num_heads_idx=num_heads_idx,
        mlp_ratio_idx=mlp_ratio_idx,
        **kwargs
    )
    return model

def vit_base_patch32_224(img_size=224, patch_size=32, embed_dim=768, num_classes=1000, depth_list=[12,13,14], num_heads_list=[3,4,6,8], mlp_ratio_list=[3,4,5], drop_rate=0.1, **kwargs):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_classes=num_classes,
        depth_list=depth_list,
        num_heads_list=num_heads_list,
        mlp_ratio_list=mlp_ratio_list,
        drop_rate=drop_rate,
        **kwargs
    )
    return model


def vit_base_patch16_224(img_size=224, patch_size=16, embed_dim=768, num_classes=1000, depth_list=[12,13,14], num_heads_list=[3,4,6,8], mlp_ratio_list=[3,4,5], drop_rate=0.1, **kwargs):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_classes=num_classes,
        depth_list=depth_list,
        num_heads_list=num_heads_list,
        mlp_ratio_list=mlp_ratio_list,
        drop_rate=drop_rate,
        **kwargs
    )
    return model


def vit_base_patch14_224(img_size=224, patch_size=14, embed_dim=768, num_classes=1000, depth_list=[12,13,14], num_heads_list=[3,4,6,8], mlp_ratio_list=[3,4,5], drop_rate=0.1, **kwargs):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_classes=num_classes,
        depth_list=depth_list,
        num_heads_list=num_heads_list,
        mlp_ratio_list=mlp_ratio_list,
        drop_rate=drop_rate,
        **kwargs
    )
    return model

def sub_vit_base_patch16_224(img_size=224, patch_size=16, embed_dim=768, num_classes=1000, depth_idx=0, num_heads_idx=[0,1,2,3], mlp_ratio_idx=[0,1,2,3], **kwargs):
    model = Sub_VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_classes=num_classes,
        depth_idx=depth_idx,
        num_heads_idx=num_heads_idx,
        mlp_ratio_idx=mlp_ratio_idx,
        **kwargs
    )
    return model

def vit_large_patch32_224(img_size=224, patch_size=32, embed_dim=1024, num_classes=1000, depth_list=[12,13,14], num_heads_list=[3,4,6,8], mlp_ratio_list=[3,4,5], drop_rate=0.1, **kwargs):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_classes=num_classes,
        depth_list=depth_list,
        num_heads_list=num_heads_list,
        mlp_ratio_list=mlp_ratio_list,
        drop_rate=drop_rate,
        **kwargs
    )
    return model


def vit_large_patch16_224(img_size=224, patch_size=16, embed_dim=1024, num_classes=1000, depth_list=[12,13,14], num_heads_list=[3,4,6,8], mlp_ratio_list=[3,4,5], drop_rate=0.1, **kwargs):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_classes=num_classes,
        depth_list=depth_list,
        num_heads_list=num_heads_list,
        mlp_ratio_list=mlp_ratio_list,
        drop_rate=drop_rate,
        **kwargs
    )
    return model


def vit_large_patch14_224(img_size=224, patch_size=14, embed_dim=1024, num_classes=1000, depth_list=[12,13,14], num_heads_list=[3,4,6,8], mlp_ratio_list=[3,4,5], drop_rate=0.1, **kwargs):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_classes=num_classes,
        depth_list=depth_list,
        num_heads_list=num_heads_list,
        mlp_ratio_list=mlp_ratio_list,
        drop_rate_list=drop_rate,
        **kwargs
    )
    return model


def vit_huge_patch32_224(img_size=224, patch_size=32, embed_dim=1280, num_classes=1000, depth_list=[12,13,14], num_heads_list=[3,4,6,8], mlp_ratio_list=[3,4,5], drop_rate=0.1, **kwargs):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_classes=num_classes,
        depth_list=depth_list,
        num_heads_list=num_heads_list,
        mlp_ratio_list=mlp_ratio_list,
        drop_rate=drop_rate,
        **kwargs
    )
    return model


def vit_huge_patch16_224(img_size=224, patch_size=16, embed_dim=1280, num_classes=1000, depth_list=[12,13,14], num_heads_list=[3,4,6,8], mlp_ratio_list=[3,4,5], drop_rate=0.1, **kwargs):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_classes=num_classes,
        depth_list=depth_list,
        num_heads_list=num_heads_list,
        mlp_ratio_list=mlp_ratio_list,
        drop_rate=drop_rate,
        **kwargs
    )
    return model


def vit_huge_patch14_224(img_size=224, patch_size=14, embed_dim=1280, num_classes=1000, depth_list=[12,13,14], num_heads_list=[3,4,6,8], mlp_ratio_list=[3,4,5], drop_rate=0.1, **kwargs):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_classes=num_classes,
        depth_list=depth_list,
        num_heads_list=num_heads_list,
        mlp_ratio_list=mlp_ratio_list,
        drop_rate=drop_rate,
        **kwargs
    )
    return model



if __name__ == '__main__':
    model = vit_small_patch16_224()
    print(model)
    input = torch.randn(1, 3, 224, 224)
    depth_list = [12,13,14]
    depth_idx = np.random.randint(3)
    attn_idx = random_choice(4, depth_list[depth_idx])
    mlp_idx = random_choice(3, depth_list[depth_idx])
    output = model(input, depth_idx, attn_idx, mlp_idx)
    print(output.shape)
    from thop import profile
    macs, params = profile(model, inputs=(input, depth_idx, attn_idx, mlp_idx))
    print("Flops: {}, Params: {}".format(macs / 1e9, params / 1e6))