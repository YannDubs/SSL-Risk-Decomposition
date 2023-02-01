import types
from torch.hub import load_state_dict_from_url
from torchvision import transforms


from hub.augmentations import get_augmentations, get_normalization
from hub.helpers import VITWrapper, get_intermediate_layers

import timm

__all__ = ["get_beit_models"]


BEIT_MODELS = {
"beit_vitB16_pt22k": "https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k.pth",
"beit_vitL16_pt22k": "https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k.pth",
"beitv2_vitB16_pt1k_ep300": "https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_300e.pth",
"beitv2_vitB16_pt1k": "https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k.pth",
"beitv2_vitL16_pt1k": "https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k.pth",
}

def get_beit_models(name, architecture, representation="cls", normalize="imagenet", is_train_transform=False):

    state_dict = load_state_dict_from_url(
        url=BEIT_MODELS[name],
        map_location="cpu",
        file_name=name
    )['model']


    encoder = timm.create_model(architecture, num_classes=0, global_pool='')

    state_dict = {k: v for k, v in state_dict.items()
                  if ("relative_position_index" not in k) and
                     ("head." not in k) and
                     ("mask_token" not in k) and
                     ("cls_pt_layers" not in k)}

    if "rel_pos_bias.relative_position_bias_table" in state_dict:
        # copies shared relative position table to each block
        rel_pos_bias = state_dict.pop("rel_pos_bias.relative_position_bias_table")
        for i in range(len(encoder.blocks)):
            state_dict[f"blocks.{i}.attn.relative_position_bias_table"] = rel_pos_bias.clone()

    missing_keys, unexpected_keys= encoder.load_state_dict(state_dict, strict=False)
    actual_missing_keys = [k for k in missing_keys if "relative_position_index" not in k]
    assert len(actual_missing_keys) == 0, f"Missing keys: {actual_missing_keys}"
    assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"

    # makes timm compatible with VITWrapper
    encoder.get_intermediate_layers = types.MethodType(get_intermediate_layers, encoder)
    encoder = VITWrapper(encoder, representation=representation)

    if is_train_transform:
        preprocessor = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(224,
                                         scale=(0.08, 1.0),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            get_normalization(mode="imagenet")])
    else:
        preprocessor = get_augmentations(interpolation=transforms.InterpolationMode.BICUBIC,
                                         normalize=normalize, pre_resize=256)


    return encoder, preprocessor