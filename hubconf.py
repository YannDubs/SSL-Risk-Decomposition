import pdb

dependencies = [ "torch", "torchvision", "timm"]

import logging as _logging
import pathlib as _pathlib

BASE_DIR = _pathlib.Path(__file__).absolute().parents[0]

def metadata_dict(drop_missing=True):
    """Returns the metadata as a dictionary of dictionaries. If `drop_missing`,
    removes both missing ??? and not applicable `None` values.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("Please install `pyaml` to use metadata_dict")

    with open(BASE_DIR/'metadata.yaml') as f:
        metadata = yaml.safe_load(f)

    if drop_missing:
        metadata = {k1: {k2: {k3:v for k3, v in d2.items()
                              if v is not None and v != "???"}
                         for k2, d2 in d.items()}
                    for k1, d in metadata.items()}
    return metadata

def metadata_df(is_multiindex=False, is_lower=True, **kwargs):
    """Returns the metadata as a pandas dataframe. If `is_multiindex` then returns then keeps
    the first level of keys as a multi index of columns. If `is_lower` then lower cases all the
    strings in the dataframe.
    """
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        raise ImportError("Please install `pandas` to use metadata_df")

    metadata_flatten = {k1: {(k2, k3): v
                             for k2, d2 in d.items()
                             for k3, v in d2.items()
                             }
                        for k1, d in metadata_dict(**kwargs).items()}
    df = pd.DataFrame.from_dict(metadata_flatten, orient="index")

    if not is_multiindex:
        df = df.droplevel(0, axis=1)

    if is_lower:
        df.applymap(lambda s: s.lower() if isinstance(s,str) else s)
        if df.columns.nlevels > 1:
            df.columns = pd.MultiIndex.from_tuples(tuple(c.lower() if isinstance(c, str) else c
                                                         for c in t)
                                                   for t in df.columns )
        else:
            df.columns = [c.lower() if isinstance(c, str) else c
                          for c in df.columns ]
        df.index = [i.lower() if isinstance(i, str) else i
                    for i in df.index if isinstance(i, str)]

    cols_to_types = _metadata_cols_to_types()
    assert set(cols_to_types.keys()) == set(df.columns.get_level_values(-1))
    for c, dtype in cols_to_types.items():
        i_col = df.columns.get_level_values(-1) == c
        if isinstance(dtype, pd.Int64Dtype):
            # for int needs to first convert to float if nan
            df[df.columns[i_col]] = pd.to_numeric(df.iloc[:, i_col].squeeze(), errors='coerce').astype('Int64').to_frame()
        else:
            df[df.columns[i_col]] = df.iloc[:, i_col].astype(dtype)

    return df



##### BYOL #####
# pretrained models are from https://github.com/deepmind/deepmind-research/tree/master/byol
# converted to pytorch using https://github.com/ajtejankar/byol-convert

try:
    from hub.byol import get_byol_models as _get_byol_models

    def byol_rn50_augCrop(**kwargs):
        return _get_byol_models('byol_rn50_augCrop', "res50x1_crop_only", architecture='resnet50', **kwargs)

    def byol_rn50_augCropBlur(**kwargs):
        return _get_byol_models('byol_rn50_augCropBlur', "res50x1_crop_and_blur_only", architecture='resnet50', **kwargs)

    def byol_rn50_augCropColor(**kwargs):
        return _get_byol_models('byol_rn50_augCropColor', "res50x1_crop_and_color_only", architecture='resnet50', **kwargs)

    def byol_rn50_augNocolor(**kwargs):
        return _get_byol_models('byol_rn50_augNocolor', "res50x1_no_color", architecture='resnet50', **kwargs)

    def byol_rn50_augNogray(**kwargs):
        return _get_byol_models('byol_rn50_augNogray', "res50x1_no_grayscale", architecture='resnet50', **kwargs)

    def byol_rn50_bs64(**kwargs):
        return _get_byol_models('byol_rn50_bs64', "res50x1_batchsize_64", architecture='resnet50', **kwargs)

    def byol_rn50_bs128(**kwargs):
        return _get_byol_models('byol_rn50_bs128', "res50x1_batchsize_128", architecture='resnet50', **kwargs)

    def byol_rn50_bs256(**kwargs):
        return _get_byol_models('byol_rn50_bs256', "res50x1_batchsize_256", architecture='resnet50', **kwargs)

    def byol_rn50_bs512(**kwargs):
        return _get_byol_models('byol_rn50_bs512', "res50x1_batchsize_512", architecture='resnet50', **kwargs)

    def byol_rn50_bs1024(**kwargs):
        return _get_byol_models('byol_rn50_bs1024', "res50x1_batchsize_1024", architecture='resnet50', **kwargs)

    def byol_rn50_bs2048(**kwargs):
        return _get_byol_models('byol_rn50_bs2048', "res50x1_batchsize_2048", architecture='resnet50', **kwargs)

    def byol_rn50_bs4096(**kwargs):
        return _get_byol_models('byol_rn50_bs4096', "pretrain_res50x1", architecture='resnet50', **kwargs)

except ImportError as e:
    _logging.warning(f"BYOL models not available because of the following import error: \n {e}")

##### VISSL #####
# pretrained models are from https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md

try:
    from hub.vissl import get_vissl_models as _get_vissl_models

    def barlow_rn50(**kwargs):
        return _get_vissl_models('barlow_rn50', architecture='resnet50', **kwargs)

    def barlow_rn50_ep300(**kwargs):
        return _get_vissl_models('barlow_rn50_ep300', architecture='resnet50', **kwargs)

    def mocov2_rn50_vissl(**kwargs):
        return _get_vissl_models('mocov2_rn50_vissl', architecture='resnet50', **kwargs)

    def rotnet_rn50_in1k(**kwargs):
        return _get_vissl_models('rotnet_rn50_in1k', architecture='resnet50', **kwargs)

    def rotnet_rn50_in22k(**kwargs):
        return _get_vissl_models('rotnet_rn50_in22k', architecture='resnet50', **kwargs)

    def simclr_rn50(**kwargs):
        return _get_vissl_models('simclr_rn50', architecture='resnet50', **kwargs)

    def simclr_rn50_ep200(**kwargs):
        return _get_vissl_models('simclr_rn50_ep200', architecture='resnet50', **kwargs)

    def simclr_rn50_ep400(**kwargs):
        return _get_vissl_models('simclr_rn50_ep400', architecture='resnet50', **kwargs)

    def simclr_rn50_ep800(**kwargs):
        return _get_vissl_models('simclr_rn50_ep800', architecture='resnet50', **kwargs)

    def simclr_rn50_bs4096_ep100(**kwargs):
        return _get_vissl_models('simclr_rn50_bs4096_ep100', architecture='resnet50', **kwargs)

    def simclr_rn50w2(**kwargs):
        return _get_vissl_models('simclr_rn50w2', architecture='resnet50', width_multiplier=2, **kwargs)

    def simclr_rn50w2_ep100(**kwargs):
        return _get_vissl_models('simclr_rn50w2_ep100', architecture='resnet50', width_multiplier=2, **kwargs)

    def simclr_rn50w4(**kwargs):
        return _get_vissl_models('simclr_rn50w4', architecture='resnet50', width_multiplier=4, **kwargs)

    def simclr_rn101(**kwargs):
        return _get_vissl_models('simclr_rn101', architecture='resnet101', **kwargs)

    def simclr_rn101_ep100(**kwargs):
        return _get_vissl_models('simclr_rn101_ep100', architecture='resnet101', **kwargs)

    def jigsaw_rn50_in22k(**kwargs):
        return _get_vissl_models('jigsaw_rn50_in22k', architecture='resnet50', **kwargs)

    def jigsaw_rn50(**kwargs):
        return _get_vissl_models('jigsaw_rn50', architecture='resnet50', **kwargs)

    def clusterfit_rn50(**kwargs):
        return _get_vissl_models('clusterfit_rn50', architecture='resnet50', **kwargs)

    def npid_rn50(**kwargs):
        return _get_vissl_models('npid_rn50', architecture='resnet50', **kwargs)

    def npidpp_rn50(**kwargs):
        return _get_vissl_models('npidpp_rn50', architecture='resnet50', **kwargs)

    # not working until https://github.com/facebookresearch/vissl/issues/516
    def npidpp_rn50w2(**kwargs):
        return _get_vissl_models('npidpp_rn50w2', architecture='resnet50', width_multiplier=2, **kwargs)

    def pirl_rn50(**kwargs):
        return _get_vissl_models('pirl_rn50', architecture='resnet50', **kwargs)

    def pirl_rn50_ep200(**kwargs):
        return _get_vissl_models('pirl_rn50_ep200', architecture='resnet50', **kwargs)

    def pirl_rn50_headMLP(**kwargs):
        return _get_vissl_models('pirl_rn50_headMLP', architecture='resnet50', **kwargs)

    def pirl_rn50_ep200_headMLP(**kwargs):
        return _get_vissl_models('pirl_rn50_ep200_headMLP', architecture='resnet50', **kwargs)

    def pirl_rn50w2(**kwargs):
        return _get_vissl_models('pirl_rn50w2', architecture='resnet50', width_multiplier=2, **kwargs)

    def pirl_rn50w2_headMLP(**kwargs):
        return _get_vissl_models('pirl_rn50w2_headMLP', architecture='resnet50', width_multiplier=2, **kwargs)

except ImportError as e:
    _logging.warning(f"VISSL models not available because of the following import error: \n {e}")

##### VICReg #####
# pretrained models are from https://github.com/facebookresearch/vicreg
try:
    from hub.vicreg import get_vicreg_models as _get_vicreg_models

    def vicreg_rn50(**kwargs):
        return _get_vicreg_models('resnet50', **kwargs)

    def vicreg_rn50w2(**kwargs):
        return _get_vicreg_models('resnet50x2', **kwargs)

except ImportError as e:
    _logging.warning(f"VICReg models not available because of the following import error: \n {e}")

##### VICRegL #####
# pretrained models are from https://github.com/facebookresearch/vicregl
try:
    from hub.vicregl import get_vicregl_models as _get_vicregl_models

    def vicregl_rn50_alpha09(**kwargs):
        return _get_vicregl_models('resnet50_alpha0p9', **kwargs)

    def vicregl_rn50_alpha075(**kwargs):
        return _get_vicregl_models('resnet50_alpha0p75', **kwargs)

    def vicregl_convnexts_alpha09(**kwargs):
        return _get_vicregl_models('convnext_small_alpha0p9', **kwargs)

    def vicregl_convnexts_alpha075(**kwargs):
        return _get_vicregl_models('convnext_small_alpha0p75', **kwargs)

    def vicregl_convnextb_alpha09(**kwargs):
        return _get_vicregl_models('convnext_base_alpha0p9', **kwargs)

    def vicregl_convnextb_alpha075(**kwargs):
        return _get_vicregl_models('convnext_base_alpha0p75', **kwargs)

    def vicregl_convnextxl_alpha075(**kwargs):
        return _get_vicregl_models('convnext_xlarge_alpha0p75', **kwargs)

except ImportError as e:
    _logging.warning(f"VICRegL models not available because of the following import error: \n {e}")

##### SwAV #####
# pretrained models are from https://github.com/facebookresearch/swav/

try:
    from hub.swav import get_swav_models as _get_swav_models

    def swav_rn50(**kwargs):
        return _get_swav_models('swav_rn50', "resnet50", architecture='resnet50', **kwargs)

    def swav_rn50_ep100(**kwargs):
        return _get_swav_models('swav_rn50_ep100', "resnet50_ep100", architecture='resnet50', **kwargs)

    def swav_rn50_ep200(**kwargs):
        return _get_swav_models('swav_rn50_ep200', "resnet50_ep200", architecture='resnet50', **kwargs)

    def swav_rn50_ep200_bs256(**kwargs):
        return _get_swav_models('swav_rn50_ep200_bs256', "resnet50_ep200_bs256", architecture='resnet50', **kwargs)

    def swav_rn50_ep400(**kwargs):
        return _get_swav_models('swav_rn50_ep400', "resnet50_ep400", architecture='resnet50', **kwargs)

    def swav_rn50_ep400_2x224(**kwargs):
        return _get_swav_models('swav_rn50_ep400_2x224', "resnet50_ep400_2x224", architecture='resnet50', **kwargs)

    def swav_rn50_ep400_bs256(**kwargs):
        return _get_swav_models('swav_rn50_ep400_bs256', "resnet50_ep400_bs256", architecture='resnet50', **kwargs)

    def swav_rn50w2(**kwargs):
        return _get_swav_models('swav_rn50w2', "resnet50w2", architecture='resnet50w2', **kwargs)

    def swav_rn50w4(**kwargs):
        return _get_swav_models('swav_rn50w4', "resnet50w4", architecture='resnet50w4', **kwargs)

    def swav_rn50w5(**kwargs):
        return _get_swav_models('swav_rn50w5', "resnet50w5", architecture='resnet50w5', **kwargs)

    def dc2_rn50_ep400_2x224(**kwargs):
        return _get_swav_models('dc2_rn50_ep400_2x224', 'dc2_rn50_ep400_2x224', architecture='resnet50', **kwargs)

    def dc2_rn50_ep400_2x160_4x96(**kwargs):
        return _get_swav_models('dc2_rn50_ep400_2x160_4x96', 'dc2_rn50_ep400_2x160_4x96', architecture='resnet50', **kwargs)

    def dc2_rn50_ep800_2x224_6x96(**kwargs):
        return _get_swav_models('dc2_rn50_ep800_2x224_6x96',  'dc2_rn50_ep800_2x224_6x96', architecture='resnet50', **kwargs)

    def selav2_rn50_ep400_2x224(**kwargs):
        return _get_swav_models('selav2_rn50_ep400_2x224', 'selav2_rn50_ep400_2x224', architecture='resnet50', **kwargs)

    def selav2_rn50_ep400_2x160_4x96(**kwargs):
        return _get_swav_models('selav2_rn50_ep400_2x160_4x96', 'selav2_rn50_ep400_2x160_4x96', architecture='resnet50', **kwargs)

except ImportError as e:
    _logging.warning(f"SwAV models not available because of the following import error: \n {e}")

##### SimSiam #####
# pretrained models are from https://github.com/facebookresearch/simsiam

try:
    from hub.simsiam import get_simsiam_models as _get_simsiam_models

    def simsiam_rn50_bs512_ep100(**kwargs):
        return _get_simsiam_models('simsiam_rn50_bs512_ep100', **kwargs)

    def simsiam_rn50_bs256_ep100(**kwargs):
        return _get_simsiam_models('simsiam_rn50_bs256_ep100', **kwargs)

except ImportError as e:
    _logging.warning(f"SimSiam models not available because of the following import error: \n {e}")



##### ISSL #####
# pretrained models are from https://github.com/YannDubs/Invariant-Self-Supervised-Learning

try:
    from hub.issl import get_issl_models as _get_issl_models

    def dissl_resnet50_dNone_e100_m2(**kwargs):
        return _get_issl_models('dissl_resnet50_dNone_e100_m2', **kwargs)

    def dissl_resnet50_d8192_e100_m2(**kwargs):
        return _get_issl_models('dissl_resnet50_d8192_e100_m2', **kwargs)

    def dissl_resnet50_dNone_e400_m2(**kwargs):
        return _get_issl_models('dissl_resnet50_dNone_e400_m2', **kwargs)

    def dissl_resnet50_dNone_e400_m6(**kwargs):
        return _get_issl_models('dissl_resnet50_dNone_e400_m6', **kwargs)

    def dissl_resnet50_d8192_e400_m6(**kwargs):
        return _get_issl_models('dissl_resnet50_d8192_e400_m6', **kwargs)

    def dissl_resnet50_d8192_e800_m8(**kwargs):
        return _get_issl_models('dissl_resnet50_d8192_e800_m8', **kwargs)

except ImportError as e:
    _logging.warning(f"ISSL models not available because of the following import error: \n {e}")

##### Riskdec #####
# pretrained models available on our directory https://github.com/YannDubs/SSL-Risk-Decomposition

try:
    from hub.riskdec import get_riskdec_models as _get_riskdec_models

    def dissl_resnet50_dNone_e100_m2_augLarge(**kwargs):
        return _get_riskdec_models('dissl_resnet50_dNone_e100_m2_augLarge', **kwargs)

    def dissl_resnet50_dNone_e100_m2_augSmall(**kwargs):
        return _get_riskdec_models('dissl_resnet50_dNone_e100_m2_augSmall', **kwargs)

    def dissl_resnet50_dNone_e100_m2_headTLinSLin(**kwargs):
        return _get_riskdec_models('dissl_resnet50_dNone_e100_m2_headTLinSLin', **kwargs)

    def dissl_resnet50_dNone_e100_m2_headTMlpSMlp(**kwargs):
        return _get_riskdec_models('dissl_resnet50_dNone_e100_m2_headTMlpSMlp', **kwargs)

    def dissl_resnet50_d4096_e100_m2(**kwargs):
        return _get_riskdec_models('dissl_resnet50_d4096_e100_m2', dim=4096, **kwargs)

    def simclr_resnet50_dNone_e100_m2(**kwargs):
        return _get_riskdec_models('simclr_resnet50_dNone_e100_m2', **kwargs)

    def simclr_resnet50_dNone_e100_m2_data010(**kwargs):
        return _get_riskdec_models('simclr_resnet50_dNone_e100_m2_data010', **kwargs)

    def simclr_resnet50_dNone_e100_m2_data030(**kwargs):
        return _get_riskdec_models('simclr_resnet50_dNone_e100_m2_data030', **kwargs)

    def simclr_resnet50_dNone_e100_m2_headTLinSLin(**kwargs):
        return _get_riskdec_models('simclr_resnet50_dNone_e100_m2_headTLinSLin', **kwargs)

    def simclr_resnet50_dNone_e100_m2_headTMlpSLin(**kwargs):
        return _get_riskdec_models('simclr_resnet50_dNone_e100_m2_headTMlpSLin', **kwargs)

    def simclr_resnet50_dNone_e100_m2_headTMlpSMlp(**kwargs):
        return _get_riskdec_models('simclr_resnet50_dNone_e100_m2_headTMlpSMlp', **kwargs)

    def simclr_resnet50_dNone_e100_m2_headTNoneSNone(**kwargs):
        return _get_riskdec_models('simclr_resnet50_dNone_e100_m2_headTNoneSNone', **kwargs)

    def simclr_resnet50_d8192_e100_m2(**kwargs):
        return _get_riskdec_models('simclr_resnet50_d8192_e100_m2', dim=8192, **kwargs)

    def speccl_resnet50_bs384_ep100(**kwargs):
        return _get_riskdec_models('speccl_resnet50_bs384_ep100', is_speccl=True, **kwargs)

except ImportError as e:
    _logging.warning(f"RiskDec models not available because of the following import error: \n {e}")


##### Lossyless #####
# pretrained models are from https://github.com/YannDubs/lossyless

try:
    from hub.lossyless import get_lossyless_models as _get_lossyless_models

    def lossyless_vitb32_b001(**kwargs):
        return _get_lossyless_models('clip_compressor_b001', **kwargs)

    def lossyless_vitb32_b005(**kwargs):
        return _get_lossyless_models('clip_compressor_b005', **kwargs)

    def lossyless_vitb32_b01(**kwargs):
        return _get_lossyless_models('clip_compressor_b01', **kwargs)

except ImportError as e:
    _logging.warning(f"Lossyless models not available because of the following import error: \n {e}")


##### CLIP #####
# pretrained models are from https://github.com/openai/CLIP

try:
    from hub.clip import get_clip_models as _get_clip_models

    def clip_rn50(**kwargs):
        return _get_clip_models('RN50', **kwargs)

    def clip_rn50x4(**kwargs):
        return _get_clip_models('RN50x4', **kwargs)

    def clip_rn50x16(**kwargs):
        return _get_clip_models('RN50x16', **kwargs)

    def clip_rn50x64(**kwargs):
        return _get_clip_models('RN50x64', **kwargs)

    def clip_rn101(**kwargs):
        return _get_clip_models('RN101', **kwargs)

    def clip_vitB16(**kwargs):
        return _get_clip_models('ViT-B/16', **kwargs)

    def clip_vitB32(**kwargs):
        return _get_clip_models('ViT-B/32', **kwargs)

    def clip_vitL14(**kwargs):
        return _get_clip_models('ViT-L/14', **kwargs)

    def clip_vitL14_px336(**kwargs):
        return _get_clip_models('ViT-L/14@336px', **kwargs)

except ImportError as e:
    _logging.warning(f"CLIP models not available because of the following import error: \n {e}")

##### DINO #####
# pretrained models are from https://github.com/facebookresearch/dino

try:
    from hub.dino import get_dino_models as _get_dino_models

    def dino_rn50(**kwargs):
        return _get_dino_models("dino_resnet50", family="resnet", **kwargs)

    def dino_vitS16_last(**kwargs):
        return _get_dino_models("dino_vits16", **kwargs)

    def dino_vitS8_last(**kwargs):
        return _get_dino_models("dino_vits8", **kwargs)

    def dino_vitB16_last(**kwargs):
        return _get_dino_models("dino_vitb16", **kwargs)

    def dino_vitB8_last(**kwargs):
        return _get_dino_models("dino_vitb8", **kwargs)

    def dino_vitS16(**kwargs):
        return _get_dino_models("dino_vits16", representation="4xcls", **kwargs)

    def dino_vitB16(**kwargs):
        return _get_dino_models("dino_vitb16", representation="cls+avg", **kwargs)

    def dino_vitB8(**kwargs):
        return _get_dino_models("dino_vitb8", representation="cls+avg", **kwargs)

    def dino_vitS16_extractB(**kwargs):
        return _get_dino_models("dino_vits16", representation="cls+avg", **kwargs)

    def dino_vitB16_extractS(**kwargs):
        return _get_dino_models("dino_vitb16", representation="4xcls", **kwargs)

except ImportError as e:
    _logging.warning(f"DINO models not available because of the following import error: \n {e}")


##### IBOT #####
# pretrained models are from https://github.com/bytedance/ibot

try:
    from hub.ibot import get_ibot_models as _get_ibot_models

    def ibot_vitB16(**kwargs):
        return _get_ibot_models("ibot_vitB16", "vitb16", 'vit_base_patch16_224', **kwargs)

    def ibot_vitS16(**kwargs):
        return _get_ibot_models("ibot_vitS16", "vits16", 'vit_small_patch16_224', **kwargs)

    def ibot_vitL16(**kwargs):
        return _get_ibot_models("ibot_vitL16", "vitl16", 'vit_large_patch16_224', **kwargs)

    def ibot_vitB16_extractB(**kwargs):
        return _get_ibot_models("ibot_vitB16", "vitb16", 'vit_base_patch16_224', representation="cls+avg", **kwargs)

    def ibot_vitS16_extractS(**kwargs):
        return _get_ibot_models("ibot_vitS16", "vits16", 'vit_small_patch16_224', representation="4xcls", **kwargs)

except ImportError as e:
    _logging.warning(f"IBOT models not available because of the following import error: \n {e}")


##### MUGS #####
# pretrained models are from https://github.com/sail-sg/mugs

try:
    from hub.mugs import get_mugs_models as _get_mugs_models

    def mugs_vits16_ep100(**kwargs):
        return _get_mugs_models("mugs_vits16_ep100", 'vit_small_patch16_224', **kwargs)

    def mugs_vits16_ep300(**kwargs):
        return _get_mugs_models("mugs_vits16_ep300", 'vit_small_patch16_224', **kwargs)

    def mugs_vits16_ep800(**kwargs):
        return _get_mugs_models("mugs_vits16_ep800", 'vit_small_patch16_224', **kwargs)

    def mugs_vitb16_ep400(**kwargs):
        return _get_mugs_models("mugs_vitb16_ep400", 'vit_base_patch16_224', **kwargs)

    def mugs_vitl16_ep250(**kwargs):
        return _get_mugs_models("mugs_vitl16_ep250", 'vit_large_patch16_224', **kwargs)

    def mugs_vits16_ep800_extractS(**kwargs):
        return _get_mugs_models("mugs_vits16_ep800", 'vit_small_patch16_224', representation="4xcls", **kwargs)

    def mugs_vitb16_ep400_extractB(**kwargs):
        return _get_mugs_models("mugs_vitb16_ep400", 'vit_base_patch16_224', representation="cls+avg", **kwargs)

except ImportError as e:
    _logging.warning(f"MUGS models not available because of the following import error: \n {e}")

##### MAE #####
# pretrained models are from https://github.com/facebookresearch/mae

try:
    from hub.mae import get_mae_models as _get_mae_models

    def mae_vitB16(**kwargs):
        return _get_mae_models("mae_vitB16", 'vit_base_patch16_224', **kwargs)

    def mae_vitL16(**kwargs):
        return _get_mae_models("mae_vitL16", 'vit_large_patch16_224', **kwargs)

    def mae_vitH14(**kwargs):
        return _get_mae_models("mae_vitH14", 'vit_huge_patch14_224', **kwargs)

except ImportError as e:
    _logging.warning(f"MAE models not available because of the following import error: \n {e}")


##### MSN #####
# pretrained models are from https://github.com/facebookresearch/msn

try:
    from hub.msn import get_msn_models as _get_msn_models

    def msn_vits16_ep800(**kwargs):
        return _get_msn_models("msn_vits16_ep800", 'vit_small_patch16_224', **kwargs)

    def msn_vitb16_ep600(**kwargs):
        return _get_msn_models("msn_vitb16_ep600", 'vit_base_patch16_224', **kwargs)

    def msn_vitb4_ep300(**kwargs):
        return _get_msn_models("msn_vitb4_ep300", 'vit_base_patch4_224', **kwargs)

    def msn_vitl16_ep600(**kwargs):
        return _get_msn_models("msn_vitl16_ep600", 'vit_large_patch16_224', **kwargs)

    def msn_vitl7_ep200(**kwargs):
        return _get_msn_models("msn_vitl7_ep200", 'vit_large_patch7_224', **kwargs)

except ImportError as e:
    _logging.warning(f"MSN models not available because of the following import error: \n {e}")

### MOCOV3 ###
# pretrained models are from https://github.com/facebookresearch/moco-v3

try:
    from hub.mocov3 import get_mocov3_models as _get_mocov3_models

    def mocov3_rn50_ep100(**kwargs):
        return _get_mocov3_models("mocov3_rn50_ep100", 'resnet50', **kwargs)

    def mocov3_rn50_ep300(**kwargs):
        return _get_mocov3_models("mocov3_rn50_ep300", 'resnet50', **kwargs)

    def mocov3_rn50_ep1000(**kwargs):
        return _get_mocov3_models("mocov3_rn50_ep1000", 'resnet50', **kwargs)

    def mocov3_vitS_ep300(**kwargs):
        return _get_mocov3_models("mocov3_vitS_ep300", 'vit_small_patch16_224', **kwargs)

    def mocov3_vitB_ep300(**kwargs):
        return _get_mocov3_models("mocov3_vitB_ep300", 'vit_base_patch16_224', **kwargs)

except ImportError as e:
    _logging.warning(f"MOCOV3 models not available because of the following import error: \n {e}")

### MOCO ###
# pretrained models are from https://github.com/facebookresearch/moco

try:
    from hub.moco import get_moco_models as get_moco_models

    def mocov1_rn50_ep200(**kwargs):
        return get_moco_models("mocov1_rn50_ep200", **kwargs)

    def mocov2_rn50_ep200(**kwargs):
        return get_moco_models("mocov2_rn50_ep200", **kwargs)

    def mocov2_rn50_ep800(**kwargs):
        return get_moco_models("mocov2_rn50_ep800", **kwargs)

except ImportError as e:
    _logging.warning(f"MOCO models not available because of the following import error: \n {e}")

### PYCONTRAST ###
# pretrained models are from https://github.com/HobbitLong/PyContrast
try:
    from hub.pycontrast import get_pycontrast_models as get_pycontrast_models

    def infomin_rn50_200ep(**kwargs):
        return get_pycontrast_models("infomin_rn50_200ep", **kwargs)

    def infomin_rn50_800ep(**kwargs):
        return get_pycontrast_models("infomin_rn50_800ep", **kwargs)

except ImportError as e:
    _logging.warning(f"Pycontrast models not available because of the following import error: \n {e}")


### MMSelfSup ###
# pretrained models are from https://github.com/open-mmlab/mmselfsup/blob/master/docs/en/model_zoo.md
try:
    from hub.mmselfsup import get_mmselfsup_models as get_mmselfsup_models

    def relativeloc_rn50_70ep_mmselfsup(**kwargs):
        return get_mmselfsup_models("relativeloc_rn50_70ep_mmselfsup", **kwargs)

    def odc_rn50_440ep_mmselfsup(**kwargs):
        return get_mmselfsup_models("odc_rn50_440ep_mmselfsup", **kwargs)

    def densecl_rn50_200ep_mmselfsup(**kwargs):
        return get_mmselfsup_models("densecl_rn50_200ep_mmselfsup", **kwargs)

    def simsiam_rn50_bs256_ep200_mmselfsup(**kwargs):
        return get_mmselfsup_models("simsiam_rn50_bs256_ep200_mmselfsup", **kwargs)

    def simclr_rn50_bs256_ep200_mmselfsup(**kwargs):
        return get_mmselfsup_models("simclr_rn50_bs256_ep200_mmselfsup", **kwargs)

    def deepcluster_rn50_bs512_ep200_mmselfsup(**kwargs):
        return get_mmselfsup_models("deepcluster_rn50_bs512_ep200_mmselfsup", **kwargs)

except ImportError as e:
    _logging.warning(f"MMSelfSup models not available because of the following import error: \n {e}")

### BEIT ###
# taken from

try:
    from hub.beit import get_beit_models as _get_beit_models

    def beit_vitB16_pt22k(**kwargs):
        return _get_beit_models("beit_vitB16_pt22k", 'beit_base_patch16_224', normalize="half", **kwargs)

    def beit_vitL16_pt22k(**kwargs):
        return _get_beit_models("beit_vitL16_pt22k", 'beit_large_patch16_224', normalize="half", **kwargs)

    def beitv2_vitB16_pt1k_ep300(**kwargs):
        return _get_beit_models("beitv2_vitB16_pt1k_ep300", 'beit_base_patch16_224', **kwargs)

    def beitv2_vitB16_pt1k(**kwargs):
        return _get_beit_models("beitv2_vitB16_pt1k", 'beit_base_patch16_224', **kwargs)

    def beitv2_vitL16_pt1k(**kwargs):
        return _get_beit_models("beitv2_vitL16_pt1k", 'beit_large_patch16_224', **kwargs)

    def beitv2_vitB16_pt1k_extractB(**kwargs):
        return _get_beit_models("beitv2_vitB16_pt1k", 'beit_base_patch16_224', representation="cls+avg", **kwargs)

except ImportError as e:
    _logging.warning(f"BEIT models not available because of the following import error: \n {e}")

### TIMM ###
try:
    from hub.timm import get_timm_models as _get_timm_models

    def sup_vitB8(**kwargs):
        # TODO newer version of timm 'vit_base_patch8_224.augreg_in21k_ft_in1k'
        return _get_timm_models('vit_base_patch8_224', **kwargs)

    def sup_vitB8_dino(**kwargs):
        # TODO newer version of timm 'vit_base_patch8_224.augreg_in21k_ft_in1k'
        return _get_timm_models('vit_base_patch8_224', representation_vit='cls+avg', **kwargs)

    def sup_vitB16(**kwargs):
        # TODO newer version of timm 'vit_small_patch16_224.augreg_in21k_ft_in1k'
        return _get_timm_models('vit_base_patch16_224', **kwargs)

    def sup_vitB16_dino(**kwargs):
        # TODO newer version of timm 'vit_small_patch16_224.augreg_in21k_ft_in1k'
        return _get_timm_models('vit_base_patch16_224', representation_vit="cls+avg", **kwargs)

    def sup_vitB16_dino_extractS(**kwargs):
        # TODO newer version of timm 'vit_small_patch16_224.augreg_in21k_ft_in1k'
        return _get_timm_models('vit_base_patch16_224', representation_vit="4xcls", **kwargs)

    def sup_vitB32(**kwargs):
        # TODO newer version of timm 'vit_base_patch32_224.augreg_in21k_ft_in1k'
        return _get_timm_models('vit_base_patch32_224', **kwargs)

    # no pretrained weights for those
    # def sup_vitH14(**kwargs):
    #     return _get_timm_models('vit_huge_patch14_224', **kwargs)
    #
    # def sup_vitL14(**kwargs):
    #     return _get_timm_models('vit_large_patch14_224', **kwargs)

    def sup_vitL16(**kwargs):
        # TODO newer version of timm 'vit_large_patch16_224.augreg_in21k_ft_in1k'
        return _get_timm_models('vit_large_patch16_224', **kwargs)

    def sup_vitS16(**kwargs):
        # TODO newer version of timm 'vit_small_patch16_224.augreg_in21k_ft_in1k'
        return _get_timm_models('vit_small_patch16_224', **kwargs)

    def sup_vitS16_dino(**kwargs):
        # TODO newer version of timm 'vit_small_patch16_224.augreg_in21k_ft_in1k'
        return _get_timm_models('vit_small_patch16_224', representation_vit="4xcls", **kwargs)

    def sup_vitS16_dino_extractB(**kwargs):
        # TODO newer version of timm 'vit_small_patch16_224.augreg_in21k_ft_in1k'
        return _get_timm_models('vit_small_patch16_224', representation_vit="cls+avg", **kwargs)

    def sup_convnextS(**kwargs):
        # TODO newer version of timm 'convnext_base.fb_in1k'
        return _get_timm_models('convnext_small', **kwargs)

    def sup_convnextB(**kwargs):
        # TODO newer version of timm 'convnext_small.fb_in1k'
        return _get_timm_models('convnext_base', **kwargs)

except ImportError as e:
    _logging.warning(f"TIMM models not available because of the following import error: \n {e}")

### TORCHVISION ###

try:
    from hub.torchvision import get_torchvision_models as _get_torchvision_models

    def sup_rn50(**kwargs):
        return _get_torchvision_models("resnet50", **kwargs)

    def sup_rn50w2(**kwargs):
        return _get_torchvision_models("wide_resnet50_2", **kwargs)

    def sup_rn101(**kwargs):
        return _get_torchvision_models("resnet101", **kwargs)

except ImportError as e:
    _logging.warning(f"Torchvision models not available because of the following import error: \n {e}")


### INITIALIZATION ###

try:
    from hub.initialized import get_initialized_models as _get_initialized_models

    def init_rn50(**kwargs):
        return _get_initialized_models("resnet50", **kwargs)

    def init_rn50_d4096(**kwargs):
        return _get_initialized_models("resnet50", new_dim=4096, **kwargs)

    def init_rn50_d8192(**kwargs):
        return _get_initialized_models("resnet50", new_dim=8192, **kwargs)

    def init_rn50_d1024(**kwargs):
        return _get_initialized_models("resnet50", new_dim=1024, **kwargs)

    def init_rn50_d512(**kwargs):
        return _get_initialized_models("resnet50", new_dim=512, **kwargs)

    def init_rn101(**kwargs):
        return _get_initialized_models("resnet101", **kwargs)

    def init_rn50w2(**kwargs):
        return _get_initialized_models("resnet50w2", **kwargs)

    def init_vitB8(**kwargs):
        return _get_initialized_models('vit_base_patch8_224', **kwargs)

    def init_vitB8_dino(**kwargs):
        return _get_initialized_models('vit_base_patch8_224', representation_vit='cls+avg', **kwargs)

    def init_vitB16(**kwargs):
        return _get_initialized_models('vit_base_patch16_224', **kwargs)

    def init_vitB16_dino(**kwargs):
        return _get_initialized_models('vit_base_patch16_224', representation_vit="cls+avg", **kwargs)

    def init_vitB16_dino_extractS(**kwargs):
        return _get_initialized_models('vit_base_patch16_224', representation_vit="4xcls", **kwargs)

    def init_vitB32(**kwargs):
        return _get_initialized_models('vit_base_patch32_224', **kwargs)

    def init_vitL16(**kwargs):
        return _get_initialized_models('vit_large_patch16_224', **kwargs)

    def init_vitS16(**kwargs):
        return _get_initialized_models('vit_small_patch16_224', **kwargs)

    def init_vitS16_dino(**kwargs):
        return _get_initialized_models('vit_small_patch16_224', representation_vit="4xcls", **kwargs)

    def init_vitS16_dino_extractB(**kwargs):
        return _get_initialized_models('vit_small_patch16_224', representation_vit="cls+avg", **kwargs)

except ImportError as e:
    _logging.warning(f"Initialized models not available because of the following import error: \n {e}")


def _metadata_cols_to_types():
    import pandas as pd
    return {'objective': pd.StringDtype(),  # ssl objective
            'ssl_mode': pd.StringDtype(),  # coarse type of ssl, eg generative vs contrastive
            'version': pd.Int64Dtype(),  # version of ssl objective
            'is_stopgrad': "boolean",  # are you stopping gradients
            'is_ema': "boolean",  # are you using some exponential moving average
            'other': pd.StringDtype(),  # additional parameters that you should use to distinguish models
            'architecture_exact': pd.StringDtype(),  # exact architercture (flags minor diff)
            'architecture': pd.StringDtype(),  # typical architecture: eg resnet50
            'family': pd.StringDtype(),  # coarse type of arch (eg cnn/vit)
            'patch_size': pd.Int64Dtype(),  # patch size if applicable
            'n_parameters': pd.Int64Dtype(),  # number of param
            'z_dim': pd.Int64Dtype(),  # dim of rep
            'z_layer': pd.StringDtype(),  # which representation uses for extraction
            'epochs': pd.Int64Dtype(),  # n train epochs
            'batch_size': pd.Int64Dtype(),
            'optimizer': pd.StringDtype(),
            'learning_rate': "float64",
            'weight_decay': "float64",
            'scheduler': pd.StringDtype(),  # learning rate scheduler
            'pretraining_data': pd.StringDtype(),
            'img_size': pd.Int64Dtype(),  # smallest size of image side to input
            'views': pd.StringDtype(),  # input crop during training
            'is_aug_invariant': "boolean",  # whether the method tries to be augmentation invariant (somewhat subjective_
            'augmentations': "object",  # list of augmentations
            'where': pd.StringDtype(),  # where pretraining model can be found
            'notes': pd.StringDtype(),  # additional notes
            'month': pd.Int64Dtype(),  # publication month
            'year': pd.Int64Dtype(),  # publication year
            'license': pd.StringDtype(),  # license of pretraining weights / code
            'is_official': "boolean",  # whether official pretraining weights
            'is_industry': "boolean",  # whether the hyper parameters were tuned in industry / big tech
            'n_pus': pd.Int64Dtype(),  # number of processors used for training
            'pu_type': pd.StringDtype(),  # type of processors used for training
            'time_hours': "float64",  # total training time
            'n_classes': pd.Int64Dtype(),  # number of classes trying to predict if applicable
            'pred_dim': pd.Int64Dtype(),  # output of projection head if applicable
            'projection_hid_width': pd.Int64Dtype(),  # width of projection head
            'projection_hid_depth': pd.Int64Dtype(),  # depth of projection head
            'projection1_arch': pd.StringDtype(),  # projection architecture for large siamese
            'projection2_arch': pd.StringDtype(),  # projection architecture for smaller siamese
            'projection_same': "boolean",  # whether the parameters are tied between the two projections
            'projection_nparameters': pd.Int64Dtype(),  # number of parameters of projection head
            'top1acc_in1k_official': "float64",  # accuracy that authors said they would achieve
            'top1acc_in1k-1%_official': "float64",  # accuracy that authors said they achieved on 1% of supervised data
            'top1acc_in1k-c5_official': "float64", # accuracy that authors said they would achieve on 5 classes per labels
            'n_negatives': pd.Int64Dtype(),  # number of negatives if applicable
            'finetuning_data': pd.StringDtype()}  # what data it was finetuned on