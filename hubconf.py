
dependencies = [ "torch", "torchvision", "timm"]

import logging as _logging
import pathlib as _pathlib

BASE_DIR = _pathlib.Path(__file__).absolute().parents[0]

def metadata_dict():
    try:
        import yaml
    except ImportError:
        raise ImportError("Please install `pyaml` to use metadata_dict")

    with open(BASE_DIR/'metadata.yaml') as f:
        return yaml.safe_load(f)

def metadata_df(is_multiindex=False, is_lower=True):
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Please install `pandas` to use metadata_df")

    metadata_flatten = {k1: {(k2, k3): v
                             for k2, d2 in d.items()
                             for k3, v in d2.items()
                             }
                        for k1, d in metadata_dict().items()}
    df = pd.DataFrame.from_dict(metadata_flatten, orient="index")

    if not is_multiindex:
        df = df.droplevel(0, axis=1)

    if is_lower:
        df.applymap(lambda s: s.lower() if isinstance(s,str) else s)
        if df.index.nlevels > 1:
            df.columns = pd.MultiIndex.from_tuples(tuple(c.lower() if isinstance(c, str) else c
                                                         for c in t)
                                                   for t in df.columns )
        else:
            df.columns = [c.lower() if isinstance(c, str) else c
                          for c in df.columns ]
        df.index = [i.lower() if isinstance(i, str) else i
                    for i in df.index if isinstance(i, str)]

    return df



##### BYOL #####
# pretrained models are from https://github.com/deepmind/deepmind-research/tree/master/byol
# converted to pytorch using https://github.com/ajtejankar/byol-convert

try:
    from hub.byol import get_byol_models as _get_byol_models

    def byol_rn50_augCrop():
        return _get_byol_models('byol_rn50_augCrop', "res50x1_crop_only", architecture='resnet50')

    def byol_rn50_augCropBlur():
        return _get_byol_models('byol_rn50_augCropBlur', "res50x1_crop_and_blur_only", architecture='resnet50')

    def byol_rn50_augCropColor():
        return _get_byol_models('byol_rn50_augCropColor', "res50x1_crop_and_color_only", architecture='resnet50')

    def byol_rn50_augNocolor():
        return _get_byol_models('byol_rn50_augNocolor', "res50x1_no_color", architecture='resnet50')

    def byol_rn50_augNogray():
        return _get_byol_models('byol_rn50_augNogray', "res50x1_no_grayscale", architecture='resnet50')

    def byol_rn50_bs64():
        return _get_byol_models('byol_rn50_bs64', "res50x1_batchsize_64", architecture='resnet50')

    def byol_rn50_bs128():
        return _get_byol_models('byol_rn50_bs128', "res50x1_batchsize_128", architecture='resnet50')

    def byol_rn50_bs256():
        return _get_byol_models('byol_rn50_bs256', "res50x1_batchsize_256", architecture='resnet50')

    def byol_rn50_bs512():
        return _get_byol_models('byol_rn50_bs512', "res50x1_batchsize_512", architecture='resnet50')

    def byol_rn50_bs1024():
        return _get_byol_models('byol_rn50_bs1024', "res50x1_batchsize_1024", architecture='resnet50')

    def byol_rn50_bs2048():
        return _get_byol_models('byol_rn50_bs2048', "res50x1_batchsize_2048", architecture='resnet50')

    def byol_rn50_bs4096():
        return _get_byol_models('byol_rn50_bs4096', "pretrain_res50x1", architecture='resnet50')

except ImportError as e:
    _logging.warning(f"BYOL models not available because of the following import error: \n {e}")

##### VISSL #####
# pretrained models are from https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md

try:
    from hub.vissl import get_vissl_models as _get_vissl_models

    def barlow_rn50():
        return _get_vissl_models('barlow_rn50', architecture='resnet50')

    def barlow_rn50_ep300():
        return _get_vissl_models('barlow_rn50_ep300', architecture='resnet50')

    def mocov2_rn50_vissl():
        return _get_vissl_models('mocov2_rn50_vissl', architecture='resnet50')

    def rotnet_rn50_in1k():
        return _get_vissl_models('rotnet_rn50_in1k', architecture='resnet50')

    def rotnet_rn50_in22k():
        return _get_vissl_models('rotnet_rn50_in22k', architecture='resnet50')

    def simclr_rn50():
        return _get_vissl_models('simclr_rn50', architecture='resnet50')

    def simclr_rn50_ep200():
        return _get_vissl_models('simclr_rn50_ep200', architecture='resnet50')

    def simclr_rn50_ep400():
        return _get_vissl_models('simclr_rn50_ep400', architecture='resnet50')

    def simclr_rn50_ep800():
        return _get_vissl_models('simclr_rn50_ep800', architecture='resnet50')

    def simclr_rn50_bs4096_ep100():
        return _get_vissl_models('simclr_rn50_bs4096_ep100', architecture='resnet50')

    def simclr_rn50w2():
        return _get_vissl_models('simclr_rn50w2', architecture='resnet50', width_multiplier=2)

    def simclr_rn50w2_ep100():
        return _get_vissl_models('simclr_rn50w2_ep100', architecture='resnet50', width_multiplier=2)

    def simclr_rn50w4():
        return _get_vissl_models('simclr_rn50w4', architecture='resnet50', width_multiplier=4)

    def simclr_rn101():
        return _get_vissl_models('simclr_rn101', architecture='resnet101')

    def simclr_rn101_ep100():
        return _get_vissl_models('simclr_rn101_ep100', architecture='resnet101')

    def jigsaw_rn50_in22k():
        return _get_vissl_models('jigsaw_rn50_in22k', architecture='resnet50')

    def jigsaw_rn50():
        return _get_vissl_models('jigsaw_rn50', architecture='resnet50')

    def clusterfit_rn50():
        return _get_vissl_models('clusterfit_rn50', architecture='resnet50')

    def npid_rn50():
        return _get_vissl_models('npid_rn50', architecture='resnet50')

    def npidpp_rn50():
        return _get_vissl_models('npidpp_rn50', architecture='resnet50')

    # not working until https://github.com/facebookresearch/vissl/issues/516
    def npidpp_rn50w2():
        return _get_vissl_models('npidpp_rn50w2', architecture='resnet50', width_multiplier=2)

    def pirl_rn50():
        return _get_vissl_models('pirl_rn50', architecture='resnet50')

    def pirl_rn50_ep200():
        return _get_vissl_models('pirl_rn50_ep200', architecture='resnet50')

    def pirl_rn50_headMLP():
        return _get_vissl_models('pirl_rn50_headMLP', architecture='resnet50')

    def pirl_rn50_ep200_headMLP():
        return _get_vissl_models('pirl_rn50_ep200_headMLP', architecture='resnet50')

    def pirl_rn50w2():
        return _get_vissl_models('pirl_rn50w2', architecture='resnet50', width_multiplier=2)

    def pirl_rn50w2_headMLP():
        return _get_vissl_models('pirl_rn50w2_headMLP', architecture='resnet50', width_multiplier=2)

except ImportError as e:
    _logging.warning(f"VISSL models not available because of the following import error: \n {e}")

##### VICReg #####
# pretrained models are from https://github.com/facebookresearch/vicreg
try:
    from hub.vicreg import get_vicreg_models as _get_vicreg_models

    def vicreg_rn50():
        return _get_vicreg_models('resnet50')

    def vicreg_rn50w2():
        return _get_vicreg_models('resnet50x2')

except ImportError as e:
    _logging.warning(f"VICReg models not available because of the following import error: \n {e}")

##### SwAV #####
# pretrained models are from https://github.com/facebookresearch/vicreg

try:
    from hub.swav import get_swav_models as _get_swav_models

    def swav_rn50():
        return _get_swav_models('swav_rn50', "resnet50", architecture='resnet50')

    def swav_rn50_ep100():
        return _get_swav_models('swav_rn50_ep100', "resnet50_ep100", architecture='resnet50')

    def swav_rn50_ep200():
        return _get_swav_models('swav_rn50_ep200', "resnet50_ep200", architecture='resnet50')

    def swav_rn50_ep200_bs256():
        return _get_swav_models('swav_rn50_ep200_bs256', "resnet50_ep200_bs256", architecture='resnet50')

    def swav_rn50_ep400():
        return _get_swav_models('swav_rn50_ep400', "resnet50_ep400", architecture='resnet50')

    def swav_rn50_ep400_2x224():
        return _get_swav_models('swav_rn50_ep400_2x224', "resnet50_ep400_2x224", architecture='resnet50')

    def swav_rn50_ep400_bs256():
        return _get_swav_models('swav_rn50_ep400_bs256', "resnet50_ep400_bs256", architecture='resnet50')

    def swav_rn50w2():
        return _get_swav_models('swav_rn50w2', "resnet50w2", architecture='resnet50w2')

    def swav_rn50w4():
        return _get_swav_models('swav_rn50w4', "resnet50w4", architecture='resnet50w4')

    def swav_rn50w5():
        return _get_swav_models('swav_rn50w5', "resnet50w5", architecture='resnet50w5')

    def dc2_rn50_ep400_2x224():
        return _get_swav_models('dc2_rn50_ep400_2x224', 'dc2_rn50_ep400_2x224', architecture='resnet50')

    def dc2_rn50_ep400_2x160_4x96():
        return _get_swav_models('dc2_rn50_ep400_2x160_4x96', 'dc2_rn50_ep400_2x160_4x96', architecture='resnet50')

    def dc2_rn50_ep800_2x224_6x96():
        return _get_swav_models('dc2_rn50_ep800_2x224_6x96',  'dc2_rn50_ep800_2x224_6x96', architecture='resnet50')

    def selav2_rn50_ep400_2x224():
        return _get_swav_models('selav2_rn50_ep400_2x224', 'selav2_rn50_ep400_2x224', architecture='resnet50')

    def selav2_rn50_ep400_2x160_4x96():
        return _get_swav_models('selav2_rn50_ep400_2x160_4x96', 'selav2_rn50_ep400_2x160_4x96', architecture='resnet50')

except ImportError as e:
    _logging.warning(f"SwAV models not available because of the following import error: \n {e}")

##### SimSiam #####
# pretrained models are from https://github.com/facebookresearch/simsiam

try:
    from hub.simsiam import get_simsiam_models as _get_simsiam_models

    def simsiam_rn50_bs512_ep100():
        return _get_simsiam_models('simsiam_rn50_bs512_ep100')

    def simsiam_rn50_bs256_ep100():
        return _get_simsiam_models('simsiam_rn50_bs256_ep100')

except ImportError as e:
    _logging.warning(f"SimSiam models not available because of the following import error: \n {e}")



##### ISSL #####
# pretrained models are from https://github.com/YannDubs/Invariant-Self-Supervised-Learning

try:
    from hub.issl import get_issl_models as _get_issl_models

    def dissl_resnet50_dNone_e100_m2():
        return _get_issl_models('dissl_resnet50_dNone_e100_m2')

    def dissl_resnet50_d8192_e100_m2():
        return _get_issl_models('dissl_resnet50_d8192_e100_m2')

    def dissl_resnet50_dNone_e400_m2():
        return _get_issl_models('dissl_resnet50_dNone_e400_m2')

    def dissl_resnet50_dNone_e400_m6():
        return _get_issl_models('dissl_resnet50_dNone_e400_m6')

    def dissl_resnet50_d8192_e400_m6():
        return _get_issl_models('dissl_resnet50_d8192_e400_m6')

    def dissl_resnet50_d8192_e800_m8():
        return _get_issl_models('dissl_resnet50_d8192_e800_m8')

except ImportError as e:
    _logging.warning(f"ISSL models not available because of the following import error: \n {e}")

##### Riskdec #####
# pretrained models available on our directory https://github.com/YannDubs/SSL-Risk-Decomposition

try:
    from hub.riskdec import get_riskdec_models as _get_riskdec_models

    def dissl_resnet50_dNone_e100_m2_augLarge():
        return _get_riskdec_models('dissl_resnet50_dNone_e100_m2_augLarge')

    def dissl_resnet50_dNone_e100_m2_augSmall():
        return _get_riskdec_models('dissl_resnet50_dNone_e100_m2_augSmall')

    def dissl_resnet50_dNone_e100_m2_headTLinSLin():
        return _get_riskdec_models('dissl_resnet50_dNone_e100_m2_headTLinSLin')

    def dissl_resnet50_dNone_e100_m2_headTMlpSMlp():
        return _get_riskdec_models('dissl_resnet50_dNone_e100_m2_headTMlpSMlp')

    def dissl_resnet50_d4096_e100_m2():
        return _get_riskdec_models('dissl_resnet50_d4096_e100_m2', dim=4096)

    def simclr_resnet50_dNone_e100_m2():
        return _get_riskdec_models('simclr_resnet50_dNone_e100_m2')

    def simclr_resnet50_dNone_e100_m2_data010():
        return _get_riskdec_models('simclr_resnet50_dNone_e100_m2_data010')

    def simclr_resnet50_dNone_e100_m2_data030():
        return _get_riskdec_models('simclr_resnet50_dNone_e100_m2_data030')

    def simclr_resnet50_dNone_e100_m2_headTLinSLin():
        return _get_riskdec_models('simclr_resnet50_dNone_e100_m2_headTLinSLin')

    def simclr_resnet50_dNone_e100_m2_headTMlpSLin():
        return _get_riskdec_models('simclr_resnet50_dNone_e100_m2_headTMlpSLin')

    def simclr_resnet50_dNone_e100_m2_headTMlpSMlp():
        return _get_riskdec_models('simclr_resnet50_dNone_e100_m2_headTMlpSMlp')

    def simclr_resnet50_dNone_e100_m2_headTNoneSNone():
        return _get_riskdec_models('simclr_resnet50_dNone_e100_m2_headTNoneSNone')

    def simclr_resnet50_d8192_e100_m2():
        return _get_riskdec_models('simclr_resnet50_d8192_e100_m2', dim=8192)

    def speccl_bs384_ep100():
        return _get_riskdec_models('speccl_bs384_ep100', is_speccl=True)

except ImportError as e:
    _logging.warning(f"RiskDec models not available because of the following import error: \n {e}")


##### Lossyless #####
# pretrained models are from https://github.com/YannDubs/lossyless

try:
    from hub.lossyless import get_lossyless_models as _get_lossyless_models

    def lossyless_b001():
        return _get_lossyless_models('clip_compressor_b001')

    def lossyless_b005():
        return _get_lossyless_models('clip_compressor_b005')

    def lossyless_b01():
        return _get_lossyless_models('clip_compressor_b01')

except ImportError as e:
    _logging.warning(f"Lossyless models not available because of the following import error: \n {e}")


##### CLIP #####
# pretrained models are from https://github.com/openai/CLIP

try:
    from hub.clip import get_clip_models as _get_clip_models

    def clip_rn50():
        return _get_clip_models('RN50')

    def clip_rn50x4():
        return _get_clip_models('RN50x4')

    def clip_rn50x16():
        return _get_clip_models('RN50x16')

    def clip_rn50x64():
        return _get_clip_models('RN50x64')

    def clip_rn101():
        return _get_clip_models('RN101')

    def clip_vitB16():
        return _get_clip_models('ViT-B/16')

    def clip_vitB32():
        return _get_clip_models('ViT-B/32')

    def clip_vitL14():
        return _get_clip_models('ViT-L/14')

    def clip_vitL14_px336():
        return _get_clip_models('ViT-L/14@336px')

except ImportError as e:
    _logging.warning(f"CLIP models not available because of the following import error: \n {e}")

##### DINO #####
# pretrained models are from https://github.com/facebookresearch/dino

try:
    from hub.dino import get_dino_models as _get_dino_models

    def dino_rn50():
        return _get_dino_models("dino_resnet50")

    def dino_vitS16_last():
        return _get_dino_models("dino_vits16")

    def dino_vitS8_last():
        return _get_dino_models("dino_vits8")

    def dino_vitB16_last():
        return _get_dino_models("dino_vitb16")

    def dino_vitB8_last():
        return _get_dino_models("dino_vitb8")

    def dino_vitS16():
        return _get_dino_models("dino_vits16", representation="4xcls")

    def dino_vitB16():
        return _get_dino_models("dino_vitb16", representation="cls+avg")

    def dino_vitB8():
        return _get_dino_models("dino_vitb8", representation="cls+avg")

    def dino_vitS16_extractB():
        return _get_dino_models("dino_vits16", representation="cls+avg")

    def dino_vitB16_extractS():
        return _get_dino_models("dino_vitb16", representation="4xcls")

except ImportError as e:
    _logging.warning(f"DINO models not available because of the following import error: \n {e}")


##### IBOT #####
# pretrained models are from https://github.com/bytedance/ibot

try:
    from hub.ibot import get_ibot_models as _get_ibot_models

    def ibot_vitB16():
        return _get_ibot_models("ibot_vitB16", "vitb16", 'vit_base_patch16_224')

    def ibot_vitS16():
        return _get_ibot_models("ibot_vitS16", "vits16", 'vit_small_patch16_224')

    def ibot_vitL16():
        return _get_ibot_models("ibot_vitL16", "vitl16", 'vit_large_patch16_224')

    def ibot_vitB16_extractB():
        return _get_ibot_models("ibot_vitB16", "vitb16", 'vit_base_patch16_224', representation="cls+avg")

    def ibot_vitS16_extractS():
        return _get_ibot_models("ibot_vitS16", "vits16", 'vit_small_patch16_224', representation="4xcls")

except ImportError as e:
    _logging.warning(f"IBOT models not available because of the following import error: \n {e}")


##### MUGS #####
# pretrained models are from https://github.com/sail-sg/mugs

try:
    from hub.mugs import get_mugs_models as _get_mugs_models

    def mugs_vits16_ep100():
        return _get_mugs_models("mugs_vits16_ep100", 'vit_small_patch16_224')

    def mugs_vits16_ep300():
        return _get_mugs_models("mugs_vits16_ep300", 'vit_small_patch16_224')

    def mugs_vits16_ep800():
        return _get_mugs_models("mugs_vits16_ep800", 'vit_small_patch16_224')

    def mugs_vitb16_ep400():
        return _get_mugs_models("mugs_vitb16_ep400", 'vit_base_patch16_224')

    def mugs_vitl16_ep250():
        return _get_mugs_models("mugs_vitl16_ep250", 'vit_large_patch16_224')

    def mugs_vits16_ep800_extractS():
        return _get_mugs_models("mugs_vits16_ep800", 'vit_small_patch16_224', representation="4xcls")

    def mugs_vitb16_ep400_extractB():
        return _get_mugs_models("mugs_vitb16_ep400", 'vit_base_patch16_224', representation="cls+avg")

except ImportError as e:
    _logging.warning(f"MUGS models not available because of the following import error: \n {e}")

##### MAE #####
# pretrained models are from https://github.com/facebookresearch/mae

try:
    from hub.mae import get_mae_models as _get_mae_models

    def mae_vitB16():
        return _get_mae_models("mae_vitB16", 'vit_base_patch16_224')

    def mae_vitL16():
        return _get_mae_models("mae_vitL16", 'vit_large_patch16_224')

    def mae_vitH14():
        return _get_mae_models("mae_vitH14", 'vit_huge_patch14_224')

except ImportError as e:
    _logging.warning(f"MAE models not available because of the following import error: \n {e}")


##### MSN #####
# pretrained models are from https://github.com/facebookresearch/msn

try:
    from hub.msn import get_msn_models as _get_msn_models

    def msn_vits16_ep800():
        return _get_msn_models("msn_vits16_ep800", 'vit_small_patch16_224')

    def msn_vitb16_ep600():
        return _get_msn_models("msn_vitb16_ep600", 'vit_base_patch16_224')

    def msn_vitb4_ep300():
        return _get_msn_models("msn_vitb4_ep300", 'vit_base_patch4_224')

    def msn_vitl16_ep600():
        return _get_msn_models("msn_vitl16_ep600", 'vit_large_patch16_224')

    def msn_vitl7_ep200():
        return _get_msn_models("msn_vitl7_ep200", 'vit_large_patch7_224')

except ImportError as e:
    _logging.warning(f"MSN models not available because of the following import error: \n {e}")

### MOCOV3 ###
# pretrained models are from https://github.com/facebookresearch/moco-v3

try:
    from hub.mocov3 import get_mocov3_models as _get_mocov3_models

    def mocov3_rn50_ep100():
        return _get_mocov3_models("mocov3_rn50_ep100", 'resnet50')

    def mocov3_rn50_ep300():
        return _get_mocov3_models("mocov3_rn50_ep300", 'resnet50')

    def mocov3_rn50_ep1000():
        return _get_mocov3_models("mocov3_rn50_ep1000", 'resnet50')

    def mocov3_vitS_ep300():
        return _get_mocov3_models("mocov3_vitS_ep300", 'vit_small_patch16_224')

    def mocov3_vitB_ep300():
        return _get_mocov3_models("mocov3_vitB_ep300", 'vit_base_patch16_224')

except ImportError as e:
    _logging.warning(f"MOCOV3 models not available because of the following import error: \n {e}")

### MOCO ###
# pretrained models are from https://github.com/facebookresearch/moco

try:
    from hub.moco import get_moco_models as get_moco_models

    def mocov1_rn50_ep200():
        return get_moco_models("mocov1_rn50_ep200")

    def mocov2_rn50_ep200():
        return get_moco_models("mocov2_rn50_ep200")

    def mocov2_rn50_ep800():
        return get_moco_models("mocov2_rn50_ep800")

except ImportError as e:
    _logging.warning(f"MOCO models not available because of the following import error: \n {e}")

### PYCONTRAST ###
# pretrained models are from https://github.com/HobbitLong/PyContrast
try:
    from hub.pycontrast import get_pycontrast_models as get_pycontrast_models

    def infomin_rn50_200ep():
        return get_pycontrast_models("infomin_rn50_200ep")

    def infomin_rn50_800ep():
        return get_pycontrast_models("infomin_rn50_800ep")

except ImportError as e:
    _logging.warning(f"Pycontrast models not available because of the following import error: \n {e}")


### MMSelfSup ###

try:
    from hub.mmselfsup import get_mmselfsup_models as get_mmselfsup_models

    def relativeloc_rn50_70ep_mmselfsup():
        return get_mmselfsup_models("relativeloc_rn50_70ep_mmselfsup")

    def odc_rn50_440ep_mmselfsup():
        return get_mmselfsup_models("odc_rn50_440ep_mmselfsup")

    def densecl_rn50_200ep_mmselfsup():
        return get_mmselfsup_models("densecl_rn50_200ep_mmselfsup")

    def simsiam_rn50_bs256_ep200_mmselfsup():
        return get_mmselfsup_models("simsiam_rn50_bs256_ep200_mmselfsup")

    def simclr_rn50_bs256_ep200_mmselfsup():
        return get_mmselfsup_models("simclr_rn50_bs256_ep200_mmselfsup")

    def deepcluster_rn50_bs512_ep200_mmselfsup():
        return get_mmselfsup_models("deepcluster_rn50_bs512_ep200_mmselfsup")

except ImportError as e:
    _logging.warning(f"MMSelfSup models not available because of the following import error: \n {e}")

### BEIT ###
try:
    from hub.beit import get_beit_models as _get_beit_models

    def beit_vitB16_pt22k():
        return _get_beit_models("beit_vitB16_pt22k", 'beit_base_patch16_224', normalize="half")

    def beit_vitL16_pt22k():
        return _get_beit_models("beit_vitL16_pt22k", 'beit_large_patch16_224', normalize="half")

    def beitv2_vitB16_pt1k_ep300():
        return _get_beit_models("beitv2_vitB16_pt1k_ep300", 'beit_base_patch16_224')

    def beitv2_vitB16_pt1k():
        return _get_beit_models("beitv2_vitB16_pt1k", 'beit_base_patch16_224')

    def beitv2_vitL16_pt1k():
        return _get_beit_models("beitv2_vitL16_pt1k", 'beit_large_patch16_224')

    def beitv2_vitB16_pt1k_extractB():
        return _get_beit_models("beitv2_vitB16_pt1k", 'beit_base_patch16_224', representation="cls+avg")

except ImportError as e:
    _logging.warning(f"BEIT models not available because of the following import error: \n {e}")

### TIMM ###
try:
    from hub.timm import get_timm_models as _get_timm_models

    def sup_vitB8():
        return _get_timm_models('vit_base_patch8_224')

    def sup_vitB8_dino():
        return _get_timm_models('vit_base_patch8_224', representation_vit='cls+avg')

    def sup_vitB16():
        return _get_timm_models('vit_base_patch16_224')

    def sup_vitB16_dino():
        return _get_timm_models('vit_base_patch16_224', representation_vit="cls+avg")

    def sup_vitB16_dino_extractS():
        return _get_timm_models('vit_base_patch16_224', representation_vit="4xcls")

    def sup_vitB32():
        return _get_timm_models('vit_base_patch32_224')

    def sup_vitH14():
        return _get_timm_models('vit_huge_patch14_224')

    def sup_vitL14():
        return _get_timm_models('vit_large_patch14_224')

    def sup_vitL16():
        return _get_timm_models('vit_large_patch16_224')

    def sup_vitS16():
        return _get_timm_models('vit_small_patch16_224')

    def sup_vitS16_dino():
        return _get_timm_models('vit_small_patch16_224', representation_vit="4xcls")

    def sup_vitS16_dino_extractB():
        return _get_timm_models('vit_small_patch16_224', representation_vit="cls+avg")

except ImportError as e:
    _logging.warning(f"TIMM models not available because of the following import error: \n {e}")

### TORCHVISION ###

try:
    from hub.torchvision import get_torchvision_models as _get_torchvision_models

    def sup_rn50():
        return _get_torchvision_models("resnet50")

    def sup_rn50w2():
        return _get_torchvision_models("wide_resnet50_2")

    def sup_rn101():
        return _get_torchvision_models("resnet101")

except ImportError as e:
    _logging.warning(f"Torchvision models not available because of the following import error: \n {e}")
