
dependencies = [ "torch", "torchvision"]

import logging
from pathlib import Path
import utils.helpers as _helpers

BASE_DIR = Path(__file__).absolute().parents[0]

def metadata():
    _helpers.check_import('yaml')
    import yaml

    with open(BASE_DIR/'metadata.yaml') as f:
        return yaml.safe_load(f)

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
    logging.warning(f"BYOL models not available because of the following import error: \n {e}")

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
    logging.warning(f"VISSL models not available because of the following import error: \n {e}")

##### VICReg #####
# pretrained models are from https://github.com/facebookresearch/vicreg
try:
    from hub.vicreg import get_vicreg_models as _get_vicreg_models

    def vicreg_rn50():
        return _get_vicreg_models('resnet50')

    def vicreg_rn50w2():
        return _get_vicreg_models('resnet50x2')

except ImportError as e:
    logging.warning(f"VICReg models not available because of the following import error: \n {e}")

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
    logging.warning(f"SwAV models not available because of the following import error: \n {e}")

##### SimSiam #####
# pretrained models are from https://github.com/facebookresearch/simsiam

try:
    from hub.simsiam import get_simsiam_models as _get_simsiam_models

    def simsiam_rn50_bs512_ep100():
        return _get_simsiam_models('simsiam_rn50_bs512_ep100')

    def simsiam_rn50_bs256_ep100():
        return _get_simsiam_models('simsiam_rn50_bs256_ep100')

except ImportError as e:
    logging.warning(f"SimSiam models not available because of the following import error: \n {e}")



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
    logging.warning(f"ISSL models not available because of the following import error: \n {e}")

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
    logging.warning(f"RiskDec models not available because of the following import error: \n {e}")


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
    logging.warning(f"Lossyless models not available because of the following import error: \n {e}")


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
    logging.warning(f"CLIP models not available because of the following import error: \n {e}")

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
    logging.warning(f"DINO models not available because of the following import error: \n {e}")


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

except ImportError as e:
    logging.warning(f"IBOT models not available because of the following import error: \n {e}")
