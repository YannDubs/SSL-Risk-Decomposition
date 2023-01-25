from __future__ import annotations

from utils.helpers import StrFormatter

__all__ = ["PRETTY_RENAMER"]

PRETTY_RENAMER = StrFormatter(
    exact_match={'train_test': r"100\%",
                 'train-balsbst-ntrain0.01_test': r"1\%",
                'train-nperclass-5_test': "5 shot",
'train-nperclass-3_test': "3 shot",
'train-nperclass-30_test': "30 shot",
                 },
    substring_replace={
        # Math stuff
        "beta": r"$\beta$",
        "calFissl": r"$\mathcal{F}_{issl}$",
        "calFpred": r"$\mathcal{F}_{pred}$",
        "calF --": r"$\mathcal{F}^{-}$",
        "calF ++": r"$\mathcal{F}^{+}$",
        "calF": r"$\mathcal{F}$",
        " --": r"⁻",
        " ++": r"⁺",
        # General
        "Enc_Gen": "Enc. Gen.",
        "Probe_Gen": "Probe Gen.",
        "Approx": "Approx.",
        "Agg_": "Agg. ",
        "Train_": "",
        "Trainrealaug_": "",
        "Trainaug_": "",
        # Post underscore
        "_": " ",
        "Alpha": r"\alpha",
        "Vits": "ViT-S",
        "Vitb": "ViT-B",
        "Vith": "ViT-H",
        "Vitl": "ViT-L",
        "Vit": "ViT",
        "Convnexts": "ConvNext-S",
        "Convnextb": "ConvNext-B",
        "Convnextl": "ConvNext-L",
        "Convnextxl": "ConvNext-XL",
        "Convnext": "ConvNext",
        "Resnet50": "Rn50",
        "Resnet101": "Rn101",
        "Resnet": "ResNet",
        "4Xcls": "4xCls",
        # losses name
        "Rn50D": "RN50 dim. ",
        "Rn50W": "RN50w",
        "Rn50X": "RN50x",
        "Locnet": "LocNet",
        "Rotnet": "RotNet",
        "Jigsaw": "Jigsaw",
        "Simclr": "SimCLR",
        "Moco": "MoCo",
        "Speccl": "SpecCL",
        "Infomin": "InfoMin",
        "Densecl": "DenseCL",
        "Deepcluster": "DeepCluster",
        "Dc2": "DeepClusterv2",
        "Clusterfit": "ClusterFit",
        "Swav": "SwAV",
        "Sela": "SeLa",
        "Ibot": "iBOT",
        "Simsiam": "SimSiam",
        "Lossyless": "LossyLess",
        "Barlow": "BarlowTwins",
        "Vicreg": "VICReg",
        "Vicregl": "VICRegL",
        "Beit": "BEiT",
        "Npidpp": "NPID++",
        "v1": "-v1",
        "v2": "-v2",
        "v3": "-v3",
        "Z Dim": "Z Dim.",
        "Is Aug ": "",
        "Rank": "Eff. Dim.",
        "Agg. Risk Norm": "Agg. Risk Norm.",
        "Is Industry": "Industry?",
        "N Augmentations": "Num. Aug.",
        "Nviews": "Num. Views",
        "Hidden": "Hid",
        #"Projection Nparameters Hidden": "Projection Nparameters",
        "Nparameters": "N Parameters",
        "N Parameters": "Num. Param.",
        "Architecture Exact": "Exact Arch.",
        "Architecture": "Arch.",
        "Arch": "Arch.",
        "Projection2": "Proj.",
        "Projection1": "1st Proj.",
        "Projection": "Proj.",
        "Hid": "Hid.",
        "Vars": "Intra/Inter Variance",
        "Adamw": "AdamW",
        "Normshap": "Normalized Shap",
"Identity": "None",
        ###
        "..": ".",
    },
    to_upper=["Rn50", "Rn101", "Npid", "Pirl", "Byol", "Clip", "Mugs", "Odc", "Dino", "Msn", "Dissl", "Byol",
              "Mae", "Ssl", "Sgd", "Lars", "Shap", "Mlp"],

)
