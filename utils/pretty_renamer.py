from __future__ import annotations

from utils.helpers import StrFormatter

__all__ = ["PRETTY_RENAMER"]

PRETTY_RENAMER = StrFormatter(
    exact_match={},
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
        # Post underscore
        "_": " ",
        "Vits": "ViT-S",
        "Vitb": "ViT-B",
        "Vith": "ViT-H",
        "Vitl": "ViT-L",
        "Resnet": "RN",
        # losses name
        "Locnet": "LocNet",
        "Rotnet": "RotNet",
        "Jigsaw": "Jigsaw",
        "Simclr": "SimCLR",
        "Moco": "MoCo",
        "Speccl": "SpecCL",
        "Deepcluster": "DeepCluster",
        "Clusterfit": "ClusterFit",
        "Swav": "SwAV",
        "Sela": "SeLa",
        "Ibot": "iBOT",
        "Simsiam": "SimSiam",
        "Barlow": "Barlow Twins",
        "Vicreg": "VICReg",
        "Beit": "BEiT",
        "Npidpp": "NPID++",
    },
    to_upper=["Rn50", "Rn101", "Npid", "Pirl", "Byol", "Clip", "Mugs", "Odc", "Dino", "Msn", "Dissl", "BYOL",
              "Mae"],
)
