#!/usr/bin/env bash

experiment="statistics"
notes="**Goal**: estimate all the desired statistics of the representations."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

#
kwargs_multi="
representor=barlow_rn50,barlow_rn50_ep300,mocov2_rn50_vissl,rotnet_rn50_in1k,rotnet_rn50_in22k,simclr_rn50,simclr_rn50_ep200,simclr_rn50_ep400,simclr_rn50_ep800,simclr_rn50_bs4096_ep100,simclr_rn101,simclr_rn101_ep100,jigsaw_rn50,jigsaw_rn50_in22k,clusterfit_rn50,npid_rn50,npidpp_rn50,vicreg_rn50,swav_rn50,swav_rn50_ep100,swav_rn50_ep200,swav_rn50_ep200_bs256,swav_rn50_ep400,swav_rn50_ep400_2x224,swav_rn50_ep400_bs256,dc2_rn50_ep400_2x224,dc2_rn50_ep400_2x160_4x96,dc2_rn50_ep800_2x224_6x96,selav2_rn50_ep400_2x224,simsiam_rn50_bs512_ep100,simsiam_rn50_bs256_ep100,dissl_resnet50_dNone_e100_m2_augLarge,dissl_resnet50_dNone_e100_m2_augSmall,dissl_resnet50_dNone_e100_m2_headTLinSLin,dissl_resnet50_dNone_e100_m2_headTMlpSMlp,simclr_resnet50_dNone_e100_m2,simclr_resnet50_dNone_e100_m2_data010,simclr_resnet50_dNone_e100_m2_data030,simclr_resnet50_dNone_e100_m2_headTLinSLin,simclr_resnet50_dNone_e100_m2_headTMlpSLin,simclr_resnet50_dNone_e100_m2_headTMlpSMlp,simclr_resnet50_dNone_e100_m2_headTNoneSNone,speccl_resnet50_bs384_ep100,simclr_resnet50_d8192_e100_m2,infomin_rn50_200ep,infomin_rn50_800ep,mugs_vits16_ep100,mugs_vits16_ep300,mugs_vits16_ep800,mugs_vitb16_ep400,mugs_vits16_ep800_extractS,mugs_vitb16_ep400_extractB,msn_vits16_ep800,msn_vitb16_ep600,msn_vitb4_ep300,mocov3_rn50_ep100,mocov3_rn50_ep300,mocov3_rn50_ep1000,mocov3_vitS_ep300,mocov3_vitB_ep300,mocov1_rn50_ep200,mocov2_rn50_ep200,mocov2_rn50_ep800,relativeloc_rn50_70ep_mmselfsup,odc_rn50_440ep_mmselfsup,densecl_rn50_200ep_mmselfsup,simsiam_rn50_bs256_ep200_mmselfsup,simclr_rn50_bs256_ep200_mmselfsup,deepcluster_rn50_bs512_ep200_mmselfsup,mae_vitB16,lossyless_vitb32_b001,lossyless_vitb32_b005,lossyless_vitb32_b01,ibot_vitB16,ibot_vitS16,ibot_vitB16_extractB,ibot_vitS16_extractS,dissl_resnet50_dNone_e100_m2,dissl_resnet50_dNone_e400_m2,dissl_resnet50_dNone_e400_m6,dino_rn50,dino_vitS16_last,dino_vitS8_last,dino_vitB16_last,dino_vitB8_last,dino_vitS16,dino_vitB16,dino_vitB8,dino_vitS16_extractB,dino_vitB16_extractS,clip_rn50,clip_rn101,clip_vitB16,clip_vitB32,byol_rn50_augCrop,byol_rn50_augCropBlur,byol_rn50_augCropColor,byol_rn50_augNocolor,byol_rn50_augNogray,byol_rn50_bs64,byol_rn50_bs128,byol_rn50_bs256,byol_rn50_bs512,byol_rn50_bs1024,byol_rn50_bs2048,byol_rn50_bs4096,beit_vitB16_pt22k,beitv2_vitB16_pt1k,beitv2_vitB16_pt1k_extractB,beitv2_vitB16_pt1k_ep300
seed=123
data=imagenet
data.kwargs.subset_raw_dataset=0.1
data.kwargs.is_avoid_raw_dataset=True
"




if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python statistics.py +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/torch_"$experiment".log 2>&1 &

    sleep 10

  done
fi

