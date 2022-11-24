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
representor=simclr_rn50w4,simclr_rn50w2,simclr_rn50w2_ep100,vicreg_rn50w2,swav_rn50w2,swav_rn50w4,simclr_resnet50_d8192_e100_m2,dissl_resnet50_d4096_e100_m2,mugs_vitl16_ep250,msn_vitl16_ep600,msn_vitl7_ep200,mae_vitL16,mae_vitH14,ibot_vitL16,dissl_resnet50_d8192_e100_m2,dissl_resnet50_d8192_e400_m6,dissl_resnet50_d8192_e800_m8,clip_rn50x4,clip_rn50x16,clip_rn50x64,clip_vitL14,clip_vitL14_px336,beit_vitL16_pt22k,beitv2_vitL16_pt1k
seed=123
data=imagenet
data.kwargs.subset_raw_dataset=0.1
data.kwargs.is_avoid_raw_dataset=True
"

kwargs_multi="
representor=swav_rn50w4
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

