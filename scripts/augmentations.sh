#!/usr/bin/env bash

experiment="augmentations"
notes="**Goal**: evaluate some models with data augmentations for the probes."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

#
kwargs_multi="
representor=swav_rn50,dissl_resnet50_dNone_e400_m6,clip_vitB32,lossyless_vitb32_b005,msn_vits16_ep800,simclr_rn50,dissl_resnet50_d8192_e400_m6
seed=123
predictor=torch_linear_aug
data=imagenet_imgs
"


kwargs_multi="
representor=lossyless_vitb32_b005
seed=123
predictor=torch_linear_aug
data=imagenet_imgs
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python main_augs.py +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/torch_"$experiment".log 2>&1 &

    sleep 10

  done
fi

