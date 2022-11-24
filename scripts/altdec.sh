#!/usr/bin/env bash

experiment="altdec"
notes="**Goal**: evaluate all the vissl models."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
timeout=$time
"


kwargs_multi="
seed=123
predictor=torch_linear_delta_hypopt
is_alternative_decomposition=true
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "representor=byol_rn50_bs256,byol_rn50_bs2048 experiment=byol" "representor=barlow_rn50,simclr_rn50 experiment=vissl" # "representor=swav_rn50 experiment=swav" "representor=dissl_resnet50_dNone_e400_m6,dissl_resnet50_d8192_e400_m6 experiment=dissl" "representor=clip_vitB32 experiment=clip" "representor=lossyless_vitb32_b005 experiment=lossyless" "representor=msn_vits16_ep800 experiment=msn"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/torch_"$experiment".log 2>&1 &

    sleep 10

  done
fi

