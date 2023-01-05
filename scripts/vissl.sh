#!/usr/bin/env bash

experiment="vissl"
notes="**Goal**: evaluate all the vissl models."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

# can run npidpp_rn50w2
# once https://github.com/facebookresearch/vissl/issues/516

kwargs_multi="
representor=barlow_rn50,barlow_rn50_ep300,mocov2_rn50_vissl,rotnet_rn50_in1k,rotnet_rn50_in22k,simclr_rn50,simclr_rn50_ep200,simclr_rn50_ep400,simclr_rn50_ep800,simclr_rn50_bs4096_ep100,simclr_rn50w2,simclr_rn50w2_ep100,simclr_rn101,simclr_rn101_ep100,jigsaw_rn50,jigsaw_rn50_in22k,clusterfit_rn50,npid_rn50,npidpp_rn50,pirl_rn50,pirl_rn50_ep200,pirl_rn50_headMLP,pirl_rn50_ep200_headMLP,pirl_rn50w2,pirl_rn50w2_headMLP
seed=123
predictor=torch_linear_erm
"

# TODO evaluating with unit representation to check uniformity and alignment

# torch_linear_delta_hypopt

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/torch_"$experiment".log 2>&1 &

    sleep 10

  done
fi

