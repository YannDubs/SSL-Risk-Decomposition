#!/usr/bin/env bash

experiment="riskdec"
notes="**Goal**: evaluate all the models we pretrained."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

kwargs_multi="
representor=dissl_resnet50_dNone_e100_m2_augLarge,dissl_resnet50_dNone_e100_m2_augSmall,dissl_resnet50_dNone_e100_m2_headTLinSLin,dissl_resnet50_dNone_e100_m2_headTMlpSMlp,simclr_resnet50_dNone_e100_m2,simclr_resnet50_dNone_e100_m2_data010,simclr_resnet50_dNone_e100_m2_data030,simclr_resnet50_dNone_e100_m2_headTLinSLin,simclr_resnet50_dNone_e100_m2_headTMlpSLin,simclr_resnet50_dNone_e100_m2_headTMlpSMlp,simclr_resnet50_dNone_e100_m2_headTNoneSNone,speccl_resnet50_bs384_ep100
seed=123
predictor=sk_logistic_hypopt
data.kwargs.is_avoid_raw_dataset=True
data.kwargs.subset_raw_dataset=0.3
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

