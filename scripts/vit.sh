#!/usr/bin/env bash

experiment="vit"
notes="**Goal**: evaluate all the vit models."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"


# ,mae_vitL16,mae_vitH14,mae_vitB16_ft1k,ibot_vitS16,ibot_vitB16,beit_vitL16,beit_vitB16_in22k_ft22k,beit_vitB16_in22k,beit_vitB16
kwargs_multi="
representor=mugs_vits16_ep800,mugs_vits16_ep300,mugs_vits16_ep100,mugs_vitL16_ep250,mugs_vitb16_ep400,mocov3_vitS_ep300,mocov3_vitB_ep300
seed=123
predictor=torch_linear,torch_linear_dino
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

wait
