#!/usr/bin/env bash

# parses special mode for running the script
source `dirname $0`/utils.sh

experiment=$prfx"architectures"
notes="**Goal**: compare swav trained for a different number of epochs."

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"

# every arguments that you are sweeping over
kwargs_multi="
representor=simclr_rn101,simclr_rn50,dino_vitS16,dino_vitB16
"
# also add simclr_rn50w2, dino_vitB8 from other experiments

kwargs_multi="
representor=dino_vitS16
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 10

  done
fi

wait

# ADD plotting behavior