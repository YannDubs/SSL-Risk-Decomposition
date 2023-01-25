#!/usr/bin/env bash

experiment="openclip"
notes="**Goal**: evaluate all the openclip models."

# parses special mode for running the script
source `dirname $0`/utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
timeout=$time
"


kwargs_multi="
representor=openclip_vitG14_extractPred,openclip_vitG14_extractS,openclip_vitG14_extractB
seed=123
predictor=sk_logistic_hypopt
data.kwargs.is_avoid_raw_dataset=True
data.kwargs.subset_raw_dataset=0.3
"

kwargs_multi="
representor=openclip_vitH14_extractPred,openclip_vitH14_extractS,openclip_vitH14_extractB
seed=123
predictor=sk_logistic_hypopt
data.kwargs.is_avoid_raw_dataset=True
data.kwargs.subset_raw_dataset=0.3
"



if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    #sleep 10

  done
fi

