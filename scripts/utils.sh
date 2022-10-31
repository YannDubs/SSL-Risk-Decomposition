#!/usr/bin/env bash

add_kwargs=""
prfx=""
time="2880" # 2 days
is_plot_only=false
server=""
mode=""
main="main_tuneall.py"

# MODE ?
while getopts ':s:p:t:v:a:i' flag; do
  case "${flag}" in
    p )
      prfx="${OPTARG}"
      echo "prefix=$prfx ..."
      ;;
    s )
      server="${OPTARG}"
      add_kwargs="${add_kwargs} server=$server"
      echo "$server server ..."
      ;;
    i )
      add_kwargs="${add_kwargs} hydra.launcher.partition=jag-hi"
      echo "High priority mode"
      ;;
    v )
      is_plot_only=true
      prfx=${OPTARG}
      echo "Visualization/plotting only ..."
      ;;
    t )
      time=${OPTARG}
      echo "Time ${OPTARG} minutes"
      ;;
    a )
      add_kwargs="${add_kwargs} ${OPTARG}"
      echo "Adding ${OPTARG}"
      ;;
    \? )
      echo "Usage: "$name".sh [-scmvtai]"
      exit 1
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      ;;
  esac
done

experiment="${prfx}""$experiment"
results="results/exp_$experiment"
tuning="tuning/exp_$experiment"
pretrained="pretrained/exp_$experiment"
checkpoints="checkpoints/exp_$experiment"
logs="logs/exp_$experiment"

#if [[ "$is_plot_only" = false && "$mode" != "continue" ]] ; then
#  if [[ -d "$checkpoints" || -d "$logs" || -d "$tuning" || -d "$pretrained" ]]; then
#
#    echo -n "$checkpoints and/or pretrained/... exist and/or logs/... exist. Should I delete them (y/n) ? "
#    read answer
#
#    if [ "$answer" != "${answer#[Yy]}" ] ;then
#        echo "Deleted $pretrained"
#        rm -rf $pretrained
#        echo "Deleted $checkpoints"
#        rm -rf $checkpoints
#        echo "Deleted $logs"
#        rm -rf $logs
#        echo "Deleted $tuning"
#        rm -rf $tuning
#    fi
#  fi
#fi

# make sure that result folder exist for when you are saving a hypopt optuna database
mkdir -p $results