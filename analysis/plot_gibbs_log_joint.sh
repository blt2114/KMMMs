#!/bin/sh
# this plot the log probability of the joint distribution by iteration.  The 
# log probabilities are stripped out of the log.

if [ "$#" -ne 1 ]; then
    echo  "./plot_gibbs_log_joint.sh  <log_file>"
    exit
fi

log_file=$1

grep -E "log_p joint"  < $1 | awk '{print $3}' | python analysis/plot_points.py
