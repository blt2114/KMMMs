#!/bin/bash

# this script run a gibbs sampler to fit motifs to synthetic data

output_dir="example/out2"
mkdir $output_dir
python inference/gibbs_sampler.py \
    -kmer_table example/dat/kmer_table_condensed.txt \
    -I 1000  \
    -o $output_dir \
    -log log \
    -n_jobs 4 \
    -pi 5.0 \
    -bg_static \
    -n_motifs 2 \
    -alpha_bg 100000 \
    -alpha_m 6000 \
    -log_p_longer -100.0
