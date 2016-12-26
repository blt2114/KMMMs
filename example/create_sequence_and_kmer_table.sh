#!/bin/sh
# create_sequence_and_kmer_table.sh generates synthetic sequence and creates a K-mer table from it.

python example/gen_synthetic_sequence.py > example/dat/synthetic.fa

data_prep/create_kmer_table 7 example/dat/synthetic.fa >  example/dat/kmer_table.txt

python data_prep/condense_kmer_table_with_reverse_complements.py example/dat/kmer_table.txt>  example/dat/kmer_table_condensed.txt
