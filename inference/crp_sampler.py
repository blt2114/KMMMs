""" crp_sampler runs a Gibbs Samper to fit a KMMM, finding the number of
motifs dynamically by sampling from the chinese restaurant process (CRP) """

import sys
from collections import deque
import numpy as np
from inference import motif, tools, background, inference_tools
import argparse

def main():
    ### establish arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('-kmer_table', dest="kmer_table_fn",action="store",
            help="path to the Kmer table", type=str, required= True)
    parser.add_argument('-I', dest="I",action="store",
            help="number of iterations",type=int, default = 1000)
    parser.add_argument('-K', dest="K",action="store", help="Kmer length",
            type=int, default = 7)
    parser.add_argument('-o', dest="output_dir",action="store",
            help="path to output directory",type=str, required=True)
    parser.add_argument('-log', dest="log_file",action="store",
            help="name of log file, if not given, stdtout is used", type=str,
            default="stdout")
    parser.add_argument("-n_jobs",dest="n_jobs",action="store",
            help="number of jobs to distribute across",type=int,
            default=1)
    parser.add_argument("-bg_markov",dest="bg_markov_len",action="store",
            help="the order of the markov chain used to model background sequence",
            type=int, default=2)
    parser.add_argument("-min_len",dest="min_len",action="store",
            help="the minimum length of motifs",
            type=int, default=6)
    parser.add_argument("-max_motifs",dest="max_motifs",action="store",
            help="the maximum number of motifs to learn",
            type=int, default=6)
    parser.add_argument("-alpha_crp",dest="alpha_crp",action="store", 
            help="the size the crp prior", type=float, default=1.0)
    parser.add_argument("-pi",dest="pi",action="store", help="the size the"+
            "uniform dirichlet prior on the base distributions of motifs",
            type=float, default=1.0)
    parser.add_argument('-log_p_longer', dest="log_p_longer", action="store",
            help="log_probability of extending motif",type=float,default=-25.0)
    parser.add_argument("-boost",dest="boost",action="store", help="when a"+
            " new motif is created, the count is inflated by this many counts",
            type=float, default=1.0)
    parser.add_argument("-sym_motifs", dest="sym_motifs", action="store_true",
            help="True if we are to enforce motif symmetry")
    parser.add_argument("-bg_static", dest="bg_static", action="store_true",
            help="use static, fully random background. this hugely"
            " increases speed of iteration")
    ### Parse and process args
    try:
        args = parser.parse_args()
    except IOError, msg:
        parser.error(str(msg))

    kmer_table_fn = args.kmer_table_fn
    n_iter = args.I
    output_dir = args.output_dir + "/"
    log_fn = output_dir+ args.log_file
    if log_fn == "stdout":
        f = sys.stdout
    else:
        f = open(log_fn, "w")
    n_jobs = args.n_jobs
    pi_mag = args.pi
    bg_markov_len = args.bg_markov_len
    boost = args.boost
    bg_static = True if args.bg_static else False
    alpha_crp = args.alpha_crp
    K = args.K
    min_len = args.min_len
    sym_motifs=args.sym_motifs
    log_p_longer = args.log_p_longer
    max_motifs = args.max_motifs


    ### Write arguments to log file for records
    tools.log("arguments: "+str(args),f)

    n_motifs = 2
    init_count = boost
    save_frequency = 50
    verbose = False

     
    ### Initialize motifs and KMMM
    motifs = inference_tools.initialize_motifs(n_motifs, init_count,f=f, 
            pi_mag= pi_mag, seq_len=min_len, min_len=min_len, 
            sym_motifs=sym_motifs, log_p_longer=log_p_longer,
            verbose=verbose)
    motif_idx = n_motifs # index of the most recently added motif


    tools.log("loading in kmer table",f)
    kmmm = tools.CRP_KMMM(kmer_table_fn, motifs, alpha_crp=alpha_crp,
                          max_components=max_motifs)
    for idx in motifs.keys():
        kmmm.component_counts[idx] = init_count

    # add in unseen component place-holder
    motifs[tools.NEW_IDX] = inference_tools.unseen_motif(f,pi_mag=pi_mag,
            seq_len=min_len,min_len=min_len, sym_motifs=sym_motifs,
            log_p_longer=log_p_longer, verbose=verbose)

    ### Initialize background model
    tools.log("creating background model",f)
    phi = background.Phi(prior_size=100, chain_len=bg_markov_len,static=bg_static,k =K)
    phi.add_seqs(kmmm.kmer_counts)
    phi.update_kmer_counts()
    phi.update()
   
    # seed motif.
#    _, _, base_counts = motifs[1].add_alignment("CACGTG",0,0,6,500)
#    kmmm.component_counts[1] += 2000
#    motifs[1].eta = deque(np.array(motifs[1].eta)+base_counts)
#    motifs[1].update_pwm(phi)

    tools.log("Beginning Sampler",f)
    for i in range(n_iter):
        tools.log("\nIteration %d"%(i+1),f)
        inference_tools.iteration(phi, motifs, kmmm, f=f, verbose=verbose, n_jobs=n_jobs)
        f = open(log_fn, 'a')

        if kmmm.component_counts[tools.NEW_IDX] != 0:
            # If anything has been assigned to the unseen component, we must add it
            motif_idx += 1
            motifs[motif_idx] = motifs.pop(tools.NEW_IDX)
            kmmm.component_counts[motif_idx] = kmmm.component_counts.pop(tools.NEW_IDX)
            kmmm.component_counts[motif_idx] += boost

        # we must also eliminate components that no longer have kmers 
        # assigned to them.
        empty_components = kmmm.remove_empty_components()
        for m_idx in empty_components:
            motifs.pop(m_idx)

        kmmm.update_gamma()
        inference_tools.log_state(phi, motifs, kmmm, f=f)
        motifs[tools.NEW_IDX] = inference_tools.unseen_motif(f,pi_mag=pi_mag,
                seq_len=min_len,min_len=min_len,sym_motifs=sym_motifs,
                log_p_longer=log_p_longer, verbose=verbose)
        kmmm.component_counts[tools.NEW_IDX] = 0
        f.flush()
        if i%save_frequency == 0:
                inference_tools.save_motifs(motifs, output_dir, 
                        "_iteration%05d"%i, phi, kmmm)

    inference_tools.save_motifs(motifs, output_dir, "_final",phi,kmmm)
    f = open(log_fn, 'a')
    tools.log("Sampling concluded",f)
    if f != sys.stdout:
        f.close()

if __name__ == "__main__":
    main()
