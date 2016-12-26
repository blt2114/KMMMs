""" gibbs_sampler runs a Gibbs Samper to fit a KMMM
"""

import argparse
import sys
from inference import tools, background, inference_tools

def main():
    ### establish arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('-kmer_table', dest="kmer_table_fn", action="store",
                        help="path to the Kmer table", type=str, required=True)
    parser.add_argument('-I', dest="I", action="store",
                        help="number of iterations", type=int, default=1000)
    parser.add_argument('-o', dest="output_dir", action="store",
                        help="path to output directory", type=str,
                        required=True)
    parser.add_argument('-log', dest="log_file", action="store",
                        help="name of log file, if not given, stdout is used",
                        type=str, default="stdout")
    parser.add_argument("-n_jobs", dest="n_jobs", action="store",
                        help="number of jobs to distribute across", type=int,
                        default=1)
    parser.add_argument("-min_len", dest="min_len", action="store",
                        help="the minimum length of motifs", type=int,
                        default=6)
    parser.add_argument("-max_len", dest="max_len", action="store",
                        help="the minimum length of motifs", type=int,
                        default=10)
    parser.add_argument("-bg_markov", dest="bg_markov_len", action="store",
                        help=("the order of the markov chain used to model "
                              "background sequence"), type=int, default=2)
    parser.add_argument("-n_motifs", dest="n_motifs", action="store",
                        help="the number of motifs to learn", type=int,
                        default=3)
    parser.add_argument("-pi", dest="pi", action="store",
                        help=("the size the uniform dirichlet prior on the"
                              " base distributions of motifs"), type=float,
                        default=1.0)
    parser.add_argument("-alpha_bg", dest="alpha_bg", action="store",
                        help=("the size of the dimension of dirichlet prior "
                              "for background"), type=float, default=10000.0)
    parser.add_argument("-alpha_m", dest="alpha_m", action="store",
                        help=("the size of the dimension of dirichlet prior"
                              " for motifs"), type=float, default=10.0)
    parser.add_argument("-bg_static", dest="bg_static", action="store_true",
                        help=("use static, fully random background. this hugely"
                              " increases speed of iteration"))
    parser.add_argument("-sym_motifs", dest="sym_motifs", action="store_true",
                        help="use reverse complement symmetric motifs.")
    parser.add_argument('-bg_update_freq', dest="bg_update_freq", action="store",
                        help="update the background at this frequency", 
                        type=int, default=1)
    parser.add_argument('-K', dest="K", action="store",
                        help="Kmer length", type=int, default=7)
    parser.add_argument('-log_p_longer', dest="log_p_longer", action="store",
                        help="log_probability of extending motif", type=float,
                        default=-25.0)
    parser.add_argument('-restart_iter', dest="restart_iter", action="store",
                        help="restart using data from this iteration",
                        type=int, default=0)

    ### Parse and process args
    try:
        args = parser.parse_args()
    except IOError, msg:
        parser.error(str(msg))

    save_frequency = 10
    kmer_table_fn = args.kmer_table_fn
    n_iter = args.I
    output_dir = args.output_dir + "/"
    log_fn = output_dir+ args.log_file
    restart_iter = args.restart_iter
    if args.log_file == "stdout":
        f = sys.stdout
    else:
        if restart_iter == 0:
            f = open(log_fn, "w")
        else:
            f = open(log_fn, "a")
            f.write("resatarting at iteration %d\n"%restart_iter)
    n_jobs = args.n_jobs
    bg_markov_len = args.bg_markov_len
    bg_update_freq = args.bg_update_freq
    pi_mag = args.pi
    bg_static = True if args.bg_static else False
    K = args.K
    log_p_longer = args.log_p_longer

    alpha_bg = args.alpha_bg
    alpha_m = args.alpha_m
    n_motifs = args.n_motifs
    min_len = args.min_len
    max_len = args.max_len

    sym_motifs = args.sym_motifs

    ### Write arguments to log file for records
    tools.log("arguments: "+str(args), f)

    ### Initialize motifs, mixture and background objects.
    if restart_iter == 0:
        motifs = inference_tools.initialize_motifs(n_motifs, f=f, pi_mag=pi_mag,
                log_p_longer=log_p_longer, seq_len=min_len, min_len=min_len,
                max_len=max_len, sym_motifs=sym_motifs, K= K)
    else:
        motifs = inference_tools.load_motifs_from_file(n_motifs, f=f, pi_mag=pi_mag,
                log_p_longer=log_p_longer, seq_len=min_len, min_len=min_len,
                max_len=max_len, sym_motifs=sym_motifs, K= K,
                out_dir=output_dir, restart_iter=restart_iter)

    ### Load Kmer table
    tools.log("loading in kmer table", f)
    kmmm = tools.KMMM(kmer_table_fn, motifs, alpha_bg=alpha_bg,
            alpha_m=alpha_m, K=K)


    tools.log("creating background model", f)
    # the prior size give the Dirichlet prior on the highest order
    # preferences defined by the Markov chain defining the background.  We
    # use 100.  This acts similarly to having a 'pseudo-count' of 100 of
    # each K-mer of length 'bg_markov_len' + 1.
    phi = background.Phi(prior_size=100, chain_len=bg_markov_len,
                         static=bg_static, k=K)
    phi.add_seqs(kmmm.kmer_counts)
    phi.update_kmer_counts()
    phi.update()

    ### Begin inference by Gibbs sampling
    tools.log("Beginning Sampler", f)
    for i in range(restart_iter+1, n_iter):
        tools.log("\nIteration %d"%(i), f)
        inference_tools.iteration(phi, motifs, kmmm, f=f, verbose=True,
                                  n_jobs=n_jobs, bg_update=i%bg_update_freq==0)
        f = open(log_fn, 'a')
        inference_tools.log_state(phi, motifs, kmmm, f=f)
        kmmm.update_gamma()
        f.flush()
        if i%save_frequency == 0: # occasionally save motifs as PSAMs
            inference_tools.save_motifs(motifs, output_dir, "iteration%05d"%i,
                                        phi, kmmm)

    # Save the final state of the motifs
    inference_tools.save_motifs(motifs, output_dir, "final", phi, kmmm)
    f = open(log_fn, 'a')
    tools.log("Sampling concluded", f)
    if f != sys.stdout:
        f.close()

if __name__ == "__main__":
    main()
