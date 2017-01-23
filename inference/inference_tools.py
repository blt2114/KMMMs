""" inference_tools provides methods for MCMC inference of KMMMs"""

import sys
from collections import defaultdict, deque
from multiprocessing import Pool
import numpy as np
from numpy import flipud, fliplr
import time
from scipy import special
from inference import motif, tools
import theano
from theano import tensor as T
from theano.ifelse import ifelse

### build theano infrastructure
print("Constructing theano computational graph")
Motif = T.matrix('motif')
Kmer = T.ivector('kmer')
Offset = T.iscalar('offset')
Is_rc = T.iscalar('is rev-comp')
Motif_oriented = ifelse(T.lt(Is_rc, 1),Motif,Motif[::-1,::-1])
# offset ranges for -(motif_len-1) to (K-1)
# when d = -(motif_len-1), kmer_d = arange(0,1) and motif_d =
# arange(motif_len-1,motif_len)
# 
# when d = (K-1), kmer_d = arange(K-1,K) and motif_d = arange(0,1)
Kmer_d = Kmer[T.arange(
    T.max([0,Offset]),
    T.min([Motif.shape[0]+Offset,Kmer.shape[0]])
    )]  
Motif_d = Motif_oriented[
    T.arange(
        T.max([0,-Offset]),
        T.min([Kmer.shape[0]-Offset,Motif.shape[0]])
    )
    ,:]
Base_probs_d = Motif_d[T.arange(Motif_d.shape[0]), Kmer_d].prod()

# set up theano infrastructure for handling background probabilities.
P_kmers_bg = T.fvector('p_kmers_bg')
Bg_seq_1 = Kmer[T.arange(0,T.max([0,Offset]))]
Bg_seq_1_idx_list, updates_seq_idx_1 = theano.scan(
    lambda i, prev: prev*5 + i+1,
    sequences =[Bg_seq_1],
    outputs_info=T.zeros(1,dtype=np.int32))
Bg_seq_1_idx = Bg_seq_1_idx_list[-1]
P_bg_1 = ifelse(T.lt(T.zeros_like(Offset),Offset),P_kmers_bg[Bg_seq_1_idx][0], T.ones_like(P_kmers_bg[0],dtype=np.float32))

Bg_seq_2 = Kmer[T.arange(T.min([Motif.shape[0]+Offset,Kmer.shape[0]]),Kmer.shape[0])]
Bg_seq_2_idx_list, updates_seq_idx_2 = theano.scan(
    lambda i, prev: prev*5 + i+1,
    sequences =[Bg_seq_2],
    outputs_info=T.zeros(1,dtype=np.int32))
Bg_seq_2_idx = Bg_seq_2_idx_list[-1]
P_bg_2 = ifelse(
        T.lt(Motif.shape[0]+Offset, Kmer.shape[0]),
        P_kmers_bg[Bg_seq_2_idx][0],
        T.ones_like(P_kmers_bg[0],dtype=np.float32)
        )

P_bg = P_bg_1*P_bg_2

P_align = Base_probs_d*P_bg

### Theano code for calculating counts of bases at offsets.
Kmer_d_pad = Kmer[T.arange(
    T.max([0,Offset-1]),
    T.min([Motif.shape[0]+Offset+1,Kmer.shape[0]])
    )]
Kmer_d_pad_as_mat = Kmer_d_pad.repeat(4,0).reshape([Kmer_d_pad.shape[0],4])
Count_indices_pad = T.arange(4).repeat(Kmer_d_pad.shape[0],0).reshape([4,Kmer_d_pad.shape[0]]).T
Count_as_mat_pad =T.where(
    T.eq(Count_indices_pad,Kmer_d_pad_as_mat),
    T.ones_like(Kmer_d_pad_as_mat),
    T.zeros_like(Kmer_d_pad_as_mat)
)
Zeros_d_pad_left = T.zeros([T.max([0,-Offset+1]),4])
Zeros_d_pad_right = T.zeros([T.max([0,Motif.shape[0]-Kmer.shape[0]+Offset+1]),4])
Padded_counts = T.concatenate([Zeros_d_pad_left,Count_as_mat_pad, Zeros_d_pad_right])
Padded_counts_oriented = ifelse(
        T.lt(Is_rc,1),
        Padded_counts,
        Padded_counts[::-1,::-1]
        )

# Establish the two functions that are needed
print("compiling theano computational graph")
p_align_eval = theano.function(
    [Kmer, Motif,Offset, Is_rc, P_kmers_bg],
    P_align,
    name='eval p align'
)
padded_counts_eval = theano.function(
    [Kmer,Offset, Is_rc, Motif],
    Padded_counts_oriented,
    name='eval_padded_counts'
)
print("finished compiling theano!")

def motif_aligns_and_ps(kmer, m, m_idx, phi, kmmm):
    """motif_aligns_and_ps finds all possible aligments of the motif and the
    kmer and returns likelihoods of each alignment."""
    alignments, align_ps = m.score_all_alignments(kmer, phi)
    if kmmm.variational:
        assert m.variational
        component_term = np.exp(special.digamma(float(kmmm.theta[m_idx])/len(align_ps)))
        align_ps = [p*component_term for p in align_ps]
    else:
        assert not m.variational
        align_ps = [p*kmmm.gamma[m_idx] for p in align_ps]
    if kmer != tools.rev_comp(kmer):
        # we do not  need to do this for reverse palandromic
        # seqeunces because their counts have not been collapsed
        # with reverse complements.
        align_ps = [p*2.0 for p in align_ps]

    return align_ps, alignments

def update_counts(alignments, alignment_motif_idxs, alignment_counts, kmer,
                  motifs):
    """update_counts counts up bases aligned with each position in the
    motifs specificed by the alignments and returns the sequences assigned
    to background, the base-counts, and the counts assigned to each
    component.
    """
    motifs_alignments = {}
    for m_idx in motifs.keys():
        motifs_alignments[m_idx] = [(alignment, alignment_counts[i]) for i, alignment in
                enumerate(alignments) if alignment_motif_idxs[i] == m_idx]

    component_counts = defaultdict(int)
    bg_seq_counts = defaultdict(int)
    eta = {}
    for m_idx, m in motifs.iteritems():
        eta[m_idx] = np.array(m.eta) * 0.0

    # update bg component stats
    bg_counts = alignment_counts[-1]
    component_counts[tools.BG_IDX] += bg_counts
    bg_seq_counts[kmer] += bg_counts

    # update motif component stats.
    for m_idx, motif_alignments in motifs_alignments.iteritems():
        m = motifs[m_idx]
        for (seq_st, motif_st, rev_comp), count_a  in motif_alignments:
            if count_a == 0:
                continue
            component_counts[m_idx] += count_a
            # get alignment length
            l = min(len(kmer) - seq_st, m.length() - motif_st)

            # get observed bases for alignment and background sequences
            (bg_seg_1, bg_seg_2, base_counts) = m.add_alignment(
                kmer, seq_st, motif_st, l, count=count_a, is_rc=rev_comp
                )
            eta[m_idx] += base_counts
            if bg_seg_1 != None:
                bg_seq_counts[bg_seg_1] += count_a
            if bg_seg_2 != None:
                bg_seq_counts[bg_seg_2] += count_a
    return bg_seq_counts, eta, component_counts

def iteration_kmer(kmer, count, motifs, phi, kmmm):
    """ iteration_kmer performs the update/ sampling step for the kmer
    provided.

    This is done either done using variational updates or by sampling.

    Args:
        kmer: the kmer to do the update for.
        count: the number of occurences of the kmer
        motifs: dictionary mapping motif indices to motif objects.
        phi: background model object
        kmmm: mixture model object

    Returns:
        p_kmer: the probability of observing the kmer as an arbitrary
            subsequence of the provided kmer's length
        eta: dictionary of base counts, mapping each motif idx to an
            np.array of counts.
        bg_seq_counts: counts of each sequence attributed to the background
            component.
        component_counts: dictionary mapping component idxs to the number of
            kmers assigned to them.
    """
    align_ps = []
    kmer_vec = tools.kmer_as_vec(kmer)
    p_kmers_bg = phi.kmer_probs_array
    motif_align_ranges = {}
    # get the probabilities for all alignments in the motif component.
    if kmer != tools.rev_comp(kmer):
        # we do not  need to do this for reverse palandromic
        # seqeunces because their counts have not been collapsed
        # with reverse complements.
        palandrome_factor = 2.0
    else:
        palandrome_factor = 1.0
    range_start = 0
    for m_idx, m in motifs.iteritems():
        beta = np.array(m.beta)[1:-1] # don't include padding
        all_ds = np.arange(-(beta.shape[0]-1),len(kmer))
        all_is_rc = np.array([0]*len(all_ds)+[1]*len(all_ds), dtype=np.int32)
        all_ds = np.concatenate([all_ds,all_ds])
        p_offsets = np.array([p_align_eval(kmer_vec, beta, d, is_rc, p_kmers_bg) for
                d, is_rc in zip(all_ds, all_is_rc)])
        p_offsets /= len(p_offsets)
        p_offsets *= kmmm.gamma[m_idx]*palandrome_factor
        align_ps.extend(p_offsets)
        motif_align_ranges[m_idx] = (range_start,range_start+len(p_offsets))
        range_start += len(p_offsets)

    bg_prob = phi.prob(kmer)*kmmm.gamma[tools.BG_IDX]

    # Since we lump reverse complements.
    # by the same logic as for the motif probs, this does not need
    # to be doubled for reverse palandromic sequences.
    bg_prob *= palandrome_factor
    align_ps.append(bg_prob)
    p_kmer = sum(align_ps)

    align_ps /= sum(align_ps) # normalize before sampling

    if kmmm.variational:
        sys.stderr.write("variational not supported currently\n")
        sys.exit(1)
        alignment_counts = count*align_ps
    else: # Draw the alignments
        alignment_counts = np.random.multinomial(count, align_ps)

    ### calculate log_p of multinomial draw
    log_p_align_counts = special.gammaln(count+1.0)
    for i, count_align in enumerate(alignment_counts):
        log_p_align_counts -= special.gammaln(count_align+1.0)
        log_p_align_counts += count_align * np.log(align_ps[i])

    component_counts = defaultdict(int)
    bg_seq_counts = defaultdict(int)
    eta = {}
    for m_idx, m in motifs.iteritems():
        (begin, out) = motif_align_ranges[m_idx]
        counts = alignment_counts[begin:out]
        beta = np.array(m.beta)[1:-1] # don't include padding
        all_ds = np.arange(-(beta.shape[0]-1),len(kmer))
        all_is_rc = np.array([0]*len(all_ds)+[1]*len(all_ds), dtype=np.int32)
        all_ds = np.concatenate([all_ds,all_ds])
        motif_counts_kmer = np.sum([
            count*padded_counts_eval(kmer_vec,d,is_rc,beta) for d, is_rc, count in zip(all_ds,all_is_rc,counts)
            ], axis=0)
        eta[m_idx] = motif_counts_kmer
        component_counts[m_idx] = sum(counts)
    bg_seq_counts[kmer] = alignment_counts[-1]
    return p_kmer, eta, bg_seq_counts, component_counts, log_p_align_counts

def iteration_chunk((kmers, kmer_counts, motifs, phi, kmmm)):
    eta = {}
    for m_idx, m in motifs.iteritems():
        eta[m_idx] = np.array(m.eta) * 0.0
    bg_seq_counts = defaultdict(int)
    component_counts = defaultdict(int)
    log_p_kmer_aligns = {}
    p_kmers = {}
    sys.stderr.write("running")
    for i, kmer in enumerate(kmers):
        sys.stderr.write("\revaluating %6d/%6d"%(i,len(kmers)))
        p_kmer, eta_k, bg_seq_counts_k, component_counts_k, log_p_kmer_align = iteration_kmer(
            kmer, kmer_counts[kmer], motifs, phi, kmmm
            )
        p_kmers[kmer] = p_kmer
        log_p_kmer_aligns[kmer] = log_p_kmer_align
        for m_idx in motifs.keys():
            eta[m_idx] += eta_k[m_idx]
        for compnt, count_c in component_counts_k.iteritems():
            component_counts[compnt] += count_c
        for seq, count in bg_seq_counts_k.iteritems():
            bg_seq_counts[seq] += count
    sys.stderr.write("finished\n")
    return p_kmers, eta, bg_seq_counts, component_counts, log_p_kmer_aligns

def iteration(phi, motifs, kmmm, stochastic=False, batch_size=None,
              verbose=False, f=sys.stdout, n_jobs=1, bg_update=True):
    """ iteration runs an iteration of the gibbs sampler.

    Args:
        phi: the background model, a dictionary mapping from kmer to their
            frequencies.
        motifs:
            dictionary mapping from motif indices to motif objects.
        kmmm: an instance of tools.KMMM, containing the kmer table and
            mixture proportions.
        stochastic: true if only a subset of kmer will be used.
        batch_size: number of samples to use in each batch when creating
                estimate for variational updates.
        verbose: True if we wish to log nonvital information
        f: filehandle to which logs are written
        n_jobs: number of processes to split kmers across.
        bg_update: set to true if we don't want to update the background
    """
    if stochastic: assert batch_size is not None
    kmer_counts = kmmm.kmer_counts
    kmmm.update_gamma()
    kmmm.clear_counts()
    phi.clear_counts()
    phi.clear_seqs()

    for m_idx, m in motifs.iteritems():
        m.clear_alignments()
    # for now just one motif.
    component_counts = defaultdict(int)
    batch = np.array(kmer_counts.keys())
    if stochastic: batch = batch[np.random.permutation(len(batch))[:batch_size]]

    eta = {}
    for m_idx, m in motifs.iteritems():
        eta[m_idx] = np.array(m.eta)*0.0
    bg_seq_counts = defaultdict(int)
    # we want a large number of chunks so that can be split well if one or
    # more of the processes doesn't get much runtime.
    chunks = np.array_split(batch, n_jobs*4)
    fn = f.name 
    f.close()
    args = [(chunk, kmer_counts, motifs, phi, kmmm) for chunk in chunks]
    if n_jobs != 1:
        p = Pool(n_jobs)
        returns = p.map(iteration_chunk, args)
    else:
        returns = []
        for chunk in chunks:
            returns.append(iteration_chunk((chunk, kmer_counts, motifs, phi, kmmm)))
    for (p_kmers_c, eta_c, bg_seq_counts_c, component_counts_c,
            log_p_kmer_aligns) in returns:
        for kmer, p in p_kmers_c.iteritems():
            kmmm.add_p_kmer(kmer, p) # track of the prob of each k-mer
            kmmm.add_p_kmer_aligns(kmer, log_p_kmer_aligns[kmer]) # track of the prob of each k-mer
        for m_idx in motifs.keys():
            eta[m_idx] += eta_c[m_idx]
        for compnt, count_c in component_counts_c.iteritems():
            component_counts[compnt] += count_c
        for seq, count in bg_seq_counts_c.iteritems():
            bg_seq_counts[seq] += count

    f = open(fn,'a')
    # make sure that the probability of falling under all the components
    # is indeed equal to 1.0, (or within rounding error of it).
    p_all = sum(kmmm.p_kmers.values())
    if verbose:
        tools.log("p_all: %s"%str(p_all), f)
    if not kmmm.variational:
        assert abs(p_all - 1.0) < 0.1

    if bg_update:
        phi.add_seqs(bg_seq_counts)
        phi.update_kmer_counts()
        phi.update()
    for m_idx, m in motifs.iteritems():
        m.eta = deque(eta[m_idx])
        if isinstance(m, motif.DynamicMotif):
            m.update_pwm(phi=phi)
        else:
            m.update_pwm()

    kmmm.update_counts(component_counts)
    f.close()

def motifs_log(motifs, gamma):
    """ motifs_log create a logging string that describes the motifs

    Args:
        motifs: dictionary of motifs
        gamma: the motif proportions

    Returns:
        a printable string that describes the state of the motifs.
    """
    motifs_str = ""
    for m_idx, m in motifs.iteritems():
        if m_idx == tools.NEW_IDX:
            continue
        motifs_str += m.string(gamma_m=gamma[m_idx], idx=m_idx)
    return motifs_str

def log_state(phi, motifs, kmmm, stochastic=False, f=sys.stdout):
    """ log_state logs information about the log probabilities of the latent
    variables and joint distribution.

    Args:
        phi: the background model, a dictionary mapping from kmer to their
            frequencies.
        motifs:
            dictionary mapping from motif indices to motif objects.
        kmmm: an instance of tools.KMMM, containing the kmer table and
            mixture proportions.

    Returns:
        the log probability of the joint distribution.
    """
    if stochastic:
        log_p_multinomial = 0.0
        log_p_kmer_multinomials = 0.0
    else:
        log_p_multinomial = tools.log_p_kmer_table(kmmm.kmer_counts, kmmm.p_kmers)
        log_p_kmer_multinomials = sum(kmmm.log_p_kmer_aligns.values())
    log_p_gamma = kmmm.log_prob_given_alpha()
    log_p_beta = sum([m.log_prob_given_pi() for m in motifs.values()])
    log_p_joint = log_p_multinomial + log_p_kmer_multinomials + log_p_gamma + log_p_beta

    tools.log("component counts: %s"%str(kmmm.component_counts), f)
    tools.log("gamma: %s"%str(kmmm.gamma), f)
    if stochastic:
        tools.log("learning rate: %f"%kmmm.learning_rate(), f)
    tools.log(motifs_log(motifs, kmmm.gamma), f)
    tools.log("\nlog_p joint: %s"%str(log_p_joint), f)
    tools.log("\tlog_p gamma: %s"%str(log_p_gamma), f)
    tools.log("\tlog_p beta: %s"%str(log_p_beta), f)
    tools.log("\tlog_p multinomial: %s"%str(log_p_multinomial), f)
    tools.log("\tlog_p K-mer multinomials: %s"%str(log_p_kmer_multinomials), f)
    return log_p_joint

def initialize_motifs(n_motifs, init_count=0, seq_len=7, f=sys.stdout,
                      pi_mag=None, log_p_longer=-15, min_len=6, max_len=10,
                      variational=False, phi=None, verbose=False, kappa=0,
                      tau=None, sym_motifs=False, K=None):
    """initialize_motifs creates random motifs and adds them to the
    mixuture.

    Args:
        n_motifs: number of motifs to initialize
        seq_len: initial motif length
        f: file to log to
        pi_mag: dirichlet prior on motif parameters
        log_p_longer: prior probability of motif being longer than its
            current length.
        min_len: minimum length motifs can shrink to
        variational: true to initialize variational motifs.
        phi: background model, if none, 0.25 prob of each base is assumed
        verbose: true if log verbosely in motifs code
    """
    motifs = {}
    motif_idx = 0
    for _ in range(n_motifs):
        seq = "".join(tools.BASES[np.random.randint(len(tools.BASES))] for _ in
                      range(seq_len))
        motif_idx += 1
        if sym_motifs:
            motifs[motif_idx] = motif.SymmetricMotif(
                seq_len,
                pi_mag=pi_mag,
                variational=variational,
                verbose=verbose,
                kappa=kappa,
                tau=tau,
                min_len=min_len,
                max_len=max_len,
                f=f,
                log_p_longer=log_p_longer,
                K=K
                )
        else:
            motifs[motif_idx] = motif.DynamicMotif(
                seq_len,
                pi_mag=pi_mag,
                variational=variational,
                verbose=verbose,
                kappa=kappa,
                tau=tau,
                min_len=min_len,
                max_len=max_len,
                f=f,
                log_p_longer=log_p_longer,
                K=K
                )
        if variational:
            motifs[motif_idx].add_noise_to_var_dist()

        _, _, base_counts = motifs[motif_idx].add_alignment(seq, 0, 0, len(seq), init_count)
        motifs[motif_idx].eta = deque(np.array(motifs[motif_idx].eta)
                                      + base_counts)
        motifs[motif_idx].update_pwm(phi)
    return motifs

def load_motifs_from_file(n_motifs, init_count=0, seq_len=7, f=sys.stdout,
                      pi_mag=None, log_p_longer=-15, min_len=6, max_len=10,
                      variational=False, phi=None, verbose=False, kappa=0,
                      tau=None, sym_motifs=False, K=None, out_dir=None,
                      restart_iter=None):
    """load_motifs_from_file initializes in the same manner as
    initialize_motifs, and then replaces signal with saved motifs.

    """
    motifs = initialize_motifs(n_motifs, init_count, seq_len, f, pi_mag,
            log_p_longer, min_len, max_len, variational, phi, verbose,
            kappa, tau, sym_motifs, K)

    motif_idx = 0
    for m in motifs.keys():
        fn = out_dir + "motif%02d_iteration%05d.psam"%(m,restart_iter)
        pwm=tools.load_motif_from_PSAM(fn)
        motifs[m].beta = deque(pwm)
        motifs[m].motif_len = len(pwm)-2
        motifs[m].eta = deque(np.zeros(pwm.shape))
        motifs[m].beta_rc = deque(flipud(fliplr(pwm)))
        motifs[m].iteration = restart_iter
        motifs[m].iteration = restart_iter
        if variational:
            motifs[m].lmbda = deque(pwm*100+pi_mag)
            motifs[m].lmbda_rc = deque(pwm*100+pi_mag)
    return motifs

def unseen_motif(f=sys.stdout, pi_mag=0.6, seq_len=7, log_p_longer=-15,
                 min_len=6, max_len=10, phi=None, sym_motifs=False,
                 verbose=False):
    """unseen_motif creates a motif object that stands in as un unseen
    component.

    The new motifs pwm must be uniform.  So that the sampling of kmers into
    this component is not biased.

    This is meant specifically for when we are using an unseen motif with the
    Chinese Restaurant Process.

    Args:
        f: file to log to
        pi_mag: dirichlet prior on motif parameters
        seq_len: lenght of motifs
        log_p_longer: prior probability of motif being longer than its
            current length.
        min_len: minimum length motifs can shrink to
        phi: background model, if none, 0.25 prob of each base is assumed
    """
    if sym_motifs:
        m_new = motif.SymmetricMotif(
            seq_len,
            pi_mag=pi_mag,
            min_len=min_len,
            variational=False,
            verbose=verbose,
            kappa=0.0,
            max_len=max_len,
            f=f,
            log_p_longer=log_p_longer
            )
    else:
        m_new = motif.DynamicMotif(
            seq_len,
            pi_mag=pi_mag,
            min_len=min_len,
            variational=False,
            verbose=verbose,
            kappa=0.0,
            max_len=max_len,
            f=f,
            log_p_longer=log_p_longer
            )
    m_new.update_pwm(phi)
    # we must set to equal probability of every base so that it is not
    # biased
    m_new.beta = deque(0.25*np.ones(np.array(m_new.beta).shape))
    if m_new.variational:
        motif.VariationalMotif.update_rc(m_new)
    else:
        motif.Motif.update_rc(m_new)
    return m_new

def save_motifs(motifs, root_dir, name_base, phi, kmmm):
    """save_motifs writes the motifs givenas PSAMS and as strings to files
    in the directory provided.

    Args:
        motifs: a list of motif objects
        root_dir: full path of directory to write files to as a string.
    """
    # save motif description as text file first
    gamma = kmmm.gamma
    component_counts = kmmm.component_counts
    f = open(root_dir +"/"+name_base+"motifs.txt", "w")
    f.write(motifs_log(motifs, gamma))
    f.close()

    for m_idx, m in motifs.iteritems():
        if m_idx == tools.NEW_IDX:
            continue
        fn = root_dir +"/"+"motif%02d_"%m_idx+name_base+".psam"
        m.save_as_psam(fn)
