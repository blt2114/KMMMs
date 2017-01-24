"""tools is meant to be a depository of general tools used in KMMMs.

Such functions include.
    - reading Kmer tables
    - working with sequences (reverse complements, encoding)
    - handling indices and offsets?

This also contains a class, KMMM for mixtures.

"""
from collections import defaultdict
import sys
import math
import numpy as np
from  scipy import special

BASES = ['A', 'C', 'G', 'T']
IDX = {}
for i, b in enumerate(BASES):
    IDX[b] = i
BG_IDX = 0 # index for Background component
NEW_IDX = -1 # index for unseen component for sampling with CRP
PAIRS = {
    'A':'T',
    'T':'A',
    'C':'G',
    'G':'C'
    }

def read_kmer_table(filename):
    """ read_kmer_table reads a kmer table into memory

    Args:
        fn: the name of a file containing the counts of every kmer.  The
        table may or may not be complete.  Each kmer is separated by a
        newline, with a tab separating hte sequence and its count.

    Returns:
        a default dict mapping sequences to their count.
    """
    table = defaultdict(int)
    kmer_table_file = open(filename)
    for line in kmer_table_file:
        kmer, count = line.strip("\n").split()
        table[kmer] = int(count)
    kmer_table_file.close()
    return table

def load_motif_from_PSAM(fn):
    """load_motifxs_from_PSAM read and xml format PSAM and returns np.array
    containing a pwm representation of the motif.

    it is assumed that distribution for the first position is stored on the
    4th line.

    Args:
        fn: filename of the PSAM

    Returns:
        np.array containing the pwm.
    """
    f = open(fn)
    # throw away the first 3 lines
    for _ in range(3):
        f.readline()

    pwm = []
    for l in f:
        dist = l.split("\t")
        if len(dist) != 4:
            break
        vals = [float(v) for v in dist]
        vals = np.array(vals)
        vals /= sum(vals)
        pwm.append(vals)
    f.close()
    pwm = np.array(pwm)
    return pwm

def one_hot_encoding(kmer):
    """ one_hot_encoding encodes a string representation of a Kmer.

    Args:
        kmer: the sequence to be encoded

    Returns:
        an np.array containing a one hot encoding of the sequence.  This can
            be scored against PWMs by taking a dot product.
    """
    encoding = np.zeros([len(kmer), len(BASES)])
    for i, base in enumerate(kmer):
        encoding[i, IDX[base]] = 1.0
    return encoding

def rev_comp(kmer):
    """rev_comp uses biopython ot take a sequence's reverse complement

    Returns:
        The reverse complement as a string

    """
    rc = ""
    for b in kmer:
        rc = PAIRS[b] + rc
    return rc

def kmer_to_idx(seq):
    """ kmer_to_idx maps a sequence to an integer representation.

    This defines a useful bijective mapping that allows a kmer table to be
    stored in an array without explicitly storing the kmer as a key.

    Args:
        seq: string reresentation

    Returns:
        integer idx
    """
    idx = 0
    for base in seq:
        idx = idx<<2
        idx += IDX[base]
    return idx

def idx_to_kmer(idx):
    """ idx_to_kmer is the inverse mapping of kmer_to_idx.

    Args:
        the kmer encoded as an integer

    Return:
        the kmer as a string
    """
    kmer = ""
    while idx != 0:
        kmer = BASES[idx%4] + kmer
        idx = idx>> 2
    return kmer

def all_kmers(k, k_so_far=None):
    """all_kmers generates a list a k base long sequences of bases
    recursively and returns the list.

    Args:
        k: the length of the kmers
        k_so_far: the list of sequences onto which sequences of length k
            will be appended.  When this method is being used to genenerate
            7-mers, in the 3rd iteration, each member of this list will be
            length 3, and k will be 4.  In this first iteration, this is set
            to None.
    Returns:
        The list of sequences consistenting of all kmers appended onto each
        member of of k_so_far.
    """
    if k_so_far is None:
        k_so_far = [""]
    if k == 0:
        return k_so_far

    k_so_far_longer = []
    for kmer in k_so_far:
        for base in BASES:
            k_so_far_longer.append(kmer+base)

    return all_kmers(k-1, k_so_far_longer)

def subsequences(seq, k=None):
    """subsequences returns a list of all subsequences of length up to K
    which are present in seq.

    Subsequences that occur multiple times will be repeated in this list.
    """
    seqs = []
    if k is None:
        k = len(seq)

    for kmer_len in range(1, k+1):
        seqs.extend([seq[idx:idx+kmer_len] for idx in range(len(seq)-kmer_len+1)])
    return seqs

def log_p_kmer_table(kmer_counts, p_kmers):
    """log_p_kmer_table calculates the probabilty of an observed kmer table
    given as a multinomial given the multinomial probabilities.  The kmer
    table is preferably collapsed, i.e. if a given sequence is present, its
    reverse complement is not.

    Args:
        kmer_counts: the kmer table
        p_kmers: the multinomial probabilities

    Returns:
        the total log probability as a float
    """
    total = 0
    log_p = 0
    total_p = sum(p_kmers.values())
    for kmer in p_kmers.keys():
        p_kmers[kmer] /= total_p
    for kmer, count in kmer_counts.iteritems():
        log_p += count*np.log(p_kmers[kmer]) - special.gammaln(count+1)
        total += count
    log_p += special.gammaln(total+1)
    return log_p

def nCk(n, k):
    """ c choose k"""
    count = math.factorial(n)/math.factorial(n-k)/math.factorial(k)
    return count

def log_nCk(n, k):
    """log of  c choose k"""
    log_count = 0.0
    for i in range(n-k+1, n+1):
        log_count += np.log(i)
    for i in range(1, k+1):
        log_count -= np.log(i)
    return log_count

def multinomial_choices(n, ks):
    """multinomial_choices return the number of ways to choose n balls into
    len(ks) +1 categories.

    Args:
        n: number of balls
        ks: the counts of all but the last category.

    Returns:
       number of such combinations.
    """
    count = 1
    n_remaining = n
    for k in ks:
        count *= nCk(n_remaining, k)
        n_remaining -= k
    return count

def log_multinomial_choices(n, ks):
    """log_multinomial_choices return the number of ways to choose n balls into
    len(ks) +1 categories.

    Args:
        n: number of balls
        ks: the counts of all but the last category.

    Returns:
       number of such combinations.
    """
    if n == 0:
        return 0.0
    log_count = 0.0
    n_remaining = n
    for k in ks:
        log_count += log_nCk(n_remaining, k)
        n_remaining -= k
    return log_count

def log_p_multinomial(X, p):
    """log_p_multinomial returns the probability of drawing a vector of
    counts from a multinomial parameterized by p.

    Args:
        X: np.array of counts representing a multinomial draw.
        p: np.array of probabilities, corresponding to X

    Returns:
        the log probability of the multinomial draw.
    """
    if sum(X) == 0:
        return 0.0
    # check that input is valid
    assert len(X) == len(p)
    eps = 0.0001
    assert abs(sum(p)- 1.0) < eps

    # calculate log prob.

    log_n_choices = special.gammaln(sum(X)+1) - sum([special.gammaln(x_i+1)
                                                     for x_i in X])
    log_p_items = sum(x_i*np.log(p_i) for (x_i, p_i) in zip(X, p))
    return log_n_choices + log_p_items

def log_p_dir_multinomial(x, alpha):
    """ analytically calculates to probability of observing the counts x
    given the dirichlet prior alpha.

    This calculates precisely the expected value approximated by the two
    estimators above.

    This is the predictive distribution over counts given a Dirichlet prior,
    the form of which we have obtained from:

    Tu, Stephen. "The dirichlet-multinomial and dirichlet-categorical models
    for bayesian in- ference." Computer Science Division, UC Berkeley, Tech.
    Rep.[Online]. Available:
    http://www.cs.berkeley.edu/stephentu/writeups/dirichlet-conjugate-prior.pdf
    (2014).

    Args:
        x: counts
        alpha: dirichlet prior

    """
    if sum(x) == 0:
        return 0.0
    log_term_1 = special.gammaln(sum(x)+1)-(np.sum([special.gammaln(x_i+1)
                                                    for x_i in x]))
    log_term_2 = special.gammaln(sum(alpha))-np.sum([special.gammaln(a)
                                                     for a in alpha])

    log_term_3_num = [special.gammaln(a + x_i) for a, x_i in
                      zip(alpha, x)]

    log_term_3_denom = special.gammaln(sum(alpha) + sum(x))
    log_term_3 = np.sum(log_term_3_num) - log_term_3_denom

    return log_term_1 + log_term_2 + log_term_3

def sample_from_log_probs(log_ps):
    """sample_from_log_probs takes a list of log probabilities, normalizes
    and samples one of them.

    We cannot calculate the normalizing evidence term as is,
    because these probabilities are too low and we will have
    underflow.  Thus we first rescale.

    Args:
        log_ps: a list of log probabilities.

    Returns:
        the index of the sampled value.
    """
    max_log_p = max(log_ps) # i.e. least negative
    log_ps -= max_log_p # now max log_p is 0.0

    # exponential and normalize
    ps = np.exp(log_ps)
    ps /= np.sum(ps)

    # draw from multinomial, get a one-hot indicator vector
    indicator = np.random.multinomial(1, pvals=ps)
    sample = np.where(indicator)[0][0]

    return sample

def KL_divergence(d1, d2):
    return sum(d1_i*np.log(d1_i/d2_i) for (d1_i, d2_i) in zip(d1, d2))

def lbeta(alpha):
    return sum(special.gammaln(a) for a in alpha) - special.gammaln(sum(alpha))

def ldirichlet_pdf(alpha, theta):
    """ldirichlet_pdf calculate the log probability density of a dirchlet.
    # this is directly off of stats.stackexchange
    # http://stats.stackexchange.com/questions/36111/calculating-log-prob-of-dirichlet-distribution-in-high-dimensions

    Args:
        alpha: the prior
        theta: the point on the simplex.

    Returns:
        the log proability.
    """
    kernel = sum((a - 1) * math.log(t) for a, t in zip(alpha, theta))
    return kernel - lbeta(alpha)

def entropy(dist):
    """entropy returns the shannon entropy of the distrubiion provided

    Args:
        dist: a discrete categorical probability distribution, represented
        with a list of floats.

    Returns:
        the Shannon entropy

    """
    eps = 0.0001
    # distribution must be normalized
    dist /= sum(dist) # remove this check
    if  abs(sum(dist) - 1) > eps:
        print "dist: "+str(dist)+" is fucked!"
        sys.exit(2)
    H = 0.0
    for p in dist:
        if p == 0:
            continue # bc lim as p -> 0 of p*log(p) == 0
        H += -p*np.math.log(p, 2)
    return H

def log(string, f=sys.stdout):
    """log write the string provided to the file provided, terminating with
    a new line"""
    f.write(string + "\n")

class KMMM(object):
    """the KMMM class keeps track of the bare bones multinomial mixture, it
    contains the kmer table, mixture proportions and a few other quantities
    usefull for working with these.

    """
    def __init__(self, kmer_table_fn, motifs, alpha_bg=10000.0, alpha_m=100.0,
                 variational=False, kappa=0, tau=None, K=None):
        """initialize mixture

        Set the counts attributed to each of the components to 0.

        Args:
            kmer_table_fn: path to the txt file containing the kmer-counts.
            motifs: dictionary containing all motifs, where keys are motif
                indices and values are the motif objects.
            alpha_bg: the dirichlet parameter corresponding to the
                background component.
            alpha_m: the dirichlet parameter corresponding to the
                motif component(s).
            variational: true if we wish to perform variational updates.
            kappa: SVI paramter for forget rate, if this is nonzero, we
                perform stochastic updates.
            tau: SVI parameter for delay
        """
        self.iteration = 0
        if kappa != 0:
            assert tau is not None
        self.kappa = kappa
        self.tau = tau
        self.p_kmers = {}
        self.log_p_kmer_aligns = {}
        self.variational = variational
        self.alpha_bg = alpha_bg
        self.alpha_m = alpha_m
        self.kmer_counts = read_kmer_table(kmer_table_fn)
        self.N = sum(self.kmer_counts.values())
        self.component_counts = {}
        self.theta = {} # variational dirichlet distribution
        self.component_counts[BG_IDX] = 0
        self.theta[BG_IDX] = alpha_bg
        self.K = K
        for m_idx in motifs.keys():
            self.component_counts[m_idx] = 0
            self.theta[m_idx] = alpha_m
        self.gamma = {}
        self.update_gamma()

    def learning_rate(self):
        if self.kappa == 0:
            return 1.0
        return (self.iteration + self.tau)**(-self.kappa)

    def motif_idxs(self):
        """motif_idxs returns the indices of the current motifs present in
        the mixture model

        Returns:
            list of integer indices.
        """
        idxs_dict = self.component_counts.copy()
        idxs_dict.pop(BG_IDX)
        return idxs_dict.keys()

    def log_prob_given_alpha(self):
        """log_prob_given_alpha returns the probability of the mixing
        proportions, gamma, under the dirichlet prior, alpha.

        Returns:
            the log probability as a float.
        """
        idxs = self.gamma.keys()
        gamma = np.array([self.gamma[idx] for idx in idxs])
        alpha = np.array([self.alpha_bg if idx == BG_IDX else self.alpha_m
                          for idx in idxs])
        return ldirichlet_pdf(alpha, gamma)

    def update_counts(self, component_counts):
        """update_counts updates the counts of each mixture component

        This is meant to be updated to accommodate having more than one
        motif.

        Args:
            component_counts: dictionary mapping component indices to
            counts.
        """
        for idx, count in component_counts.iteritems():
            if self.component_counts.has_key(idx):
                self.component_counts[idx] += count
            else:
                self.component_counts[idx] = count

    def clear_counts(self):
        for k in self.component_counts.keys():
            self.component_counts[k] = 0

    def add_p_kmer(self, kmer, p):
        """add_p_kmer adds or updates the probability of a given kmer.

        Args:
            kmer: the kmer to update
            p: the probability to update it with
        """
        self.p_kmers[kmer] = p

    def add_p_kmer_aligns(self, kmer, log_p):
        """add_p_kmer_aligns adds or updates the log probability of the
        component assignments of the kmer.

        Args:
            kmer: the kmer to update
            log_p: the log multinomial probability
        """
        self.log_p_kmer_aligns[kmer] = log_p

    def update_gamma(self):
        """update_gamma updates the motif proportions.

        For gibbs sampling, we draw from the posterior Dirichlet, and for VI
        we update theta, the variational Dirichlet.

        This is done by taking advantage of Dirichlet-Multinomial conjugacy.
        """
        if self.K == None:
            K = 1
        else:
            K = self.K
        # the posterior dirichlet parameter
        for idx in self.component_counts.keys():
            lr = self.learning_rate()
            theta_i_t = (self.component_counts[idx]/K+
                         (self.alpha_bg if idx == BG_IDX else self.alpha_m))
            self.theta[idx] = (1 - lr) * self.theta[idx] + lr * theta_i_t
        idxs = self.theta.keys()
        post_dir = np.array([self.theta[idx] for idx in idxs])
        if self.variational:
            gamma = post_dir / sum(post_dir)
        else:
            gamma = np.random.dirichlet(post_dir)
        for i, idx in enumerate(idxs):
            self.gamma[idx] = gamma[i]
        self.iteration += 1

class CRP_KMMM(KMMM):
    """ CRP_KMMM is a subclass of KMMM that includes an unseen component and
    use uses a Dirichlet Process to provide a prior over component
    proportions.
    
    This stands for Chinese Restaurant Process, KMMM
    """


    def __init__(self, kmer_table_fn, motifs, alpha_crp, max_components = 6): 
        """ initialize mixture

        Set the counts attributed to each of the components to 0.

        Args:
            kmer_table_fn: path to the txt file containing the kmer-counts. 
            motifs: dictionary containing all motifs, where keys are motif
                indices and values are the motif objects.
            alpha_crp: the dirichlet parameter corresponding to the
        """
        self.alpha_crp = alpha_crp
        self.max_k = max_components
        KMMM.__init__(self, kmer_table_fn, motifs)
        self.component_counts[NEW_IDX] = 0 
        self.component_counts[BG_IDX] = sum(self.kmer_counts.values())

    def remove_empty_components(self):
        """remove_empty_components removes components of the mixture.

        This is done both from gamma and from component_counts
        """
        empty_components = []
        for idx, count in self.component_counts.iteritems():
            if count == 0:
                empty_components.append(idx)
        for idx in empty_components:
            self.component_counts.pop(idx)
            self.gamma.pop(idx)
        return empty_components

    def update_gamma(self):
        """ update gamma as the CRP probabilities for different components.

        This will include a probability for a yet unseen component, unless
        we have already reached the maximum number of allowed components.
        """
        idxs = self.component_counts.keys()
        N = sum(self.component_counts.values())
        alpha_crp = self.alpha_crp if len(idxs) < self.max_k else 0.0 
        for idx in idxs:
            if idx == NEW_IDX: # unseen component
                self.gamma[NEW_IDX] = float(alpha_crp)/(N + alpha_crp)
            else: # existing components
                self.gamma[idx] = float(self.component_counts[idx])/(N +
                        self.alpha_crp)

    def log_prob_given_alpha(self):
        """log_prob_given_alpha returns the probability of the mixing
        proportions, gamma, under the CRP prior

        Returns:
            the log probability as a float.
        """
        K = len(self.component_counts.keys()) # number of seen components
        N = sum(self.component_counts.values())
        log_numerator = K*np.log(self.alpha_crp) + sum(
                special.gammaln(N_k) for N_k in
                self.component_counts.values()
                )   
        log_denom = (special.gammaln(N + self.alpha_crp) -
            special.gammaln(self.alpha_crp))
        return log_numerator - log_denom
