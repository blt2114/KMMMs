"""motif contains several classes for the motifs of KMMMs

All current implementations share a common interface and and are based on an
underlying Position Specific Frequency Matrix/ Position Weight Matrix (PWM)
generative model.

The primary functions of motif objects are to store and update motif
parameters and calculate the likelihoods of alignments of the motif against
given sequences.  These likelihoods are calculated with a background model
as well, which provides the likelihoods of the non-overlapping portions of
sequences in a given alignment.

This interface can be extended to include a motif with
dinucleotide features, or other more complex models.
"""
from collections import deque
import numpy as np
from numpy import fliplr, flipud
from scipy import stats, special
from inference import tools

class Motif(object):
    """Motif provides the object type for DNA sequence motifs.  This uses an
    underlying PWM model.

    The counts at each base and the motif parameters are stored in a deque,
    so that columns (corresponding to positions in the motif) can be easily
    added and removed from either side.

    """

    def __init__(self, length=5, pi_mag=1, K=None):
        """ initialize the motif

        The motif parameters are stored as deques of np.arrays so that it is
        easy to append and remove bases from the edges of the motifs.

        Args:
            length: the length of the motif, more precisely, the active
                component.
            pi_mag: the magnitude of the dirichlet prior of beta, pi. This
                prior acts as a pseudo-count on each position.
            K: we will want to scale down the number of aligned bases by K
                since we expect a factor of K more alignments when we are using
                overlapping K-mers.
        """
        self.motif_len = length

        # Position Weight Matrix will be stored as a deque of 1D np.arrays,
        # this representation is overparameterized, since in each column
        # must lie on the 3-Simplex, but keeping all for values explicity
        # allows for easier and more efficient  manipulation
        pwm = np.zeros([length+2, len(tools.BASES)])
        self.beta = deque(pwm)
        self.beta_rc = deque(flipud(fliplr(np.array(self.beta))))
        Motif.update_rc(self) # initialize reverse complement pwm parameters

        # dirichlet prior on PWM, a uniform prior is placed over each
        # position of the PWM
        self.pi_dir = pi_mag*np.ones(shape=len(tools.BASES), dtype=np.float32)

        # we use this to store counts of each base observed in each of the
        # positions along the length of the motif as well as on the sides.
        self.eta = deque(np.zeros([length+2, len(tools.BASES)]))
        self.K = K

    def length(self):
        """length returns the length of the motif

        Returns:
            integer length
        """
        return self.motif_len

    def log_prob_given_pi(self):
        """log_prob_given_pi returns the log probability of the motif under the
        dirichlet prior.

        Returns:
            float prior probability
        """
        prob = 0.0
        for i in range(1, self.motif_len+2):
            prob += tools.ldirichlet_pdf(self.pi_dir, self.beta[i])
        return prob

    def consensus(self):
        """consensus finds the consensus sequence of the motif

        Returns:
            the consensus sequence as a string
        """
        seq = ""
        for i in range(1, self.motif_len+1):
            seq += tools.BASES[self.beta[i].argmax()]
        return seq

    def clear_alignments(self):
        """clear_alignments wipes all stored sequence alignments. This is
        done so that in
        """
        for i in range(len(self.eta)):
            self.eta[i] = np.zeros(len(tools.BASES))

    def update_rc(self):
        """update_rc sets the pwm corresponding to the reverse complement of
        the motif to the upside down, mirror image of the original pwm.
        This is the reverse complement because the reverse order of the
        bases is complementary to the initial order.
        """
        self.beta_rc = deque(flipud(fliplr(np.array(self.beta))))

    def update_pwm(self):
        """update_pwm draws the pwm from its posterior.

        The posterior is defined by the observed counts at each postion,
        base_counts, and the uniform dirichlet prior, pi.

        The sampling is done taking advantage of Dirichlet-Multinomial
        conjugacy.
        """
        for i in range(len(self.eta)):
            if self.K is not None:
                self.beta[i] = np.random.dirichlet(self.eta[i]/self.K +
                    self.pi_dir)
            else:
                self.beta[i] = np.random.dirichlet(self.eta[i]+ self.pi_dir)
        Motif.update_rc(self) # initialize reverse complement pwm parameters

    def add_alignment(self, seq, seq_st, motif_st, l_overlap, count=1,
                      is_rc=False):
        """add_alignment takes a sequence and alignment and uses it add up
        the observed counts at overlapping bases.  This is necessary for
        updating the motif.

        The base counts are returned rather than used to update self.eta so
        that the sampling can be easily parallelized.

        Args:
            seq: a string representation of the aligned sequence.
            seq_st: the index of the beginning of the sequence alignemnt.
            motif_st: the index of the beginning of motif alignemnt.
            l: the length of the alignment.
            count: the number of such alignments to tally up.
            is_rc: true if the alignment provided corresponds to the reverse
                complement of the motif

        Returns:
            a list of the segment(s) of seq that are not overlapping with
            the motif and the counts of bases in each position of the motif
            as a matrix.

        """
        # since we deal with reverse complements by considering the reverse
        # of the motif, when adding an alignment corresponding to a reverse
        # complement, we must add counts to the reverse complement of the
        # base counts.
        base_counts = np.zeros([len(self.eta), len(tools.BASES)])

        for i in range(l_overlap):
            base_counts[motif_st+i+1][tools.IDX[seq[seq_st+i]]] += count

        # add counts for outside the range of the sequence.
        if seq_st != 0:
            base_counts[motif_st][tools.IDX[seq[seq_st-1]]] += count
        if seq_st+l_overlap != len(seq):
            base_counts[motif_st+l_overlap+1][tools.IDX[seq[seq_st+l_overlap]]] += count

        if is_rc: # we must then flip this back.
            base_counts = fliplr(flipud(base_counts))

        # the non-overlapping portions of the sequence, s_rest, may be both
        # at the beginning of and end of the sequence.
        bg_seg_1 = None if seq_st == 0 else seq[:seq_st]
        bg_seg_2 = None if seq_st + l_overlap == len(seq) else seq[seq_st+l_overlap:]
        return (bg_seg_1, bg_seg_2, base_counts)

    def p_overlap(self, s_overlap, motif_st, l_overlap, is_rc=False):
        """ p_overlap caclute the probability of a given sequence segment
        overlapping with a motif, given the alignment.

        Args:
            s_overlap: string representing overlapping portion of the motif
            motif_st: the begin idx of the motif
            l_overlap: the length of the alignment.
            is_rc: if the reverse complementary motif should be used.

        Returns:
            a float representation of the likelihood of observing
            overlapping sequence given this alignment.
        """
        motif_end = motif_st + l_overlap
        pwm = self.beta_rc if is_rc else self.beta
        pwm_overlap = np.array(pwm)[motif_st+1:motif_end+1]
        assert len(pwm_overlap) == len(s_overlap)
        prob = np.sum(tools.one_hot_encoding(s_overlap)*pwm_overlap,
                      axis=1).prod()
        return prob

    def score_alignment(self, seq, seq_st, motif_st, l_overlap, phi,
                        is_rc=False):
        """ score_alignment returns the probability of an alignment

        If we consider reverse complements by looking at the reverse
        complement of the motif, we calculate the probability of the
        non-overlapping portion of the sequence, 's_rest', in precisely the
        same way.

        Args:
            seq: a string representation of the aligned sequence.
            seq_st: the index of the beginning of the sequence alignemnt.
            motif_st: the index of the beginning of motif alignemnt.
            l_overlap: the length of the alignment.
            phi: background model
            is_rc: if we are working with the reverse complement of the
                motif.

        Returns:
            the likelihood of the alignment, as a float.
        """
        # calculate the probability of the overlapping portion of the
        # sequence.
        s_overlap = seq[seq_st:seq_st + l_overlap]
        p_s_overlap = self.p_overlap(s_overlap, motif_st, l_overlap, is_rc)

        # calculate the probability of the non overlapping portion(s) of the
        # sequence
        p_s_rest = 1.0
        if seq_st != 0: # if there is part of the sequence on the left
            p_s_rest *= phi.prob(seq[:seq_st])
        if seq_st+l_overlap != len(seq): # additional sequence on right
            p_s_rest *= phi.prob(seq[seq_st+l_overlap:])

        # total probability is the product of overlapping and nonoverlapping
        # segments.
        return p_s_overlap*p_s_rest

    def all_alignments(self, k):
        """all_alignments creates a list of all possible alignments of the
        motifs with sequence of length K.

        Args:
            k: the length of the sequence

        Returns:
            a list of all possible alignments, such that each element of the
            list is a tuple of the form (seq_st, motif_st, rev_comp) or
            types (int, int, boolean)
        """
        alignments = []

        # alignments with the motif starting before the sequence.
        seq_st = 0
        for motif_st in range(1, self.motif_len):
            alignments.append((seq_st, motif_st, False))
            alignments.append((seq_st, motif_st, True))

        # alignments with the motif starting at or after the sequence.
        motif_st = 0
        for seq_st in range(0, k):
            alignments.append((seq_st, motif_st, False))
            alignments.append((seq_st, motif_st, True))
        return alignments

    def score_all_alignments(self, seq, phi):
        """score_all_alignments calculates the likelihoods of each
        possible alignment.

        The likelihoods are calculate given that there is some overlap.
        Since all overlaps can occur with equal probability, we scale down
        by the number of alignments.

        Args:
            seq: string representation of the sequence.
            phi: the background model.

        Returns:
            a list of the alignments, in the form (seq_st, motif_st, is_rc),
            and their probabilities, as list.
        """
        align_ps = []
        alignments = self.all_alignments(len(seq))

        align_ps = [self.score_alignment(seq, seq_st, motif_st,
                    min(self.motif_len-motif_st, len(seq)-seq_st) , phi, is_rc)
                    for (seq_st, motif_st, is_rc) in alignments]

        # All offsets are equal prior proabilities so we must account for
        # this here.
        align_ps = [p/len(align_ps) for p in align_ps]
        return alignments, align_ps

    def change_size(self, is_left, add):
        """change_size adjusts the size of the motif

        Either adds or removes a base position from the left or right side
        of the motif.

        Args:
            is_left: True if change is on the left side.
            add: True is adding another base, otherwise a base is removed.
        """
        if add: # motif gets longer.
            self.motif_len += 1
            if is_left:
                self.beta.appendleft(np.random.dirichlet(self.pi_dir))
                self.eta.appendleft(np.zeros(4))
            else:
                self.beta.append(np.random.dirichlet(self.pi_dir))
                self.eta.append(np.zeros(4))
        else:
            self.motif_len -= 1
            if is_left:
                self.beta.popleft()
                self.eta.popleft()
            else:
                self.beta.pop()
                self.eta.pop()

    def string(self, gamma_m = None, idx = None):
        """string creates a string representation of the motif

        This includes the consensus sequence, PWM and entropy of the
        distribution at each position.
        """
        m_str = ""
        if idx != None:
            m_str += "\nMotif: %d"%idx +"\n"
        m_str += "Consensus:\t"+"\t".join(self.consensus()) + "\n"
        if gamma_m != None:
                m_str += "gamma:\t"+str(gamma_m)+ "\n"

        m_str += "PWM:\t"+"\t".join(str(i) for i in
                range(self.motif_len+2)) + "\n"
        m_str += "\n".join("\t".join([tools.BASES[k]]+
                                     [ "{0:.2f}".format(i) for i in j])
                           for k, j in enumerate(np.array(self.beta).T))

        m_str += "\nEntropy\t" + "\t".join("{0:.2f}".format(
            tools.entropy(i)) for i in self.beta)
        m_str += "\n"
        return m_str

    def save_as_psam(self, fn):
        """save_as_psam write the motif to a file as a PSAM in the format that
        can be read by the REDUCE_SUITE LogoGenerator

        Args:
            fn: the destination to write the psam to.
        """
        f = open(fn,"w")
        f.write("<psam_length>%d</psam_length>\n\n"%(self.motif_len+2))

        f.write("<psam>\n")
        for dist in self.beta:
            dist_strs = [str(p_b) for p_b in dist]
            f.write("\t".join(dist_strs)+"\n")
        f.write("</psam>")
        f.close()

class VariationalMotif(Motif):
    """VariationalMotif is a subclass of motif, that keeps posteriors the
    betas.

    """
    def __init__(self, length=5, pi_mag=1, kappa=0,tau=None, K=None):
        """ initialize the motif

        Initialization of variational parameters is tricky.  We do not want
        to initialize where the most likely pwm has equal probability of
        bases in every position as this will take a long time to fit....

        Args:
            length: the length of the motif, more precisely, the active
                component.
            pi_mag: the magnitude of the dirichlet prior of beta, pi. This
                prior acts as a pseudo-count on each position.
            kappa: forget rate, between 0.5 and 1, closer to 1.0 means
                forgets faster -->  Chain movement will slow down faster.
                if kappa = 0, and each batch is full dataset, this it normal
                VI.
            tau: delay parameter, downweights early samples (as if starting
                far into the chain).
        """
        if kappa != 0: # in this case we perform SVI updates.
            assert tau is not None

        # we must additionally store the variational dirichlet parameter.
        self.lmbda = deque(pi_mag*np.ones([length+2, len(tools.BASES)]))
        self.lmbda_rc = deque(flipud(fliplr(np.array(self.lmbda))))

        Motif.__init__(self, length, pi_mag=pi_mag, K=K)

        self.iteration = 0
        self.tau=tau
        self.kappa=kappa

    def update_rc(self):
        """update_rc sets the pwm corresponding to the reverse complement of
        the motif to the upside down, mirror image of the original pwm.
        This is the reverse complement because the reverse order of the
        bases is complementary to the initial order.
        """
        Motif.update_rc(self)
        self.lmbda_rc = deque(flipud(fliplr(np.array(self.lmbda))))

    def add_noise_to_var_dist(self, noise_level=100):
        """ add_noise_to_var_dist adds noise to the variational
        distribution, this is necessary to differentiate motifs at
        initialization.

        Args:
            noise_level: scaling parameter of gamma distrubute noise added
                to variational parameters at initialization.
        """
        for i in range(len(self.lmbda)):
            self.eta[i] += np.random.gamma(noise_level, scale=100,
                                           size=len(tools.BASES))
            self.lmbda[i] += self.eta[i]
        self.update_pwm(None) # pass phi as None
        self.update_rc() # adds reverse complement parameters

    def learning_rate(self):
        """ calculate learning rate as per SVI method (Hoffman et al.,
        2013)

        This is only relevant for SVI.  If we are performing normal VI,
        kappa is 0 and the learning rate is 1.
        """
        if self.kappa == 0:
            return 1.0
        return (self.iteration + self.tau)**(-self.kappa)

    def change_size(self, is_left, add):
        """change_size adjusts the size of the motif

        Either adds or removes a base position.

        Args:
            is_left: True if change is on the left side.
            add: True is adding another base, otherwise a base is removed.
        """
        Motif.change_size(self, is_left, add)
        if add: # motif gets longer.
            if is_left:
                self.lmbda.appendleft(self.pi_dir*np.ones(4))
            else:
                self.lmbda.append(self.pi_dir*np.ones(4))
        else:
            if is_left:
                self.lmbda.popleft()
            else:
                self.lmbda.pop()
        self.update_rc()

    def p_overlap(self, s_overlap, motif_st, l_overlap, is_rc=False):
        """ p_overlap caclute the probability of a given sequence segment
        overlapping with a motif, given the alignment.

        Args:
            s_overlap: string representing overlapping portion of the motif
            motif_st: the begin idx of the motif
            l_overlap: the length of the alignment.
            is_rc: if the reverse complementary motif should be used.

        Returns:
            a float representation of the likelihood of observing
            overlapping sequence given this alignment.
        """
        motif_end = motif_st + l_overlap
        pwm = self.lmbda_rc if is_rc else self.lmbda
        pwm_overlap = np.array(pwm)[motif_st+1:motif_end+1]
        assert len(pwm_overlap) == len(s_overlap)
        prod = 1.0
        for base, rho in zip(s_overlap, pwm_overlap):
            prod *= np.exp(special.digamma(rho[tools.IDX[base]])-
                           special.digamma(sum(rho)))
        return prod

    def score_all_alignments(self, seq, phi):
        """score_all_alignments calculates the likelihoods of each
        possible alignment.

        The likelihoods are calculate given that there is some overlap.
        Since all overlaps can occur with equal probability, in the non
        variational version, we scale down by the number of alignments.  In
        this version, however, we consider the variation distribution over
        offsets as well, so we do not need to do this.

        Args:
            seq: string representation of the sequence.
            phi: the background model.

        Returns:
            a list of the alignments, in the form (seq_st, motif_st, is_rc),
            and their probabilities, as list.
        """
        align_ps = []
        alignments = self.all_alignments(len(seq))

        align_ps = [self.score_alignment(seq, seq_st, motif_st,
                    min(self.motif_len-motif_st, len(seq)-seq_st), phi, is_rc)
                    for (seq_st, motif_st, is_rc) in alignments]

        # All offsets are equal prior proabilities so we must account for
        # this but rather than doing it here, as we do for the non-variational
        # motif scoring, we do this within inference.
        return alignments, align_ps

    def update_pwm(self):
        """update_pwm updates the variational parameters of the motif.

        The update takes the current estimate provide by the base counts and
        updates the motif based on the learning rate.

        The iteration of the motif is also incremented.
        """
        for i in range(len(self.eta)):
            lr = self.learning_rate()
            estimate_t = self.eta[i] + self.pi_dir
            self.lmbda[i] = (1 - lr) * self.lmbda[i] + lr * estimate_t
            self.beta[i] = self.lmbda[i]/sum(self.lmbda[i])
        self.update_rc()
        self.iteration += 1

class DynamicMotif(VariationalMotif, Motif):
    """DynamicMotif is a subclass of Motif, which allows for learning the
    length of the motif.

    Dynamic motifs may change shape when they are updated; During the update
    step, the shape of the motif is sampled.

    The prior on motif shape is that the length is geometrically
    distributed, and must meet the constraint that base in the motif are
    consecutive (i.e. no spacers).
    """
    def __init__(self, length=5, pi_mag=100, min_len=4, max_len=10,
            variational=False, tau=None, kappa=0, log_p_longer=-15,
            verbose=False, f=None, K=None):
        """ initialize the motif

        The motif parameters are stored as deques of np.arrays so that it is
        easy to append and remove bases from the edges of the motifs.

        We may also update Dynamic motifs in the context of Variational
        inference.  Here, we simple choose the maximum likelihood shape.

        Args:
            length: the length of the motif, more precisely, the active
                component.
            pi_mag: the magnitude of the dirichlet prior of beta, pi. This
                prior acts as a pseudo-count on each position.
            min_len: the minimum number of bases that can be in a motif.
            max_len: the largest number of bases a motif can be.
            variational: if we want to use the Variational Motif as our base
                class.
            tau: delay parameter for SVI updates.
            kappa: forget rate parameter for SVI updates.
            log_p_longer: log prior probability of motif being longer than its
                current length.  In practice, this must be a large value,
                especially if we have a large Kmer table.
            verbose: True if we want to log more often (for debugging)
            f: file handle to log to if verbose
        """
        assert length >= min_len and length <= max_len
        self.min_len = min_len
        self.max_len = max_len

        # this parameterizes our geometric prior on motif length.
        self.log_p_longer=log_p_longer

        if verbose: # we only need file handle if verbose
            assert f is not None
        self.verbose=verbose
        self.f=f

        if variational:
            if kappa != 0:
                assert  tau is not None
        self.variational = variational

        # Initialize using one of the base classes.
        if variational:
            VariationalMotif.__init__(self, length, pi_mag=pi_mag,
                                      tau=tau, kappa=kappa, K=K)
        else:
            Motif.__init__(self, length, pi_mag=pi_mag, K=K)

    def update_pwm(self, phi):
        """update_pwm performs an update of PWM distributions and motif
        shape.

        The PWM parameters are updated according to the base class, either
        with a (stochastic) variational update, or sampling.

        After the PWM, the shape is updated.  If variational, with the MAP
        estimate, otherwise by sampling.

        In both cases, we must work in log space to prevent numerical
        underflow.

        Args:
            phi: background model, this is necessary to compute the
                likelihood of base counts if positions are not part
                of motifs.  If phi is none, 0.25 prob of each base is
                assumed, i.e. a random background model.
        """

        # First update PWM paramters, using correct base class method.
        if self.variational:
            VariationalMotif.update_pwm(self)
        else:
            Motif.update_pwm(self)

        if phi is None:
            phi_bases = np.array([0.25 for b in tools.BASES])
        else:
            phi_bases = np.array([phi.prob(b) for b in tools.BASES])

        # we consider changing motif from both sides.
        for is_left in [True, False]:
            base_counts = (self.eta[0 if is_left else -1], # bases out of motif
                           self.eta[1 if is_left else -2] # at edge of motif
                          )

            # calculate likelihood under motif
            log_p_motif = (tools.log_p_dir_multinomial(base_counts[0], self.pi_dir),
                           tools.log_p_dir_multinomial(base_counts[1], self.pi_dir)
                          )

            # calculate likelihood under background
            log_p_bg = (tools.log_p_multinomial(base_counts[0], phi_bases),
                        tools.log_p_multinomial(base_counts[1], phi_bases)
                        )

            # calculate prior * likelihood for 3 possiblities
            log_p_remove = sum(log_p_bg)
            log_p_same = log_p_motif[1]+log_p_bg[0] + self.log_p_longer
            log_p_add = sum(log_p_motif) + 2*self.log_p_longer

            if self.verbose:
                tools.log("p remove, same, add: (%f, %f, %f)"%(log_p_remove,
                          log_p_same, log_p_add), self.f)
            log_ps = np.array([log_p_remove, log_p_same, log_p_add])

            # if variational, take MAP, otherwise sample
            if self.variational:
                motif_state = np.argmax(log_ps)
            else:
                motif_state = tools.sample_from_log_probs(log_ps)

            # check that changing length doesn't break limits.
            if int(motif_state) == 2 and self.motif_len >= self.max_len:
                if self.verbose:
                    tools.log("motif already at max length", self.f)
                motif_state = 1
            elif int(motif_state) == 0 and self.motif_len <= self.min_len:
                if self.verbose:
                    tools.log("motif already at min length", self.f)
                motif_state = 1

            if motif_state == 1: # motif stays the same length.
                if self.verbose:
                    tools.log("motif length staying the same", self.f)
            else:
                add_base = (motif_state == 2)
                if self.verbose:
                    tools.log("%s base on the %s"%(
                        "adding" if add_base else "removing",
                        "left" if is_left else "right"), self.f)
                if isinstance(self, SymmetricMotif):
                    SymmetricMotif.change_size(self, is_left, add_base)
                elif self.variational:
                    VariationalMotif.change_size(self, is_left, add_base)
                else:
                    Motif.change_size(self, is_left, add_base)

            if self.variational:
                VariationalMotif.update_rc(self)
            else:
                Motif.update_rc(self)

    def change_size(self, is_left, add):
        """  changes size of motif
        """
        if self.variational:
            return VariationalMotif.change_size(self, is_left, add)
        else:
            return Motif.change_size(self, is_left, add)

    def update_rc(self):
        """ updates reverse complement
        """
        if self.variational:
            return VariationalMotif.update_rc(self)
        else:
            return Motif.update_rc(self)

    def score_all_alignments(self, seq, phi):
        """score_all_alignments calculates the likelihoods of each
        possible alignment.

        Args:
            seq: sequence to score alignments of
            phi: background model

        Returns:
            scores of all alignments
        """
        if self.variational:
            return VariationalMotif.score_all_alignments(self, seq, phi)
        else:
            return Motif.score_all_alignments(self, seq, phi)

    def p_overlap(self, s_overlap, motif_st, l_overlap, is_rc=False):
        """ p_overlap caclute the probability of a given sequence segment
        overlapping with a motif, given the alignment.
        """
        if self.variational:
            return VariationalMotif.p_overlap(self, s_overlap, motif_st, l_overlap, is_rc)
        else:
            return Motif.p_overlap(self, s_overlap, motif_st, l_overlap, is_rc)

class SymmetricMotif(DynamicMotif):
    """SymmetricMotif is a subclass of DynamicMotif, in which motifs are
    restricted to being an even length and being reverse complementary
    symmetric.
    """
    def __init__(self, length=4, pi_mag=100, min_len=4, max_len=10,
                 variational=False, tau=None, kappa=0, log_p_longer=-15,
                 verbose=False, f=None, K=None):
        """ initialize the motif

        The motif parameters are stored as deques of np.arrays so that it is
        easy to append and remove bases from the edges of the motifs.

        We may also update Dynamic motifs in the context of Variational
        inference.  Here, we simple choose the maximum likelihood shape.

        Args:
            length: the length of the motif, more precisely, the active
                component.
            pi_mag: the magnitude of the dirichlet prior of beta, pi. This
                prior acts as a pseudo-count on each position.
            min_len: the minimum number of bases that can be in a motif.
            max_len: the largest number of bases a motif can be.
            variational: if we want to use the Variational Motif as our base
                class.
            tau: delay parameter for SVI updates.
            kappa: forget rate parameter for SVI updates.
            log_p_longer: log prior probability of motif being longer than its
                current length.  In practice, this must be a large value,
                especially if we have a large Kmer table.
            verbose: True if we want to log more often (for debugging)
            f: file handle to log to if verbose
        """
        assert length >= min_len and length <= max_len
        assert max_len%2 == 0 and length%2 == 0 and min_len%2 == 0
        DynamicMotif.__init__(self, length, pi_mag=pi_mag, min_len=min_len,
                              max_len=max_len, variational=variational, tau=tau,
                              kappa=kappa, log_p_longer=log_p_longer,
                              verbose=verbose, f=f, K=K)

    def change_size(self, is_left, add):
        """change_size adjusts the size of the motif

        Either adds or removes a base position from both sides of the motif.

        Args:
            is_left: This pameter is ignored for symetric motifs since the
            operation is performed on both sides..
            add: True is adding another base, otherwise a base is removed.
        """
        DynamicMotif.change_size(self, True, add)
        DynamicMotif.change_size(self, False, add)
        assert len(self.beta) == self.motif_len + 2

    def add_alignment(self, seq, seq_st, motif_st, l_overlap, count=1,
                      is_rc=False):
        """add_alignment takes a sequence and alignment and uses it add up
        the observed counts at overlapping bases.  This is necessary for
        updating the motif.

        Args:
            seq: a string representation of the aligned sequence.
            seq_st: the index of the beginning of the sequence alignemnt.
            motif_st: the index of the beginning of motif alignemnt.
            l: the length of the alignment.
            count: the number of such alignments to tally up.
            is_rc: this is ignored because the motif must be rc symmetric

        Returns:
            a list of the segment(s) of seq that are not overlapping with
            the motif and the counts of bases in each position of the motif
            as a matrix.

        """
        bg_seg_1, bg_seg_2, base_counts = DynamicMotif.add_alignment(
                self, seq, seq_st, motif_st, l_overlap, count, False)
        # To ensure that information on both sides of alignment are used
        # when we updathe motif.
        base_counts += fliplr(flipud(base_counts))
        return bg_seg_1, bg_seg_2, base_counts

    def all_alignments(self, k):
        """all_alignments creates a list of all possible alignments of the
        motifs with sequence of length K.

        Since them motif must be symmetric, the rev_comp flag is always
        False.

        Args:
            k: the length of the sequence

        Returns:
            a list of all possible alignments, such that each element of the
            list is a tuple of the form (seq_st, motif_st, rev_comp) or
            types (int, int, boolean)

        """
        alignments = []

        # alignments with the motif starting before the sequence.
        seq_st = 0
        for motif_st in range(1, self.motif_len):
            alignments.append((seq_st, motif_st, False))

        # alignments with the motif starting at or after the sequence.
        motif_st = 0
        for seq_st in range(0, k):
            alignments.append((seq_st, motif_st, False))
        return alignments

    def update_pwm(self, phi):
        """update_pwm performs an update of PWM distributions and motif
        shape.

        The PWM parameters are updated according to the base class, either
        with a (stochastic) variational update, or sampling.

        After the PWM, the shape is updated.  If variational, with the MAP
        estimate, otherwise by sampling.

        In both cases, we must work in log space to prevent numerical
        underflow.

        Args:
            phi: background model, this is necessary to compute the
                likelihood of base counts if positions are not part
                of motifs.  If phi is none, 0.25 prob of each base is
                assumed, i.e. a random background model.
        """
        assert len(self.beta) == self.motif_len +2
        DynamicMotif.update_pwm(self, phi)
        assert len(self.beta) == self.motif_len +2
        # cast away from a deque.
        beta_half = np.array(self.beta)[:(self.motif_len/2)+1]
        beta_half_rc = flipud(fliplr(beta_half))
        self.beta = deque(np.vstack([beta_half, beta_half_rc]))
        assert len(self.beta) == self.motif_len +2
        if self.variational:
            lmbda_half = np.array(self.lmbda)[:(self.motif_len/2)+1]
            lmbda_half_rc = flipud(fliplr(lmbda_half))
            self.lmbda = deque(np.vstack([lmbda_half, lmbda_half_rc]))
            assert len(self.lmbda) == self.motif_len +2
        self.update_rc()
