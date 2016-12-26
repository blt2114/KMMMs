"""background contains a class for the background model of KMMMs.

The idea is that this can provide an interface that will be shared for key
methods across different implementations of background models.
"""
from collections import defaultdict
import numpy as np
from inference import tools

class Phi(object):
    """ Phi is the background model.  This implementation is an arbitrary
    order markov chain.

    The probabilities of each sequence under phi is dynamically updated
    based on assigned sequences with Dirichlet-Multinomial conjugacy, where
    the dirichlet is over the start and transition probabilities.

    In each iteration, Kmers and Kmer segments are assigned to the
    background.  These sequences are then used to update the markov model.

    PSEUDO-CODE for this process

    for kmer in table:
        draw alignments // align kmers to phi, or to overlaps with motifs.
        for each alignment
            find seqs from background //
            add_seqs(seqs) // add sequences that are in background.
        update_kmer_counts() // add kmer counts in these sequences
        clear_seqs() // clear sequences added
    update_background_model()  // based of kmers counts calculate transition
                                  probs.
    clear_counts() // clear so that in next iteration we start from scratch.

    """

    def __init__(self, prior_size=10, chain_len=1, mle=False, static=False,
                 k=7):
        """__init__ initialized the bacgkround model.

        Args:
            prior_size: the magnitude of the dirichlet prior over transition
                probabilities for the longest chain length.  These act as
                pseudo-counts of each kmer of length chain_len+1.  So if
                chain_len is 2, and prior_size = 1, it is equivalent to
                observing 1 occurence of  of each 3-mer, 4 occurences of
                each dimer and 16 occurences of each monomer.
            chain_len: the length of the markov chain (i.e. we consider the
                probability of bases conditioned on this many previous bases)
            mle: true if mle parameters should be used instead of sampling
                when updating model for background.
            static: set to true to not dynamically update the background
                model.
            k: the length of subsequences to precalculate for inference.
        """
        self.chain_len = chain_len # the length of the markov chain.
        # sequences attributed to background
        self.seqs = defaultdict(int)

        # subsequences attributed to background
        self.kmer_counts = defaultdict(int)
        self.kmer_probs = {}
        self.markov_probs = {}
        self.pseudo_count = prior_size
        self.mle = mle
        self.static = static
        self.K = k

    def prob(self, seq):
        """ returns the probability of seeing seq as an arbitrary
        subsequence within background.

        Args:
            seq: string sequence in background.

        Returns:
            float probability of seeing that sequence in background
        """
        if self.static:
            return 0.25**len(seq)
        if not self.kmer_probs.has_key(seq):
            self.kmer_probs[seq] = self.score_kmer(seq)
        return self.kmer_probs[seq]

    def clear_counts(self):
        """clear_counts wipes the counts of kmers subsequences used to
        generate transition probabilities.
        """
        if self.static:
            return
        self.kmer_counts.clear()

    def clear_seqs(self):
        """clear_seqs wipes the the counts of sequences, from which all kmer
        subcounts are generated.
        """
        if self.static:
            return
        self.seqs.clear()

    def add_seqs(self, seq_counts):
        """ add_seqs adds the provided sequence counts to the running list
        in phi.

        Args:
            seq_counts: a dictionary mapping from sequences to their count
        """
        for seq, count in seq_counts.iteritems():
            self.seqs[seq] += count

    def update_kmer_counts(self):
        """ update_kmer_counts updates the tracked # of occurences of kmers
        in the background with a set of seqeunce coming from the
        background model.

        After these are updated, the sequences are cleared.
        """
        for seq, count in self.seqs.iteritems():
            for kmer in tools.subsequences(seq, k=self.chain_len+1):
                kmer_rc = tools.rev_comp(kmer)
                self.kmer_counts[kmer] += count
                self.kmer_counts[kmer_rc] += count
        self.seqs.clear()

    def score_kmer(self, seq):
        """score_kmer finds the probability of seq being emitted by the
        markov chain provided.

        Args:
            seq: sequence to score.

        Returns:
            the probability of this sequence under tha background model.
        """
        p_kmer = 1.0
        for i, base in enumerate(seq):
            trans_probs = self.markov_probs[seq[max(0, i-self.chain_len):i]]
            p_kmer *= trans_probs[base]
        return p_kmer

    def update(self):
        """update updates the frequencies of seqeunces described by the
        background model probabilities according to the given markov chain
        length and kmer frequences.

        a given seqeunce and its reverse complement must have the same
        probability, e.g. ACCG must be the same as CGGT

        This first translates kmer counts into kmer frequences/ transition
        probabilities.

        Next, using the markov chain defined by these probabilities, the
        method scores all kmers.

        This first iteration of does not take into account the redundancy of
        reverse complements, and stores each sequence and its reverse
        complement independently.

        Args:
            k: the length of Kmers for which to generate probabilities.
        """
        if self.static:
            return
        self.kmer_probs = {}

        self.markov_probs = {}

        # first add in pseudo counts associated with our dirichlet prior.
        # Adding these counts at this stage rather than in explicitly using
        # the proper Dirichlet multinomial conjugacy makes our
        # implementation cleaner.
        self.clear_seqs()
        kmer_len = self.chain_len + 1
        pseudo_counts = {}
        for kmer in tools.all_kmers(kmer_len):
            pseudo_counts[kmer] = self.pseudo_count
        self.add_seqs(pseudo_counts)
        self.update_kmer_counts()

        # First assemble counts of the base following each kmer of length
        # chain_len-1.  At this stage, markov_probs is in fact a collection
        # of counts, rather than probabilities.
        # p(x|y)=markov_probs[("y","x")], for 0th order,
        # markov_probs[("","A")]
        for kmer, count in self.kmer_counts.iteritems():
            if not self.markov_probs.has_key(kmer[:-1]):
                # by making this a defaultdict, we get a count of 0 for
                # unobserved transitions.
                self.markov_probs[kmer[:-1]] = defaultdict(float)
            self.markov_probs[kmer[:-1]][kmer[-1]] = float(count)

        # For each of the preceding base(s), scale the counts down to 1.0,
        # so that they represent transition probabilities.
        for _, trans_probs in self.markov_probs.iteritems():
            counts = np.array([trans_probs[b] for b in tools.BASES])
            if self.mle: # go with maximum likelihood value
                new_trans_probs = counts / sum(counts)
            else: # sample from posterior
                new_trans_probs = np.random.dirichlet(counts)
            for base in tools.BASES:
                trans_probs[base] = new_trans_probs[tools.IDX[base]]

            # make certain that this does sum to one.
            assert abs(sum(trans_probs.values()) - 1.0) <= 0.00001

        # use this to score every kmer. For lengths 1 to K
        for length in range(self.K):
            for kmer in tools.all_kmers(length+1):
                self.kmer_probs[kmer] = self.score_kmer(kmer)
        self.kmer_probs[""] = 1.0
