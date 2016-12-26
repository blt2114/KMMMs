"""gen_synthetic_sequence generates synthetic sequence according the
probabilistic segmentation model of motifs.

A random background model is assumed.

Motifs and proportions are specified.
"""
import sys
from numpy.random import choice
from numpy import fliplr, flipud
import numpy as np
from inference import tools


def rand_base():
    """rand_base returns an A, C, T, or G with equal probability."""
    return np.random.choice(tools.BASES)

def gen_motif(pwm):
    """gen_motif samples a sequence from distribution specified by the pwm
    representation of the motif by sampling from each position independently

    Args:
        pwm: pwm representation of the sequence.
    """
    seq = ""
    for i in range(len(pwm)):
        seq += np.random.choice(tools.BASES, p=pwm[i])
    return seq

def main(argv):
    if len(argv) != 1:
        sys.stderr.write("python gen_fake_data.py\n")
        sys.exit(2)

    L = 1000000 # length of sequence to generate

    # http://jaspar.binf.ku.dk/cgi-bin/jaspar_db.pl?ID=MA0125.1&rm=present&collection=CORE
    pwm_1 = tools.load_motif_from_PSAM("example/jaspar_psams/MA0125.psam")

    #http://jaspar.binf.ku.dk/cgi-bin/jaspar_db.pl?ID=MA0078.1&rm=present&collection=CORE
    pwm_2 = tools.load_motif_from_PSAM("example/jaspar_psams/MA0078.1.psam")
    pwms = [pwm_1, pwm_2]

    # these are the probabilities of their corresponding motifs overlapping
    # with a given K base long segment of sequence.
    gamma_motifs = np.array([0.07, 0.08])

    # this is the probability that a given position is the first base of a
    # motif. We arrive at this probability because each occurence of a
    # motif provides 2*K-1 offsets that overlap with the motif.
    gamma_motifs_st = []
    for i, gamma_m in enumerate(gamma_motifs):
        K = len(pwms[i])
        gamma_motifs_st.append(gamma_m/(2.0*K - 1.0))
    background_p = 1.0 - sum(gamma_motifs_st)

    start_probs = [background_p]+list(gamma_motifs_st)

    pwms_rc = []
    for pwm in pwms:
        pwms_rc.append(fliplr(flipud(pwm)))

    seq = ""
    sys.stderr.write("\n") # to log progress.
    print ">fake_sequence" # must add fasta header.
    for l in range(L):
        i = choice(range(0, len(pwms)+1), p=start_probs)
        if l%1000 == 0: sys.stderr.write("\t\t\t\rl = %d\t/%d\t\t"%(l, L))
        if i == 0:
            sys.stdout.write(rand_base())
        else:
            if np.random.choice(2) == 0: # pick if reverse complement
                pwm = pwms_rc[int(i-1)]
            else:
                pwm = pwms[int(i-1)]
            sys.stdout.write(gen_motif(pwm))

    sys.stderr.write("\n")
    print seq

if __name__ == "__main__":
    main(sys.argv)
