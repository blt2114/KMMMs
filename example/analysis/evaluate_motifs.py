"""evaluate_motifs aligns two learned motifs to the two synthetic motifs.  R
squared values are calculated and the true and fit parameters are plotted
against eachother.
"""
import sys
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from numpy import corrcoef, fliplr, flipud
from inference import tools

def compare_pwms(motif_1, motif_2):
    """compare_pwms compares two pwm motif representations

    Args:
        motif_1: np.array containing the true motif parameters
        motif_2: np.array containing the fit motif parameters

    Returns:
        The best R^2 value of a aligned segment, the aligned portion of the
            true motif, and the aligned portion of the fit motif.
    """
    K_2 = len(motif_2)
    K_1 = len(motif_1)
    min_len = np.min([K_1, K_2])

    best_corr = -np.inf
    best_m1_seg = None
    best_m2_seg = None

    for rev_comp in [False, True]:
        # this flip being the reverse complement relies on the 0th and 3rd
        # element being paired bases and the 1st and 2nd positions being
        # paired bases.  e.g. order must be ACGT
        if rev_comp:
            motif = flipud(fliplr(motif_1))
        else:
            motif = motif_1

        # sequences including the end of the m1 motif.
        for offset in range(3, min_len):
            m1_seg = motif[-offset:]
            m2_seg = motif_2[:offset]
            corr = corrcoef(m1_seg.flatten(), m2_seg.flatten())
            corr = corr[0, 1]
            if corr > best_corr:
                best_corr, best_m1_seg, best_m2_seg = corr, m1_seg, m2_seg

        # sequences including the beginning of the m1 motif.
        # if motifs are the same length, we repeat the full overlap, this is
        # fine.
        for offset in range(3, min_len):
            m1_seg = motif[:offset]
            m2_seg = motif_2[-offset:]
            corr = corrcoef(m1_seg.flatten(), m2_seg.flatten())
            corr = corr[0, 1]
            if corr > best_corr:
                best_corr, best_m1_seg, best_m2_seg = corr, m1_seg, m2_seg

        # sequences including the entirety of the smaller motif.
        if K_2 > K_1:
            start_idx = (K_2-K_1)+1
        elif K_1 > K_2:
            start_idx = (K_1-K_2)+1
        else:
            continue

        for i in range(0, start_idx):
            if K_2 > K_1:
                m1_seg = motif
                m2_seg = motif_2[i:i+min_len]
            else:
                m1_seg = motif[i:i+min_len]
                m2_seg = motif_2
            corr = corrcoef(m1_seg.flatten(), m2_seg.flatten())
            corr = corr[0, 1]
            if corr > best_corr:
                best_corr, best_m1_seg, best_m2_seg = corr, m1_seg, m2_seg

    m1_points = best_m1_seg.flatten()
    m2_points = best_m2_seg.flatten()
    return best_corr, m1_points, m2_points

def plot_comparison(axis, (x, y), title):
    """plot_comparison plots the one of the motif fits on the provided axis

    Args:
        axis: the plt axis object
        (x, y): the true and fit parameters
        title: the axis title
    """
    axis.scatter(x, y)
    axis.plot([0.0, 1.0], [0.0, 1.0], 'k-', lw=1.0)
    axis.set_title(title)
    axis.set_xlabel("True Parameter")
    axis.set_xticks(np.arange(0.0, 1.1, 0.1))
    axis.set_yticks(np.arange(0.0, 1.1, 0.1))
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlim(-0.05, 1.05)

def plot_figure(MA0125_true_points, MA0125_fit_points, MA0078_true_points,
                MA0078_fit_points):
    """plot_figure plots the figure comparing the two motifs.

    Args:
        MA0125_true_points: true motif params as np.array
        MA0125_fit_points: fit motif params as np.array
        MA0078_true_points: true motif params as np.array
        MA0078_fit_points: fit motif params as np.array
    """
    ### PLOT FIGURE ###
    font = {'size' : 18}

    matplotlib.rc('font', **font)
    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=13)

    f1 = plt.figure(figsize=(13, 6))
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(wspace=0.025)
    ax1 = plt.subplot(gs1[0])
    ax2 = plt.subplot(gs1[1])

    plot_comparison(ax1, (MA0125_true_points, MA0125_fit_points), "Nobox (MA0125.1)")
    plot_comparison(ax2, (MA0078_true_points, MA0078_fit_points), "Sox-17 (MA0078.1)")
    ax1.set_ylabel('Recovered Parameter')
    plt.setp(ax2.get_yticklabels(), visible=False)
    f1.savefig("2_motifs_synthetic.png",transparent=True)
    plt.show()

def main(argv):
    if len(argv) != 5:
        sys.stderr.write("python evaluate_motifs.py <MA0125_psam_fn> "
                         "<MA0078_psam_fn> <fit_motif_1> <fit_motif_2>\n")
        sys.exit(2)

    ### LOAD MOTIFS ###
    MA0125 = tools.load_motif_from_PSAM(argv[1])
    MA0078 = tools.load_motif_from_PSAM(argv[2])

    motif_1 = tools.load_motif_from_PSAM(argv[3])
    motif_2 = tools.load_motif_from_PSAM(argv[4])

    ### FIND BEST ALIGNMENT ###
    corr_1_to_1, MA0125_1, m1_points = compare_pwms(MA0125, motif_1)
    corr_1_to_2, MA0125_2, m2_points = compare_pwms(MA0125, motif_2)
    if corr_1_to_1 > corr_1_to_2:
        MA0125_true_points = MA0125_1
        MA0125_fit_points = m1_points
        MA0125_corr = corr_1_to_1
        MA0078_corr, MA0078_true_points, MA0078_fit_points = compare_pwms(MA0078, motif_2)
    else:
        MA0125_true_points = MA0125_2
        MA0125_fit_points = m2_points
        MA0125_corr = corr_1_to_2
        MA0078_corr, MA0078_true_points, MA0078_fit_points = compare_pwms(MA0078, motif_1)

    print "MA0125 R^2: %s"%str(MA0125_corr)
    print "MA0078 R^2: %s"%str(MA0078_corr)

    plot_figure(MA0125_true_points, MA0125_fit_points, MA0078_true_points,
                MA0078_fit_points)

if __name__ == "__main__":
    main(sys.argv)
