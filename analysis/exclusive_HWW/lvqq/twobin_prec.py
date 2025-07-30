import np as np

prec_bin1 = 56
prec_bins = 100

def comb_prec(x1, x2):
    # gewichteter Mittelwert!
    return 1/ np.sqrt(1/(x1**2) + 1/(x2**2))