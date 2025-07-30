import numpy as np

prec_bin1 = 63.3
prec_bin2 = 197.1

def comb_prec(x1, x2):
    # gewichteter Mittelwert!
    return 1/ np.sqrt(1/(x1**2) + 1/(x2**2))

print("Combined precision for bin1 and bin2: ", comb_prec(prec_bin1, prec_bin2)) # 60.26817830558345