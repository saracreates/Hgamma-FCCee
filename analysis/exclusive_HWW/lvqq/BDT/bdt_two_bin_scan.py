import numpy as np
import matplotlib.pyplot as plt

bdt_cuts = [
    0.901, 0.905, 0.91, 0.915, 0.92, 0.925, 0.93, 0.935, 0.94, 0.945,
    0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.986, 0.987,
    0.988, 0.989, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996,
    0.997, 0.998, 0.999
]

prec_09x = [
    7117.7, 5786.4, 3393.6, 3021.0, 2515.0, 2140.3, 1841.5, 1647.6,
    1461.3, 1330.7, 1149.4, 1024.7, 907.0, 795.2, 714.9, 629.3, 544.5,
    442.8, 426.9, 410.3, 392.2, 371.2, 353.5, 331.4, 313.9, 290.9,
    267.4, 242.5, 220.0, 196.3, 165.2, 130.0
]

prec_x1 = [
    103.9, 101.9, 101.0, 98.0, 96.1, 94.1, 93.0, 90.7,
    88.9, 85.8, 85.5, 83.5, 81.7, 80.3, 77.2, 74.0, 70.7, 71.9,
    70.6, 69.9, 69.3, 69.3, 68.1, 69.6, 67.5, 68.8, 70.8, 72.9,
    72.7, 69.8, 79.0, 87.0
]


if len(bdt_cuts) != len(prec_09x):
    raise ValueError("bdt_cuts and prec_09x must have the same length")
if len(bdt_cuts) != len(prec_x1):
    raise ValueError("bdt_cuts and prec_x1 must have the same length")




# Calculate matrix: combined precision 

def cal_error(x1, x2):
    # gewichteter Mittelwert!
    return 1/ np.sqrt(1/(x1**2) + 1/(x2**2))

comb_prec = cal_error(np.array(prec_09x), np.array(prec_x1))
print("Combined precision:", comb_prec)
print("Smallest combined precision:", np.min(comb_prec), " with BDT cut:", bdt_cuts[np.argmin(comb_prec)])

# Smallest combined precision: 65.76611535875608  with BDT cut: 0.997 


print("Smallest x1 precision:", np.min(prec_x1), " with BDT cut:", bdt_cuts[np.argmin(prec_x1)])
