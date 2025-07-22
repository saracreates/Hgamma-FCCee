import uproot 
import numpy as np
import matplotlib.pyplot as plt

from helper_rocs import *

save_status = True  # Set to False if you do not want to save the plots

print("Loading data for ROC curves...")

# performance of the tagger in my manual FCCAnalyses implementation
# path1 = "outputs/jettags/Hgamma/p8_ee_Hgamma_ecm240_jetbased.root"
path1 = "outputs/jettags/Hgamma/wzp6_ee_nunuH_Hbb_ecm240.root"
data1 = load_data(path1, key='events;1')

# performance of the tagger with key4hep inference
path2 = "/eos/experiment/fcc/ee/datasets/CLD_fullsim_tagging_results_key4hep/CLD_v5/hadded/results_key4hep.root"
# path2 = "/eos/experiment/fcc/ee/datasets/CLD_fullsim_tagging_results_key4hep/CLD_v7/hadded_Hxx.root"
# data2 = load_data(path2)

# all_rocs(data1, data2, 'FCCAnalyses', 'key4hep', save=save_status, name='roc-FCCAnalyses-check')

# single_roc(data1,'HZ-240', save=save_status, name='roc-HZ-240')

print("Plotted ROC curves.")