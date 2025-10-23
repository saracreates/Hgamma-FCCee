
import uproot
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import ROOT
import pickle
import argparse
import os
import yaml

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


ROOT.gROOT.SetBatch(True)
# e.g. https://root.cern/doc/master/tmva101__Training_8py.html


parser = argparse.ArgumentParser()
parser.add_argument(
    "--flavor", "-f",
    type=str,
    default="B",
    help="Choose from: B, G, TAU"
)
parser.add_argument(
    "--config", "-c",
    type=int,
    default=240,
    help="Choose from: 160, 240,365"
)

args = parser.parse_args()

if args.config == 160:
    config = load_config("/afs/cern.ch/work/l/lherrman/private/HiggsGamma/analysis/ourrepo/Hgamma-FCCee/config/config_test_160.yaml")
    config_jj = load_config("/afs/cern.ch/work/l/lherrman/private/HiggsGamma/analysis/ourrepo/Hgamma-FCCee/config/config_jj_160.yaml")
elif args.config == 240:
    config = load_config("/afs/cern.ch/work/l/lherrman/private/HiggsGamma/analysis/ourrepo/Hgamma-FCCee/config/config_240.yaml")
    config_jj = load_config("/afs/cern.ch/work/l/lherrman/private/HiggsGamma/analysis/ourrepo/Hgamma-FCCee/config/config_jj_240.yaml")
elif args.config == 365:
    config = load_config("/afs/cern.ch/work/l/lherrman/private/HiggsGamma/analysis/ourrepo/Hgamma-FCCee/config/config_test.yaml")
    config_jj = load_config("/afs/cern.ch/work/l/lherrman/private/HiggsGamma/analysis/ourrepo/Hgamma-FCCee/config/config_jj_365.yaml")


def load_process(fIn, variables, target=0, weight_sf=1.):

    f = uproot.open(fIn)
    tree = f["events;1"]
    #meta = f["meta"]
    #weight = meta.values()[2]/meta.values()[1]*weight_sf
    weight = 1.0/tree.num_entries*weight_sf
    print("Load {} with {} events and weight {}".format(fIn.replace(".root", ""), tree.num_entries, weight))
    print("Variables: ", variables)
    print("Variables from tree: ", tree.keys())
    arrays = tree.arrays(variables, library="np") # convert the signal and background data to pandas DataFrames
    df = pd.DataFrame({var: arrays[var] for var in variables})
    df['target'] = target # add a target column to indicate signal (1) and background (0)
    df['weight'] = weight
    return df



print("Parse inputs")

# configuration of signal, background, variables, files, ...
#variables = ["m_cut","jet_energy_ratio","cos_jet_dist","photons_boosted_p", "jj_m"]
#variables = ["m_cut","jet_energy_ratio","cos_jet_dist"]
variables = ["m_cut",
            "jet_energy_ratio",
            "cos_jet_dist",
            "jj_m",
            "photons_boosted_p",
            "photons_boosted_n",
            "photons_boosted_cos_theta",
            "recopart_no_gamma_n",
            "gamma_recoil_m",
            "miss_p", 
            "miss_pT", 
            "jet0_costheta", 
            "jet1_costheta", 
            "jet0_cosphi", 
            "jet1_cosphi",]
weight_sf = 1e9
ecm = config['ecm']

path = os.path.join(config['outputDir'], str(ecm),'treemaker/BDT/', config_jj['outputDir_sub'], 'H{}{}'.format(args.flavor.lower(), args.flavor.lower()))
print(path)
sig_df = load_process(path + f"/mgp8_ee_ha_ecm{ecm}_hbb.root", variables, weight_sf=weight_sf, target=1)
# bkg = load_process(path + f"p8_ee_WW_ecm{args.energy}.root", variables, weight_sf=weight_sf)
bkg = load_process(path + f"/wzp6_ee_bba_ecm{ecm}.root", variables, weight_sf=weight_sf)


# Concatenate the dataframes into a single dataframe
data = pd.concat([sig_df, bkg], ignore_index=True)


# split data in train/test events
train_data, test_data, train_labels, test_labels, train_weights, test_weights  = train_test_split(
    data[variables], data['target'], data['weight'], test_size=0.2, random_state=42
)


# conversion to numpy needed to have default feature_names (fN), needed for conversion to TMVA
train_data = train_data.to_numpy()
test_data = test_data.to_numpy()
train_labels = train_labels.to_numpy()
test_labels = test_labels.to_numpy()
train_weights = train_weights.to_numpy()
test_weights = test_weights.to_numpy()


# set hyperparameters for the XGBoost model
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.1,
    'max_depth': 5,
    'subsample': 0.5,
    'colsample_bytree': 0.5,
    'seed': 42,
    'n_estimators': 350, 
    'early_stopping_rounds': 25,
    'num_rounds': 20,
    'learning_rate': 0.1,
    'gamma': 3,
    'min_child_weight': 10,
    'max_delta_step': 0,
}
"""

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42,
    'n_estimators': 2000, 
    'early_stopping_rounds': 25,
    'learning_rate': 0.05,
    'gamma': 3,
    'min_child_weight': 20,
    'max_delta_step': 0,
    'reg_lambda':1.0,
    'reg_alpha':0.1,
}

"""


# train the XGBoost model
print("Start training")
eval_set = [(train_data, train_labels), (test_data, test_labels)]
bdt = xgb.XGBClassifier(**params)
bdt.fit(train_data, train_labels, verbose=True, eval_set=eval_set, sample_weight=train_weights)


# export model (to ROOT and pkl)
print("Export model")
fOutName = os.path.join(config['outputDir'], str(ecm),'treemaker/BDT/', config_jj['outputDir_sub'], 'H{}{}'.format(args.flavor.lower(), args.flavor.lower()), 'bdt_model_example.root')
ROOT.TMVA.Experimental.SaveXGBoost(bdt, "bdt_model", fOutName, num_inputs=len(variables))

# append the variables
variables_ = ROOT.TList()
for var in variables:
     variables_.Add(ROOT.TObjString(var))
fOut = ROOT.TFile(fOutName, "UPDATE")
fOut.WriteObject(variables_, "variables")


save = {}
save['model'] = bdt
save['train_data'] = train_data
save['test_data'] = test_data
save['train_labels'] = train_labels
save['test_labels'] = test_labels
save['variables'] = variables
pickle.dump(save, open(os.path.join(config['outputDir'], str(ecm),'treemaker/BDT/', config_jj['outputDir_sub'], 'H{}{}'.format(args.flavor.lower(), args.flavor.lower()), 'bdt_model_example.pkl'), "wb"))