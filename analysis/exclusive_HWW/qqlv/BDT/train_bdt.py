
import uproot
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import ROOT
import pickle
import argparse


ROOT.gROOT.SetBatch(True)
# e.g. https://root.cern/doc/master/tmva101__Training_8py.html


parser = argparse.ArgumentParser()
parser.add_argument(
    "--energy", "-e",
    type=int,
    default=240,
    help="Choose from: 160, 240, 365. Default: 240"
)

args = parser.parse_args()

def load_process(fIn, variables, target=0, weight_sf=1.):

    f = uproot.open(fIn)
    tree = f["events;1"]
    #meta = f["meta"]
    #weight = meta.values()[2]/meta.values()[1]*weight_sf
    weight = 1.0/tree.num_entries*weight_sf
    print("Load {} with {} events and weight {}".format(fIn.replace(".root", ""), tree.num_entries, weight))
    print("Variables: ", variables)
    print("Variables from tree: ", tree.keys())
    df = tree.arrays(variables, library="pd") # convert the signal and background data to pandas DataFrames
    df['target'] = target # add a target column to indicate signal (1) and background (0)
    df['weight'] = weight
    return df



print("Parse inputs")

# configuration of signal, background, variables, files, ...
variables = ["photon_p", "photons_boosted_n", "photon_cos_theta", "recopart_no_gamma_n",
                "gamma_recoil_m", "num_isolated_leptons", "lepton_p", "lepton_pT", "y23", "y34",
                "jet1_nconst_N2", "jet2_nconst_N2", "m_jj", "miss_p", "miss_pT", 
                "recoil_W_m", "W_qq_unboosted_costheta", "W_lv_unboosted_costheta",
                "jet1_costheta", "jet2_costheta", "jet1_cosphi", "jet2_cosphi",
                "W_qq_pT", "W_lv_pT",]
weight_sf = 1e9

path = f"outputs/{args.energy}/preselection/qqlv/trainingdata/"
sig_df = load_process(path + f"mgp8_ee_ha_ecm{args.energy}_hww.root", variables, weight_sf=weight_sf, target=1)
bkg = load_process(path + f"p8_ee_WW_ecm{args.energy}.root", variables, weight_sf=weight_sf)

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
    'n_estimators': 350, # low number for testing purposes (default 350)
    'early_stopping_rounds': 25,
    'num_rounds': 20,
    'learning_rate': 0.20,
    'gamma': 3,
    'min_child_weight': 10,
    'max_delta_step': 0,
}


# train the XGBoost model
print("Start training")
eval_set = [(train_data, train_labels), (test_data, test_labels)]
bdt = xgb.XGBClassifier(**params)
bdt.fit(train_data, train_labels, verbose=True, eval_set=eval_set, sample_weight=train_weights)


# export model (to ROOT and pkl)
print("Export model")
fOutName = f"outputs/{args.energy}/BDT/qqlv/bdt_model_example.root"
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
pickle.dump(save, open(f"outputs/{args.energy}/BDT/qqlv/bdt_model_example.pkl", "wb"))