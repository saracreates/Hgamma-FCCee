import ROOT
import os
import yaml
import argparse
import json

def load_colors(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Set up argument parser
parser = argparse.ArgumentParser(description="Run analysis for H->WW(lv)W(qq).")
parser.add_argument(
    "--energy", "-e",
    type=int,
    default=160,
    help="Choose from: 160, 240, 365. Default: 160"
)
args, _ = parser.parse_known_args()  # <-- Ignore unknown args

print("Loading configuration for energy:", args.energy)

config = load_config(f"config/config_{args.energy}.yaml")
config_WW = load_config(f"config/config_WW_qqlv_{args.energy}.yaml")



# global parameters
intLumi        = 1.
intLumiLabel   = "L = 10.8 a b^{-1}" # FIXME: use config['intLumi'] to set this dynamically
ana_tex        = 'e^{+}e^{-} #rightarrow #gamma H'
delphesVersion = '3.4.2'
energy         = config['ecm']
collider       = 'FCC-ee'
formats        = ['png','pdf']

outdir         = os.path.join(config['outputDir'], str(energy),'plots/', config_WW['outputDir_sub']) 
inputDir       = os.path.join(config['outputDir'], str(energy),'histmaker/', config_WW['outputDir_sub'])

plotStatUnc    = True

# load colors from json
colors_dic = load_colors("extras/colors.json")
colors = {}
for key, value in colors_dic.items():
    colors[key] = ROOT.TColor.GetColor(value)


procs = {}
procs['signal'] = {'AH':[f"mgp8_ee_ha_ecm{config['ecm']}_hww"]}
# procs['signal'] = {'AH':[f"p8_ee_Hgamma_ecm{config['ecm']}"]}

procs['backgrounds'] =  {
    'Aqq':[f"wzp6_ee_qqa_ecm{config['ecm']}"], 
    'Acc':[f"wzp6_ee_cca_ecm{config['ecm']}"], 
    'Abb':[f"wzp6_ee_bba_ecm{config['ecm']}"], 
    'Atautau':[f"wzp6_ee_tautaua_ecm{config['ecm']}"], 
    'Amumu':[f"wzp6_ee_mumua_ecm{config['ecm']}"], 
    'Aee':[f"wzp6_ee_eea_ecm{config['ecm']}"], 
    'WW':[f"p8_ee_WW_ecm{config['ecm']}"], 
    'WWA':[f"wzp6_ee_WWa_ecm{config['ecm']}"],
    'ZZ':[f"p8_ee_ZZ_ecm{config['ecm']}"],
    'ZH':[f"mgp8_ee_zh_ecm{config['ecm']}"]
}  

legend = {}
legend['AH'] = '#gamma H'
legend['Aqq'] = '#gamma q#bar{q}'
legend['Acc'] = '#gamma c#bar{c}'
legend['Abb'] = '#gamma b#bar{b}'
legend['WW'] = 'WW'
legend['ZZ'] = 'ZZ'
legend['Aee'] = '#gamma e^{+} e^{-}'
legend['Atautau'] = '#gamma #tau^{+} #tau^{-}'
legend['Amumu'] = '#gamma #mu^{+} #mu^{-}'
legend['ZH'] = 'ZH'


hists = {}
hists2D = {}


recoil_mass_min, recoil_mass_max = config['cuts']['recoil_mass_range']
signal_mass_min, signal_mass_max = config['cuts']['recoil_mass_signal_range']

m_jj_min, m_jj_max = config_WW['cuts']['m_jj_range']
# recoil_gammaqq_min, recoil_gammaqq_max = config_WW['cuts']['recoil_gammaqq_range']
do_inference = config_WW['do_inference']

xtitle = ["All events", f"iso < {config['cuts']['photon_iso_threshold']}", str(config['cuts']['photon_energy_range'][0]) + "< p_{#gamma} < " + str(config['cuts']['photon_energy_range'][1]), "|cos(#theta)_{#gamma}|<" + str(config['cuts']['photon_cos_theta_max']), f"n particles > {config['cuts']['min_n_reco_no_gamma']}", str(recoil_mass_min) + " < m_{recoil} < " + str(recoil_mass_max), "1 iso lepton", str(m_jj_min) + "< m_{qq} <" + str(m_jj_max), "Num const per jet > " + str(config_WW['cuts']['n_const_per_jet']),  str(signal_mass_min) + " < m_{recoil} < " + str(signal_mass_max)] # "|cos(#theta)_{W}|<" + str(WW_cos_theta_max), str(recoil_gammaqq_min) + "<m_{recoil, #gamma qq} < " + str(recoil_gammaqq_max), "p_{T, lep}>"+ str(config_WW['cuts']['pT_lepton']), "p_{T, miss} > " + str(config_WW['cuts']['pT_miss']),

if do_inference:
    xtitle.insert(-1, "BDT score > " + str(config_WW['cuts']['mva_score_cut']))
    
    # BDT scan
    
    # # remove last element
    # xtitle.pop(-1)
    # # append BDT score cut scan 
    # for mva_cut_value in config_WW['cuts']['mva_score_cut']:
    #     xtitle.append(f"BDT score > {mva_cut_value}")


hists["cutFlow"] = {
    "input":   "cutFlow",
    "output":   "cutFlow",
    "logy":     True,
    "stack":   True,
    "xmin":     0,
    "xmax":     7,
    "ymin":     1e4,
    "ymax":     1e11,
    #"xtitle":   ["All events", "iso < 0.2", "60  < p_{#gamma} < 100 ", "|cos(#theta)_{#gamma}|<0.9", "n particles > 5"],
    "xtitle":   xtitle,
    "ytitle":   "Events ",
}


hists["gamma_recoil_m_tight_cut"] = {
    "input":   "gamma_recoil_m_tight_cut",
    "output":   "gamma_recoil_m_tight_cut",
    "logy":     False,
    "stack":    True,
    "xmin":     110,
    "xmax":     150,
    "xtitle":   "Recoil (GeV)",
    "ytitle":   "Events ",
    "scaleSig": 100,
    "density": False
}

hists["gamma_recoil_m_very_tight_cut"] = {
    "input":   "gamma_recoil_m_very_tight_cut",
    "output":   "gamma_recoil_m_very_tight_cut",
    "logy":     False,
    "stack":    True,
    "xmin":     123, # 120
    "xmax":     130, # 135
    "xtitle":   "Recoil (GeV)",
    "ytitle":   "Events ",
    "scaleSig": 10,
    "density": False
}