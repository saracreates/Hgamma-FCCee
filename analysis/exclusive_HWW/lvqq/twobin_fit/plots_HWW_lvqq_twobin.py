import ROOT
import os
import yaml
import argparse
import re

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def extract_bin(label):
    match = re.search(r'bin[12]', label)
    if match:
        return match.group()
    else:
        raise ValueError("Label does not contain 'bin1' or 'bin2': {}".format(label))

# Set up argument parser
parser = argparse.ArgumentParser(description="Run two bin analysis for H->WW(lv)W(qq).")
parser.add_argument(
    "--bins", "-b",
    type=int,
    default=1,
    help="Choose from: 1 (0.9-0.997), 2 (0.997-1) for the two bin analysis"
)
args, _ = parser.parse_known_args()  # <-- Ignore unknown args

# check args
if args.bins not in [1, 2]:
    raise ValueError("Invalid bin option. Choose from: 1, 2")

config = load_config("config/config_240.yaml")
config_WW = load_config("config/config_WW_lvqq_twobin_240.yaml")

# global parameters
intLumi        = 1.
intLumiLabel   = "L = 10.8 a b^{-1}" # FIXME: use config['intLumi'] to set this dynamically
ana_tex        = 'e^{+}e^{-} #rightarrow #gamma H'
delphesVersion = '3.4.2'
energy         = config['ecm']
collider       = 'FCC-ee'
formats        = ['png','pdf']

outdir         = os.path.join(config['outputDir'], str(energy),'plots/', config_WW['outputDir_sub'][args.bins-1]) 
inputDir       = os.path.join(config['outputDir'], str(energy),'histmaker/', config_WW['outputDir_sub'][args.bins-1])

plotStatUnc    = True

colors = {}
colors['AH'] = ROOT.kRed
colors['Acc'] = ROOT.kBlue+1
colors['Aqq'] = ROOT.kGreen+2
colors['Abb'] = ROOT.kYellow+3
colors['WW'] = ROOT.kCyan
colors['ZZ'] = ROOT.kAzure-9
colors['Aee'] = ROOT.kViolet+3
colors['Atautau'] = ROOT.kOrange
colors['Amumu'] = ROOT.kMagenta
colors['ZH'] = ROOT.kGray+2

#procs = {}
#procs['signal'] = {'ZH':['wzp6_ee_mumuH_ecm240']}
#procs['backgrounds'] =  {'WW':['p8_ee_WW_ecm240'], 'ZZ':['p8_ee_ZZ_ecm240']}
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
recoil_gammaqq_min, recoil_gammaqq_max = config_WW['cuts']['recoil_gammaqq_range']
do_inference = config_WW['do_inference']

xtitle = ["All events", f"iso < {config['cuts']['photon_iso_threshold']}", str(config['cuts']['photon_energy_range'][0]) + "< p_{#gamma} < " + str(config['cuts']['photon_energy_range'][1]), "|cos(#theta)_{#gamma}|<" + str(config['cuts']['photon_cos_theta_max']), f"n particles > {config['cuts']['min_n_reco_no_gamma']}", str(recoil_mass_min) + " < m_{recoil} < " + str(recoil_mass_max), "1 iso lepton", str(m_jj_min) + "< m_{qq} <" + str(m_jj_max), "pT_{miss} > " + str(config_WW['cuts']['pT_miss']), str(recoil_gammaqq_min) + "<m_{recoil, #gamma qq} < " + str(recoil_gammaqq_max), "#const per jet > " + str(config_WW['cuts']['n_const_per_jet']), str(signal_mass_min) + " < m_{recoil} < " + str(signal_mass_max)] #"p_{miss} > 20","p_{T} > 10"

if do_inference:
    bdt_min, bdt_max = config_WW['cuts']['mva_score_cut'][extract_bin(config_WW['outputDir_sub'][args.bins-1])]
    xtitle.insert(-1, f"{bdt_min} < BDT score < {bdt_max}")


hists["cutFlow"] = {
    "input":   "cutFlow",
    "output":   "cutFlow",
    "logy":     True,
    "stack":   True,
    "xmin":     0,
    "xmax":     7,
    "ymin":     1e4,
    "ymax":     1e11,
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
    "scaleSig": 10,
    "density": False
}