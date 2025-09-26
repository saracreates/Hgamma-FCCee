import ROOT
import os
import yaml
import argparse

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Set up argument parser
parser = argparse.ArgumentParser(description="Run a specific analysis: H->jj with j=b,g,tau")
parser.add_argument(
    "--flavor", "-f",
    type=str,
    default="B",
    help="Choose from: B, G, TAU"
)
args, _ = parser.parse_known_args()  

config = load_config("/afs/cern.ch/work/l/lherrman/private/HiggsGamma/analysis/ourrepo/Hgamma-FCCee/config/config_365.yaml")
config_jj = load_config("/afs/cern.ch/work/l/lherrman/private/HiggsGamma/analysis/ourrepo/Hgamma-FCCee/config/config_jj_365.yaml")



# global parameters
intLumi        = 1.
intLumiLabel   = "L = 10.8 a b^{-1}" # FIXME: use config['intLumi'] to set this dynamically
ana_tex        = 'e^{+}e^{-} #rightarrow #gamma H'
delphesVersion = '3.4.2'
energy         = config['ecm']
collider       = 'FCC-ee'
formats        = ['png','pdf']

outdir         = os.path.join(config['outputDir'], str(energy),'plots/',config_jj['outputDir_sub'], 'H{}{}'.format(args.flavor.lower(), args.flavor.lower()))
inputDir       = os.path.join(config['outputDir'], str(energy),'histmaker/', config_jj['outputDir_sub'], 'H{}{}'.format(args.flavor.lower(), args.flavor.lower()))
print(outdir)
print(inputDir)

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
colors['tt'] = ROOT.kGray+8

#procs = {}
#procs['signal'] = {'ZH':['wzp6_ee_mumuH_ecm240']}
#procs['backgrounds'] =  {'WW':['p8_ee_WW_ecm240'], 'ZZ':['p8_ee_ZZ_ecm240']}
procs = {}
procs['signal'] = {'AH':[f"mgp8_ee_ha_ecm{config['ecm']}_h{args.flavor.lower() + args.flavor.lower()}"]}



procs['backgrounds'] =  {
    'Aqq':[f"wzp6_ee_qqa_ecm{config['ecm']}"], 
    'Acc':[f"wzp6_ee_cca_ecm{config['ecm']}"], 
    'Abb':[f"wzp6_ee_bba_ecm{config['ecm']}"], 
    'Atautau':[f"wzp6_ee_tautaua_ecm{config['ecm']}"], 
    'Amumu':[f"wzp6_ee_mumua_ecm{config['ecm']}"], 
    'Aee':[f"wzp6_ee_eea_ecm{config['ecm']}"], 
    'WW':[f"p8_ee_WW_ecm{config['ecm']}"], 
    'ZZ':[f"p8_ee_ZZ_ecm{config['ecm']}"],
    'tt':[f"p8_ee_tt_ecm{config['ecm']}"],
    #'ZH':[f"mgp8_ee_zh_ecm{config['ecm']}"]
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
legend['tt'] = 'tt'


hists = {}
hists2D = {}


recoil_mass_min, recoil_mass_max = config['cuts']['recoil_mass_range']
signal_mass_min, signal_mass_max = config['cuts']['recoil_mass_signal_range']

m_jj_min, m_jj_max = config_jj['cuts']['m_jj_range'][args.flavor]
sum_jetscores = config_jj['cuts']['sum_jetscores_min'][args.flavor]


hists["cutFlow"] = {
    "input":   "cutFlow",
    "output":   "cutFlow",
    "logy":     True,
    "stack":   True,
    "xmin":     0,
    "xmax":     8,
    "ymin":     1e4,
    "ymax":     1e11,
    #"xtitle":   ["All events", "iso < 0.2", "60  < p_{#gamma} < 100 ", "|cos(#theta)_{#gamma}|<0.9", "n particles > 5"],
    "xtitle":   ["All events", "iso in treem","lepton veto","photon momentum", "cos theta","particle n cut",  "b score cut","mjj cut", "m recoil loose","m recoil tight"], 
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

hists["gamma_recoil_m_signal_cut"] = {
    "input":   "gamma_recoil_m_signal_cut",
    "output":   "gamma_recoil_m_signal_cut",
    "logy":     False,
    "stack":    True,
    "xmin":     110,
    "xmax":     150,
    "xtitle":   "Recoil (GeV)",
    "ytitle":   "Events ",
    "scaleSig": 100,
    "density": False
}


hists["photons_p_cut_2"] = {
    "input":   "photons_p_cut_2",
    "output":   "photons_p_cut_2",
    "logy":     False,
    "stack":    True,
    "xmin":     100,
    "xmax":     200,
    "xtitle":   "photon momentum",
    "ytitle":   "Events ",
    "scaleSig": 1000,
    "density": True
}

hists["photons_p_cut_1"] = {
    "input":   "photons_p_cut_1",
    "output":   "photons_p_cut_1",
    "logy":     False,
    "stack":    True,
    "xmin":     100,
    "xmax":     200,
    "xtitle":   "photon momentum",
    "ytitle":   "Events ",
    "scaleSig": 1000,
    "density": True
}



hists["m_jj_cut0"] = {
    "input":   "m_jj_cut0",
    "output":   "m_jj_cut0",
    "logy":     False,
    "stack":    True,
    "xmin":     0,
    "xmax":     200,
    "xtitle":   "Recoil (GeV)",
    "ytitle":   "Events ",
    "scaleSig": 100,
    "density": False
}

hists["m_jj_cut5"] = {
    "input":   "m_jj_cut5",
    "output":   "m_jj_cut5",
    "logy":     False,
    "stack":    True,
    "xmin":     0,
    "xmax":     200,
    "xtitle":   "Recoil (GeV)",
    "ytitle":   "Events ",
    "scaleSig": 100,
    "density": False
}

hists["m_jj_cut6"] = {
    "input":   "m_jj_cut6",
    "output":   "m_jj_cut6",
    "logy":     False,
    "stack":    True,
    "xmin":     0,
    "xmax":     200,
    "xtitle":   "Recoil (GeV)",
    "ytitle":   "Events ",
    "scaleSig": 100,
    "density": False
}

hists["scoresum_flavor"] = {
    "input":   "scoresum_flavor",
    "output":   "scoresum_flavor",
    "logy":     False,
    "stack":    True,
    "xmin":     0,
    "xmax":     2,
    "xtitle":   "flavor score sum (GeV)",
    "ytitle":   "Events ",
    "scaleSig": 100,
    "density": False
}

hists["recojet_isB0"] = {
    "input":   "recojet_isB0",
    "output":   "recojet_isB0",
    "logy":     False,
    "stack":    True,
    "xmin":     0.9,
    "xmax":     1,
    "xtitle":   "flavor 0 score",
    "ytitle":   "Events ",
    "scaleSig": 1000,
    "density": False
}

hists["recojet_isB1"] = {
    "input":   "recojet_isB1",
    "output":   "recojet_isB1",
    "logy":     False,
    "stack":    True,
    "xmin":     0.9,
    "xmax":     1,
    "xtitle":   "flavor 1 score",
    "ytitle":   "Events ",
    "scaleSig": 1000,
    "density": False
}


hists["num_isolated_leptons"] = {
    "input":   "num_isolated_leptons",
    "output":   "num_isolated_leptons",
    "logy":     False,
    "stack":    True,
    "xmin":     0,
    "xmax":     10,
    "xtitle":   "isolated leptons",
    "ytitle":   "Events ",
    "scaleSig": 1000,
    "density": False
}


hists["num_isolated_leptons_veto"] = {
    "input":   "num_isolated_leptons_veto",
    "output":   "num_isolated_leptons_veto",
    "logy":     False,
    "stack":    True,
    "xmin":     0,
    "xmax":     10,
    "xtitle":   "isolated leptons",
    "ytitle":   "Events ",
    "scaleSig": 1000,
    "density": False
}

