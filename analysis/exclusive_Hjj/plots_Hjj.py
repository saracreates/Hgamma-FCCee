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

config = load_config("config/config_240.yaml")
config_jj = load_config("config/config_jj_240.yaml")


# global parameters
intLumi        = 1.
intLumiLabel   = "L = 10.8 a b^{-1}" # FIXME: use config['intLumi'] to set this dynamically
ana_tex        = 'e^{+}e^{-} #rightarrow #gamma H'
delphesVersion = '3.4.2'
energy         = config['ecm']
collider       = 'FCC-ee'
formats        = ['png','pdf']

outdir         = os.path.join(config['outputDir'], str(energy),'plots/',config_jj['outputDir_sub'], 'H{}{}'.format(args.flavor.lower(), args.flavor.lower()))
inputDir       =os.path.join(config['outputDir'], str(energy),'histmaker/', config_jj['outputDir_sub'], 'H{}{}'.format(args.flavor.lower(), args.flavor.lower()))

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
#colors['ZH'] = ROOT.kGray+2

#procs = {}
#procs['signal'] = {'ZH':['wzp6_ee_mumuH_ecm240']}
#procs['backgrounds'] =  {'WW':['p8_ee_WW_ecm240'], 'ZZ':['p8_ee_ZZ_ecm240']}
procs = {}
procs['signal'] = {'AH':[f"mgp8_ee_ha_ecm{config['ecm']}_h{args.flavor.lower() + args.flavor.lower()}"]}
# procs['signal'] = {'AH':[f"p8_ee_Hgamma_ecm{config['ecm']}"]}


procs['backgrounds'] =  {
    'Aqq':[f"wzp6_ee_qqa_ecm{config['ecm']}"], 
    'Acc':[f"wzp6_ee_cca_ecm{config['ecm']}"], 
    'Abb':[f"wzp6_ee_bba_ecm{config['ecm']}"], 
    'Atautau':[f"wzp6_ee_tautaua_ecm{config['ecm']}"], 
    'Amumu':[f"wzp6_ee_mumua_ecm{config['ecm']}"], 
    'Aee':[f"wzp6_ee_eea_ecm{config['ecm']}"], 
    'WW':[f"p8_ee_WW_ecm{config['ecm']}"], 
    'ZZ':[f"p8_ee_ZZ_ecm{config['ecm']}"]}

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
#legend['ZH'] = 'ZH'


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
    "xmax":     7,
    "ymin":     1e4,
    "ymax":     1e11,
    #"xtitle":   ["All events", "iso < 0.2", "60  < p_{#gamma} < 100 ", "|cos(#theta)_{#gamma}|<0.9", "n particles > 5"],
    "xtitle":   ["All events", f"iso < {config['cuts']['photon_iso_threshold']}", str(config['cuts']['photon_energy_range'][0]) + "< p_{#gamma} < " + str(config['cuts']['photon_energy_range'][1]), "|cos(#theta)_{#gamma}|<" + str(config['cuts']['photon_cos_theta_max']), f"n particles > {config['cuts']['min_n_reco_no_gamma']}", str(recoil_mass_min) + " < m_{recoil} < " + str(recoil_mass_max), str(sum_jetscores) + " < sum jet scores", str(m_jj_min) + "< m_{jj} < " + str(m_jj_max), str(signal_mass_min) + " < m_{recoil} < " + str(signal_mass_max)],
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