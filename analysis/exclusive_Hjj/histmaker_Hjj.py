import os, copy
import yaml
import argparse

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Set up argument parser
parser = argparse.ArgumentParser(description="Run a specific analysis: H->jj with j=b,g")
parser.add_argument(
    "--flavor", "-f",
    type=str,
    default="B",
    help="Choose from: B, G"
)
args, _ = parser.parse_known_args()  # <-- Ignore unknown args

if args.flavor not in ["B", "G"]:
    raise ValueError("Invalid flavor specified. Choose from: B, G")


config = load_config("/afs/cern.ch/work/l/lherrman/private/HiggsGamma/analysis/ourrepo/Hgamma-FCCee/config/config_365.yaml")
config_jj = load_config("/afs/cern.ch/work/l/lherrman/private/HiggsGamma/analysis/ourrepo/Hgamma-FCCee/config/config_jj_365.yaml")


print("Configuration:")
print(config)



ecm = config['ecm']
flavortag = args.flavor.lower() + args.flavor.lower() # bb or gg
br_flavor = config_jj['branching_ratios'][args.flavor]  # branching ratio for H->XX, e.g. H->bb or H->gg

# list of processes (mandatory)
processList = {}
for key, val in config['processList'].items():
    if key == 'mgp8_ee_ha':
        frac = float(val['fraction']) 
        br_WW = 0.215  # branching ratio for H->WW
        xsec = {'160': 2.127e-5 * br_flavor, '240': 8.773e-5 * br_flavor, '365': 2.975e-5 * br_flavor}.get(str(ecm), 0)
        entry = {
            'fraction': frac,
            'crossSection': xsec
        }
        processList[f"{key}_ecm{ecm}_h{flavortag}"] = entry
    else:
        entry = {
            'fraction': float(val['fraction']),
        }
        if 'crossSection' in val:
            entry['crossSection'] = float(val['crossSection'])  # optional
        if 'inputDir' in val:
            entry['inputDir'] = os.path.join(val['inputDir'], str(ecm))
        processList[f"{key}_ecm{ecm}"] = entry

print(processList)


# Link to the dictonary that contains all the cross section informations etc... (mandatory)
procDict = "FCCee_procDict_winter2023_IDEA.json"

# additional/custom C++ functions, defined in header files (optional)
includePaths = ["../functions.h"]

# Define the input dir (optional)
#inputDir    = "outputs/FCCee/higgs/mH-recoil/mumu/stage1"
#inputDir    = "/afs/cern.ch/work/l/lherrman/private/HiggsGamma/data"

#Optional: output directory, default is local running directory
outputDir   =  os.path.join(config['outputDir'], str(ecm),'histmaker/', config_jj['outputDir_sub'], 'H{}{}'.format(args.flavor.lower(), args.flavor.lower()))
print(outputDir)
inputDir   =  os.path.join(config['outputDir'], str(ecm),'treemaker/', config_jj['outputDir_sub'], 'H{}{}'.format(args.flavor.lower(), args.flavor.lower()))
# optional: ncpus, default is 4, -1 uses all cores available
nCPUS       = -1

# scale the histograms with the cross-section and integrated luminosity
doScale = True
intLumi = config['intLumi']


# define some binning for various histograms
bins_a_p = (100, 0, 500) # 100 MeV bins
bins_a_n = (10, 0, 10) # 100 MeV bins
bins_count = (10, 0, 10)
bins_score_sum = (100, 0, 2)


# name of collections in EDM root files
collections = {
    "GenParticles": "Particle",
    "PFParticles": "ReconstructedParticles",
    "PFTracks": "EFlowTrack",
    "PFPhotons": "EFlowPhoton",
    "PFNeutralHadrons": "EFlowNeutralHadron",
    "TrackState": "EFlowTrack_1",
    "TrackerHits": "TrackerHits",
    "CalorimeterHits": "CalorimeterHits",
    "dNdx": "EFlowTrack_2",
    "PathLength": "EFlowTrack_L",
    "Bz": "magFieldBz",
}

#cuts
photon_iso_cone_radius_min, photon_iso_cone_radius_max = config['cuts']['photon_iso_cone_radius_range']
photon_iso_threshold = config['cuts']['photon_iso_threshold']
photon_energy_min, photon_energy_max = config['cuts']['photon_energy_range']
photon_cos_theta_max = config['cuts']['photon_cos_theta_max']
recoil_mass_min, recoil_mass_max = config['cuts']['recoil_mass_range']
signal_mass_min, signal_mass_max = config['cuts']['recoil_mass_signal_range']
min_n_reco_no_gamma = config['cuts']['min_n_reco_no_gamma']

# jet clustering and tagging

## latest particle transformer model, trained on 9M jets in winter2023 samples
model_name = "fccee_flavtagging_edm4hep_wc"

## model files needed for unit testing in CI
url_model_dir = "https://fccsw.web.cern.ch/fccsw/testsamples/jet_flavour_tagging/winter2023/wc_pt_13_01_2022/"
url_preproc = "{}/{}.json".format(url_model_dir, model_name)
url_model = "{}/{}.onnx".format(url_model_dir, model_name)


## model files locally stored on /eos
model_dir = (
    "/eos/experiment/fcc/ee/jet_flavour_tagging/winter2023/wc_pt_7classes_12_04_2023/"
)
local_preproc = "{}/{}.json".format(model_dir, model_name)
local_model = "{}/{}.onnx".format(model_dir, model_name)

## get local file, else download from url
def get_file_path(url, filename):
    if os.path.exists(filename):
        return os.path.abspath(filename)
    else:
        urllib.request.urlretrieve(url, os.path.basename(url))
        return os.path.basename(url)


weaver_preproc = get_file_path(url_preproc, local_preproc)
weaver_model = get_file_path(url_model, local_model)

from addons.ONNXRuntime.jetFlavourHelper import JetFlavourHelper
from addons.FastJet.jetClusteringHelper import (
    ExclusiveJetClusteringHelper,
)

jetFlavourHelper = None
jetClusteringHelper = None


# build_graph function that contains the analysis logic, cuts and histograms (mandatory)
def build_graph(df, dataset):

  
    results = []
    df = df.Define("weight", "1.0")
    weightsum = df.Sum("weight")

    results.append(df.Histo1D(("m_jj_cut0", "", 100, 0, 200), "jj_m"))
   


    df = df.Define("photons_p", "FCCAnalyses::ReconstructedParticle::get_p(photons_all)") 
    df = df.Define("photons_n","FCCAnalyses::ReconstructedParticle::get_n(photons_all)")  #number of photons per event
    df = df.Define("photons_cos_theta","cos(FCCAnalyses::ReconstructedParticle::get_theta(photons_all))")
    


    #########
    ### CUT 0: all events
    #########
    df = df.Define("cut0", "0")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut0"))


    #Baseline selection
    results.append(df.Histo1D(("photons_p_cut_0", "", 130, 0, 130), "photons_p"))
    results.append(df.Histo1D(("photons_n_cut_0", "", *bins_a_n), "photons_n"))
    results.append(df.Histo1D(("photons_cos_theta_cut_0", "", 50, -1, 1), "photons_cos_theta"))


    #isolation cut
    df = df.Define("photons_iso", f"FCCAnalyses::ZHfunctions::coneIsolation({photon_iso_cone_radius_min}, {photon_iso_cone_radius_max})(photons_all, ReconstructedParticles)")  # is this correct?
    df = df.Define("photons_sel_iso",f"FCCAnalyses::ZHfunctions::sel_iso({photon_iso_threshold})(photons_all, photons_iso)",) # and this??
   
    df = df.Define("photons_iso_p", "FCCAnalyses::ReconstructedParticle::get_p(photons_sel_iso)") 
    df = df.Define("photons_iso_n","FCCAnalyses::ReconstructedParticle::get_n(photons_sel_iso)")  #number of photons per event
    df = df.Define("photons_iso_cos_theta","cos(FCCAnalyses::ReconstructedParticle::get_theta(photons_sel_iso))")

    results.append(df.Histo1D(("photon_isolation", "", 50, 0, 10), "photons_iso"))

    #######
    ### CUT 2: isolation (this one is already in treemaker)
    #########
    df = df.Filter("photons_sel_iso.size()>0 ")  
    df = df.Define("cut1", "1")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut1"))
  
    
    results.append(df.Histo1D(("photons_p_cut_1", "",  200, 0, 200), "photons_iso_p"))
    results.append(df.Histo1D(("photons_n_cut_1", "", *bins_a_n), "photons_iso_n"))
    results.append(df.Histo1D(("photons_cos_theta_cut_1", "", 50, -1, 1), "photons_iso_cos_theta"))
 
    

    #sort in p  and select highest energetic one
    df = df.Define("iso_highest_p","FCCAnalyses::ZHfunctions::sort_by_energy(photons_sel_iso)")

    #print and check
    #df = df.Define("photons_print", "FCCAnalyses::ZHfunctions::print_momentum(iso_highest_p)")
    #results.append(df.Histo1D(("photons_print", "", 100, 0, 100), "photons_print"))



    #energy cut
    df = df.Define("photons_boosted", f"FCCAnalyses::ReconstructedParticle::sel_p({photon_energy_min},{photon_energy_max})(iso_highest_p)") # looked okay from photons all
    #df = df.Define("photons_boosted", "FCCAnalyses::ReconstructedParticle::sel_p(60,100)(iso_highest_p)")

    df = df.Define("photons_boosted_p", "FCCAnalyses::ReconstructedParticle::get_p(photons_boosted)") # is this correct?
    df = df.Define("photons_boosted_n","FCCAnalyses::ReconstructedParticle::get_n(photons_boosted)") 
    df = df.Define("photons_boosted_cos_theta","cos(FCCAnalyses::ReconstructedParticle::get_theta(photons_boosted))")
    
    
     ##########
    ### CUT 1: isolated lepton veto
    ##########
    df = df.Define(
        "muons_iso",
        "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.5)(muons_all, ReconstructedParticles)",
    )
    df = df.Define(
        "muons_sel_iso",
        "FCCAnalyses::ZHfunctions::sel_iso(0.25)(muons_all, muons_iso)",
    )
    df = df.Define(
        "electrons_iso",
        "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.5)(electrons_all, ReconstructedParticles)",
    )
    df = df.Define(
        "electrons_sel_iso",
        "FCCAnalyses::ZHfunctions::sel_iso(0.25)(electrons_all, electrons_iso)",
    )

    df = df.Define("num_isolated_leptons", "electrons_sel_iso.size() + muons_sel_iso.size()")
    results.append(df.Histo1D(("num_isolated_leptons", "", 10, 0, 10), "num_isolated_leptons"))

    df = df.Filter("num_isolated_leptons == 0")  # no isolated lepton
    df = df.Define("cut2", "2")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut2"))

    results.append(df.Histo1D(("num_isolated_leptons_veto", "", 10, 0, 10), "num_isolated_leptons"))


    #######
    ### CUT 2: photon momentum
    #########
    df = df.Filter("photons_boosted.size()>0 ")  
    df = df.Define("cut3", "3")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut3"))
  
    
    results.append(df.Histo1D(("photons_p_cut_2", "",  200, 0, 200), "photons_boosted_p"))
    results.append(df.Histo1D(("photons_n_cut_2", "", *bins_a_n), "photons_boosted_n"))
    results.append(df.Histo1D(("photons_cos_theta_cut_2", "", 50, -1, 1), "photons_boosted_cos_theta"))
 
 
    #######
    ### CUT 3: cosine theta
    #########
    df = df.Filter(f"ROOT::VecOps::All(abs(photons_boosted_cos_theta) < {photon_cos_theta_max}) ") 
    df = df.Define("cut4", "4")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut4"))

    
    results.append(df.Histo1D(("photons_p_cut_3", "", 130, 0, 130), "photons_boosted_p"))
    results.append(df.Histo1D(("photons_n_cut_3", "", *bins_a_n), "photons_boosted_n"))
    results.append(df.Histo1D(("photons_cos_theta_cut_3", "", 50, -1, 1), "photons_boosted_cos_theta"))
    
    

    # recoil plot
    df = df.Define("gamma_recoil", f"FCCAnalyses::ReconstructedParticle::recoilBuilder({ecm})(photons_boosted)") 
    df = df.Define("gamma_recoil_m", "FCCAnalyses::ReconstructedParticle::get_mass(gamma_recoil)[0]") # recoil mass
    results.append(df.Histo1D(("gamma_recoil_m", "", 170, 80, 250), "gamma_recoil_m"))
    

    #you might want to remove this, without you get precision 110
    df = df.Define("recopart_no_gamma_n","FCCAnalyses::ReconstructedParticle::get_n(recopart_no_gamma)") 
    #########
    ### CUT 4: require at least 6 reconstructed particles (except gamma)
    #########
    df = df.Filter(f" recopart_no_gamma_n > {min_n_reco_no_gamma}") 
    df = df.Define("cut5", "5")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut5"))



    df = df.Define("recojet_isB0", "recojet_is{}[0]".format(args.flavor))
    df = df.Define("recojet_isB1", "recojet_is{}[1]".format(args.flavor))
    results.append(df.Histo1D(("recojet_isB0", "", *bins_score_sum), "recojet_isB0"))
    results.append(df.Histo1D(("recojet_isB1", "", *bins_score_sum), "recojet_isB1"))

    df = df.Define("scoresum_flavor", "recojet_is{}[0] + recojet_is{}[1]".format(args.flavor, args.flavor))
    results.append(df.Histo1D(("scoresum_flavor", "", *bins_score_sum), "scoresum_flavor"))
    


    # check missing momentum
    df = df.Define("missP", "FCCAnalyses::ZHfunctions::missingParticle(240.0, ReconstructedParticles)")
    df = df.Define("miss_p", "FCCAnalyses::ReconstructedParticle::get_p(missP)[0]")
    df = df.Define("miss_pT", "FCCAnalyses::ReconstructedParticle::get_pt(missP)[0]")
    results.append(df.Histo1D(("miss_p", "", 50, 0, 100), "miss_p"))
    results.append(df.Histo1D(("miss_pT", "", 50, 0, 100), "miss_pT"))
    
 
    #########
    ### Cut 5: sum of B-tagging scores > 1
    #########
    dic_jetscores = config_jj['cuts']['sum_jetscores_min']
    scoresum_min = dic_jetscores[args.flavor]
    # print("Using minimum sum of jet scores for {}: {}".format(args.flavor, scoresum_min))
    df = df.Filter("scoresum_flavor > {}".format(scoresum_min))  # minimum sum of jet scores
    df = df.Define("cut6", "6")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut6"))
 
  


   

    results.append(df.Histo1D(("miss_p_cut3", "", 50, 0, 100), "miss_p"))
    results.append(df.Histo1D(("miss_pT_cut3", "", 50, 0, 100), "miss_pT"))

   

    results.append(df.Histo1D(("m_jj_cut5", "", 100, 0, 200), "jj_m"))

    ##########
    ### CUT 6: Cut on inv mass of the two jets (Higgs mass)
    ##########
    mjj_min = config_jj['cuts']['m_jj_range'][args.flavor][0]
    mjj_max = config_jj['cuts']['m_jj_range'][args.flavor][1]
    df = df.Filter(f"{mjj_min} < jj_m && jj_m < {mjj_max}")  # Higgs mass range cut
    df = df.Define("cut7", "7")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut7"))

    results.append(df.Histo1D(("m_jj_cut6", "", 100, 0, 200), "jj_m"))


    #########
    ### CUT 4: gamma recoil cut
    #########
    df = df.Filter(f"{recoil_mass_min} < gamma_recoil_m && gamma_recoil_m < {recoil_mass_max}") 
    #df = df.Filter(f"{signal_mass_min} < gamma_recoil_m && gamma_recoil_m < {signal_mass_max}") 
    #df = df.Filter("115 < gamma_recoil_m && gamma_recoil_m < 170") 

    df = df.Define("cut8", "8")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut8"))

    results.append(df.Histo1D(("gamma_recoil_m_signal_cut", "", 40, 110, 150), "gamma_recoil_m"))
    #results.append(df.Histo1D(("gamma_recoil_m_signal_cut", "", 64, 116, 170), "gamma_recoil_m"))
     

   
    #########
    ### CUT 8: gamma recoil cut tight
    #########

    df = df.Filter(f"{signal_mass_min} < gamma_recoil_m && gamma_recoil_m < {signal_mass_max}") 
    df = df.Define("cut9", "9")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut9"))
    results.append(df.Histo1D(("gamma_recoil_m_tight_cut", "", 80, 110, 150), "gamma_recoil_m"))

   

    return results, weightsum