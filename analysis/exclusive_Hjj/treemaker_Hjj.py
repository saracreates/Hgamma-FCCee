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
    help="Choose from: B, G, TAU"
)
parser.add_argument(
    "--config", "-c",
    type=int,
    default=240,
    help="Choose from: 160, 240,365"
)
args, _ = parser.parse_known_args()  # <-- Ignore unknown args

if args.flavor not in ["B", "G", "TAU"]:
    raise ValueError("Invalid flavor specified. Choose from: B, G")

if args.config == 160:
    config = load_config("/afs/cern.ch/work/l/lherrman/private/HiggsGamma/analysis/ourrepo/Hgamma-FCCee/config/config_160.yaml")
    config_jj = load_config("/afs/cern.ch/work/l/lherrman/private/HiggsGamma/analysis/ourrepo/Hgamma-FCCee/config/config_jj_160.yaml")
elif args.config == 240:
    config = load_config("/afs/cern.ch/work/l/lherrman/private/HiggsGamma/analysis/ourrepo/Hgamma-FCCee/config/config_240.yaml")
    config_jj = load_config("/afs/cern.ch/work/l/lherrman/private/HiggsGamma/analysis/ourrepo/Hgamma-FCCee/config/config_jj_240.yaml")
elif args.config == 365:
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


# Production tag when running over EDM4Hep centrally produced events, this points to the yaml files for getting sample statistics (mandatory)
prodTag     = "FCCee/winter2023/IDEA/"

# Link to the dictonary that contains all the cross section informations etc... (mandatory)
procDict = "FCCee_procDict_winter2023_IDEA.json"

# additional/custom C++ functions, defined in header files (optional)
includePaths = ["../functions.h"]

# Define the input dir (optional)
#inputDir    = "outputs/FCCee/higgs/mH-recoil/mumu/stage1"
#inputDir    = "/afs/cern.ch/work/l/lherrman/private/HiggsGamma/data"

#Optional: output directory, default is local running directory
outputDir   =  os.path.join(config['outputDir'], str(ecm),'treemaker/', config_jj['outputDir_sub'], 'H{}{}'.format(args.flavor.lower(), args.flavor.lower()))
print(outputDir)

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


# Mandatory: RDFanalysis class where the use defines the operations on the TTree
class RDFanalysis:

    # __________________________________________________________
    # Mandatory: analysers funtion to define the analysers to process, please make sure you return the last dataframe, in this example it is df2
    def analysers(df):
       
        # __________________________________________________________
        # Mandatory: analysers funtion to define the analysers to process, please make sure you return the last dataframe, in this example it is df2
        results = []
        # define some aliases to be used later on
        df = df.Alias("Particle0", "Particle#0.index")
        df = df.Alias("Particle1", "Particle#1.index")
        df = df.Alias("MCRecoAssociations0", "MCRecoAssociations#0.index")
        df = df.Alias("MCRecoAssociations1", "MCRecoAssociations#1.index")
        df = df.Alias("Photon0", "Photon#0.index")
        df = df.Alias("Electron0", "Electron#0.index")
        df = df.Alias("Muon0", "Muon#0.index")
        # get all the leptons from the collection
       
        df = df.Define("photons_all","FCCAnalyses::ReconstructedParticle::get(Photon0, ReconstructedParticles)",)
        df = df.Define("electrons_all", "FCCAnalyses::ReconstructedParticle::get(Electron0, ReconstructedParticles)")
        df = df.Define("muons_all", "FCCAnalyses::ReconstructedParticle::get(Muon0, ReconstructedParticles)")
        

        df = df.Define("photons_p", "FCCAnalyses::ReconstructedParticle::get_p(photons_all)") 
        df = df.Define("photons_n","FCCAnalyses::ReconstructedParticle::get_n(photons_all)")  #number of photons per event
        df = df.Define("photons_cos_theta","cos(FCCAnalyses::ReconstructedParticle::get_theta(photons_all))")

    
        #isolation cut
        df = df.Define("photons_iso", f"FCCAnalyses::ZHfunctions::coneIsolation({photon_iso_cone_radius_min}, {photon_iso_cone_radius_max})(photons_all, ReconstructedParticles)")  # is this correct?
        df = df.Define("photons_sel_iso",f"FCCAnalyses::ZHfunctions::sel_iso({photon_iso_threshold})(photons_all, photons_iso)",) # and this??
   
        df = df.Define("photons_iso_p", "FCCAnalyses::ReconstructedParticle::get_p(photons_sel_iso)") 
        df = df.Define("photons_iso_n","FCCAnalyses::ReconstructedParticle::get_n(photons_sel_iso)")  #number of photons per event
        df = df.Define("photons_iso_cos_theta","cos(FCCAnalyses::ReconstructedParticle::get_theta(photons_sel_iso))")

        

        #########
        ### CUT 1: Photons must be isolated
        #########
    
        df = df.Filter("photons_sel_iso.size()>0 ")
        df = df.Define("recopart_no_gamma", "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles, photons_sel_iso)",)
        """
        #sort in p  and select highest energetic one
        df = df.Define("iso_highest_p","FCCAnalyses::ZHfunctions::sort_by_energy(photons_sel_iso)")

        #energy cut
        df = df.Define("photons_boosted", f"FCCAnalyses::ReconstructedParticle::sel_p({photon_energy_min},{photon_energy_max})(iso_highest_p)") # looked okay from photons all
   
        df = df.Define("photons_boosted_p", "FCCAnalyses::ReconstructedParticle::get_p(photons_boosted)") # is this correct?
        df = df.Define("photons_boosted_n","FCCAnalyses::ReconstructedParticle::get_n(photons_boosted)") 
        df = df.Define("photons_boosted_cos_theta","cos(FCCAnalyses::ReconstructedParticle::get_theta(photons_boosted))")
       
        #########
        ### CUT 2: Photons energy > 50
        #########
    
        df = df.Filter("photons_boosted.size()>0 ") 
        
        #########
        ### CUT 3: Cos Theta cut
        #########
        df = df.Filter(f"ROOT::VecOps::All(abs(photons_boosted_cos_theta) < {photon_cos_theta_max}) ") 
      
        ## create a new collection of reconstructed particles removing targeted photons
        
        df = df.Define("recopart_no_gamma_n","FCCAnalyses::ReconstructedParticle::get_n(recopart_no_gamma)") 

        df = df.Define("gamma_recoil", "FCCAnalyses::ReconstructedParticle::recoilBuilder(240)(photons_boosted)") 
        df = df.Define("gamma_recoil_m", "FCCAnalyses::ReconstructedParticle::get_mass(gamma_recoil)[0]") # recoil mass
        
       
        #########
        ### CUT 4: require at least 6 reconstructed particles (except gamma)
        #########
        df = df.Filter(f" recopart_no_gamma_n > {min_n_reco_no_gamma}") 
        
        """
        ## perform N=2 jet clustering
        global jetClusteringHelper
        global jetFlavourHelper

        ## define jet and run clustering parameters
        ## name of collections in EDM root files
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

        collections_nogamma = copy.deepcopy(collections)
        collections_nogamma["PFParticles"] = "recopart_no_gamma"

        jetClusteringHelper = ExclusiveJetClusteringHelper(collections_nogamma["PFParticles"], 2, "N2")
        df = jetClusteringHelper.define(df)

        ## define jet flavour tagging parameters

        jetFlavourHelper = JetFlavourHelper(
            collections_nogamma,
            jetClusteringHelper.jets,
            jetClusteringHelper.constituents,
        )

     

        ## define observables for tagger
        df = jetFlavourHelper.define(df)

        ## tagger inference
        df = jetFlavourHelper.inference(weaver_preproc, weaver_model, df)


        df = df.Define(
            "jets_p4",
            "JetConstituentsUtils::compute_tlv_jets({})".format(
                jetClusteringHelper.jets
            ),
        )
        df = df.Define(
            "jj_m",
            "JetConstituentsUtils::InvariantMass(jets_p4[0], jets_p4[1])",
        )

  
       
        return df


    # __________________________________________________________
    # Mandatory: output function, please make sure you return the branchlist as a python list
    def output():
        branchList = [
            "ReconstructedParticles",
            "photons_all",
            "electrons_all",
            "muons_all",
            "jj_m",
            "recopart_no_gamma",
        ]

        ## outputs jet scores and constituent breakdown
        branchList += jetFlavourHelper.outputBranches()

        print(branchList)

        return branchList