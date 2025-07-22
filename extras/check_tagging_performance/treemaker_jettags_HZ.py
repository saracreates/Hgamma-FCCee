import os, copy

datapath = "/eos/user/l/lherrman/FCC/data/HiggsGamma/"

# list of processes
processList = {
    'wzp6_ee_nunuH_Hbb_ecm240':    {'fraction':0.05,}
}

# Production tag when running over EDM4Hep centrally produced events, this points to the yaml files for getting sample statistics (mandatory)
prodTag     = "FCCee/winter2023/IDEA/"

#Optional: output directory, default is local running directory
outputDir   = "./outputs/jettags/Hgamma/" 

# Define the input dir (optional)
#inputDir    = "./localSamples/"

# additional/costom C++ functions, defined in header files (optional)
includePaths = ["helper_functions.h"]

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

bins_count = (10, 0, 10)
# Mandatory: RDFanalysis class where the use defines the operations on the TTree
class RDFanalysis:

    # __________________________________________________________
    # Mandatory: analysers funtion to define the analysers to process, please make sure you return the last dataframe, in this example it is df2
    def analysers(df):
       
        # __________________________________________________________
        # Mandatory: analysers funtion to define the analysers to process, please make sure you return the last dataframe, in this example it is df2
        results = []
        # define some aliases to be used later on
        df = df.Alias("Particle0", "Particle#0.index") # index of the mother particles
        df = df.Alias("Particle1", "Particle#1.index") # index of the daughter particles
        df = df.Alias("MCRecoAssociations0", "MCRecoAssociations#0.index")
        df = df.Alias("MCRecoAssociations1", "MCRecoAssociations#1.index")
        df = df.Alias("Photon0", "Photon#0.index")
        # get all the leptons from the collection
       
        
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

        jetClusteringHelper = ExclusiveJetClusteringHelper("ReconstructedParticles", 2, "N2")
        df = jetClusteringHelper.define(df)

        ## define jet flavour tagging parameters

        jetFlavourHelper = JetFlavourHelper(
            collections,
            jetClusteringHelper.jets,
            jetClusteringHelper.constituents,
        )

     

        ## define observables for tagger
        df = jetFlavourHelper.define(df)

        ## tagger inference
        df = jetFlavourHelper.inference(weaver_preproc, weaver_model, df)


        # check the true jet flavour
        df = df.Define("MC_pdg_flavour", "FCCAnalyses::ZHfunctions::get_higgs_daughters_MC_pdg(Particle, Particle1)")
        # df = df.Define("dummy", "FCCAnalyses::fullsimtagger::print_scores(recojet_isB)")


        # Rename columns appropriately
        flavors = {
            "U": 1, "D": 2, "S": 3, "B": 5,
            "C": 4, "G": 21, "TAU": 15,
        }

        # Rename predictions
        for flav in flavors:
            old_col = f"recojet_is{flav}"
            new_col = f"score_recojet_is{flav}"
            df = df.Define(new_col, old_col)

        # Define truth labels
        for flav, pdg in flavors.items():
            df = df.Redefine(f"recojet_is{flav}", f"FCCAnalyses::ZHfunctions::is_of_flavor(MC_pdg_flavour, {pdg})")
       
        return df

        #how do I get flavor

    # __________________________________________________________
    # Mandatory: output function, please make sure you return the branchlist as a python list
    def output():
        branchList = []

        ##  outputs jet tag properties

        flavors = ["U", "D", "S", "B", "C", "G", "TAU"]
        for flav in flavors:
            branchList += [
                f"recojet_is{flav}",
                f"score_recojet_is{flav}",
            ]

        ## outputs jet scores and constituent breakdown
        # branchList += jetFlavourHelper.outputBranches()

        print("RDataFrame is processing following columns: ",branchList)

        return branchList