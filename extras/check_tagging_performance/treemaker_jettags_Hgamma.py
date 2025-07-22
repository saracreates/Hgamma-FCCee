import os, copy

datapath = "/eos/user/l/lherrman/FCC/data/HiggsGamma/"

# list of processes
processList = {
    # 'p8_ee_Hgamma_ecm240':    {'fraction':1, 'crossSection': 8.20481e-05, 'inputDir': datapath + "240"}, # 82 ab 
    # 'ha160':                  {'fraction':1, 'crossSection': 17.3e-06, 'inputDir': datapath + "160"}, # 17.3 ab
    # 'p8_ee_Hgamma_ecm365':    {'fraction':1, 'crossSection': 3.5e-05, 'inputDir': datapath + "365"}, # 35 ab
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
       
        df = df.Define(
            "photons_all",
            "FCCAnalyses::ReconstructedParticle::get(Photon0, ReconstructedParticles)",
        )
        
        df = df.Define("photons_p", "FCCAnalyses::ReconstructedParticle::get_p(photons_all)") 
        df = df.Define("photons_n","FCCAnalyses::ReconstructedParticle::get_n(photons_all)")  #number of photons per event
        df = df.Define("photons_cos_theta","cos(FCCAnalyses::ReconstructedParticle::get_theta(photons_all))")

    
    
        #isolation cut
        df = df.Define("photons_iso", "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.5)(photons_all, ReconstructedParticles)")  # is this correct?
        df = df.Define("photons_sel_iso","FCCAnalyses::ZHfunctions::sel_iso(0.2)(photons_all, photons_iso)",) # and this??
   
        df = df.Define("photons_iso_p", "FCCAnalyses::ReconstructedParticle::get_p(photons_sel_iso)") 
        df = df.Define("photons_iso_n","FCCAnalyses::ReconstructedParticle::get_n(photons_sel_iso)")  #number of photons per event
        df = df.Define("photons_iso_cos_theta","cos(FCCAnalyses::ReconstructedParticle::get_theta(photons_sel_iso))")

        

        #########
        ### CUT 1: Photons must be isolated
        #########
    
        df = df.Filter("photons_sel_iso.size()>0 ")
      
        #sort in p  and select highest energetic one
        df = df.Define("iso_highest_p","FCCAnalyses::ZHfunctions::sort_by_energy(photons_sel_iso)")

        #energy cut
        df = df.Define("photons_boosted", "FCCAnalyses::ReconstructedParticle::sel_p(60,100)(iso_highest_p)") # looked okay from photons all
   
        df = df.Define("photons_boosted_p", "FCCAnalyses::ReconstructedParticle::get_p(photons_boosted)") # is this correct?
        df = df.Define("photons_boosted_n","FCCAnalyses::ReconstructedParticle::get_n(photons_boosted)") 
        df = df.Define("photons_boosted_cos_theta","cos(FCCAnalyses::ReconstructedParticle::get_theta(photons_boosted))")
       
        #########
        ### CUT 2: Photons energy > 50
        #########
    
        df = df.Filter("photons_boosted.size()>0 ") 
      
        ## create a new collection of reconstructed particles removing targeted photons
        df = df.Define("recopart_no_gamma", "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles, photons_boosted)",)
        df = df.Define("recopart_no_gamma_n","FCCAnalyses::ReconstructedParticle::get_n(recopart_no_gamma)") 

        df = df.Define("gamma_recoil", "FCCAnalyses::ReconstructedParticle::recoilBuilder(240)(photons_boosted)") 
        df = df.Define("gamma_recoil_m", "FCCAnalyses::ReconstructedParticle::get_mass(gamma_recoil)[0]") # recoil mass

        ########
        ### Cut 3: Only use events with MC H -> jets 
        ########
        df = df.Define("is_higgs_to_jets", "FCCAnalyses::ZHfunctions::check_higgs_to_jets(Particle, Particle1)")
        df = df.Filter("is_higgs_to_jets == 1")
      
        
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