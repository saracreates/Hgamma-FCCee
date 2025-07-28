
from addons.TMVAHelper.TMVAHelper import TMVAHelperXGB
import os, copy
import yaml
# import argparse

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config("config/config_240.yaml")
config_WW = load_config("config/config_WW_lvqq_240.yaml")

# Output directory
outputDir   = "outputs/240/preselection/lvqq/trainingdata"
inputDir = "/afs/cern.ch/work/s/saaumill/public/analyses/symlink_gammalvqq"
processList = {
    # cross sections given on the webpage: https://fcc-physics-events.web.cern.ch/fcc-ee/delphes/winter2023/idea/ 
    'p8_ee_WW_ecm240': {'fraction': 0.1, 'crossSection': 16.4385, 'inputDir': inputDir}, # 16 pb
    'mgp8_ee_ha_ecm240_hww':   {'fraction': 1, 'crossSection': 8.20481e-05* 0.2137, 'inputDir':inputDir}, 
}


# Production tag when running over EDM4Hep centrally produced events, this points to the yaml files for getting sample statistics (mandatory)
prodTag     = "FCCee/winter2023/IDEA/"

# Link to the dictonary that contains all the cross section informations etc... (mandatory)
procDict = "FCCee_procDict_winter2023_IDEA.json"

# Additional/custom C++ functions, defined in header files
includePaths = ["./../../functions.h"]


## latest particle transformer model, trained on 9M jets in winter2023 samples
model_name = "fccee_flavtagging_edm4hep_wc_v1"

## model files needed for unit testing in CI
url_model_dir = "https://fccsw.web.cern.ch/fccsw/testsamples/jet_flavour_tagging/winter2023/wc_pt_13_01_2022/"
url_preproc = "{}/{}.json".format(url_model_dir, model_name)
url_model = "{}/{}.onnx".format(url_model_dir, model_name)

## model files locally stored on /eos
model_dir = (
    "/eos/experiment/fcc/ee/jet_flavour_tagging/winter2023/wc_pt_13_01_2022/"
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

# Multithreading: -1 means using all cores
nCPUS       = -1

# cuts
photon_iso_cone_radius_min, photon_iso_cone_radius_max = config['cuts']['photon_iso_cone_radius_range']
photon_iso_threshold = config['cuts']['photon_iso_threshold']
photon_energy_min, photon_energy_max = config['cuts']['photon_energy_range']
photon_cos_theta_max = config['cuts']['photon_cos_theta_max']

ecm = config['ecm']

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


class RDFanalysis():

    # encapsulate analysis logic, definitions and filters in the dataframe
    def analysers(df):

        df = df.Alias("Particle0", "Particle#0.index") # index of the mother particles
        df = df.Alias("Particle1", "Particle#1.index") # index of the daughter particles
    
        df = df.Alias("Photon0", "Photon#0.index")
        df = df.Define("photons_all", "FCCAnalyses::ReconstructedParticle::get(Photon0, ReconstructedParticles)")

        df = df.Alias("Electron0", "Electron#0.index")
        df = df.Define("electrons_all", "FCCAnalyses::ReconstructedParticle::get(Electron0, ReconstructedParticles)")

        df = df.Alias("Muon0", "Muon#0.index")
        df = df.Define("muons_all", "FCCAnalyses::ReconstructedParticle::get(Muon0, ReconstructedParticles)")

        df = df.Define("photons_p", "FCCAnalyses::ReconstructedParticle::get_p(photons_all)") 
        df = df.Define("photons_n","FCCAnalyses::ReconstructedParticle::get_n(photons_all)")  #number of photons per event
        df = df.Define("photons_cos_theta","cos(FCCAnalyses::ReconstructedParticle::get_theta(photons_all))")
        
        df = df.Define("electrons_p", "FCCAnalyses::ReconstructedParticle::get_p(electrons_all)") 
        df = df.Define("electrons_n","FCCAnalyses::ReconstructedParticle::get_n(electrons_all)")  #number of photons per event
        df = df.Define("electrons_cos_theta","cos(FCCAnalyses::ReconstructedParticle::get_theta(electrons_all))")

        # compute the muon isolation and store muons with an isolation cut of 0df = df.25 in a separate column muons_sel_iso
        df = df.Define(
            "muons_iso",
            "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.5)(muons_all, ReconstructedParticles)",
        )
        df = df.Define(
            "muons_sel_iso",
            "FCCAnalyses::ZHfunctions::sel_iso(0.25)(muons_all, muons_iso)",
        )
        df = df.Define(
            "muons_sel_q",
            "FCCAnalyses::ReconstructedParticle::get_charge(muons_sel_iso)",
        )
        df = df.Define(
            "electrons_iso",
            "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.5)(electrons_all, ReconstructedParticles)",
        )
        df = df.Define(
            "electrons_sel_iso",
            "FCCAnalyses::ZHfunctions::sel_iso(0.25)(electrons_all, electrons_iso)",
        )
        df = df.Define(
            "electrons_sel_q",
            "FCCAnalyses::ReconstructedParticle::get_charge(electrons_sel_iso)",
        )


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


        #sort in p  and select highest energetic one
        df = df.Define("iso_highest_p","FCCAnalyses::ZHfunctions::sort_by_energy(photons_sel_iso)")

        #energy cut
        df = df.Define("photons_boosted", f"FCCAnalyses::ReconstructedParticle::sel_p({photon_energy_min},{photon_energy_max})(iso_highest_p)") # looked okay from photons all
        #df = df.Define("photons_boosted", "FCCAnalyses::ReconstructedParticle::sel_p(60,100)(iso_highest_p)")
        df = df.Define("photons_boosted_n","(float)FCCAnalyses::ReconstructedParticle::get_n(photons_boosted)") 

        df = df.Define("recopart_no_gamma", "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles, photons_boosted)",)
        df = df.Define("recopart_no_gamma_n","FCCAnalyses::ReconstructedParticle::get_n(recopart_no_gamma)")

        #########
        ### CUT 2: Photons energy > 50
        #########
        
        df = df.Filter("photons_boosted.size()>0 ")  


        #########
        ### CUT 3: Cos Theta cut
        #########
        df = df.Filter(f"ROOT::VecOps::All(abs(photons_boosted_cos_theta) < {photon_cos_theta_max}) ") 

        ########
        ### CUT 4: One isolated lepton
        ########
        df = df.Define("num_isolated_leptons", "electrons_sel_iso.size() + muons_sel_iso.size()")
        df = df.Filter("num_isolated_leptons == 1")  # one isolated lepton




        # create a collection of reco particles without the photon and without isolated leptons
        df = df.Define("RecoParticles_no_gamma_no_mu", "FCCAnalyses::ReconstructedParticle::remove(recopart_no_gamma, muons_sel_iso)")
        df = df.Define("RecoParticles_no_gamma_no_leptons", "FCCAnalyses::ReconstructedParticle::remove(RecoParticles_no_gamma_no_mu, electrons_sel_iso)")


        global jetClusteringHelper
        global jetFlavourHelper

        collections_no_gamma_no_leptons = copy.deepcopy(collections)
        collections_no_gamma_no_leptons["PFParticles"] = "RecoParticles_no_gamma_no_leptons"

        jetClusteringHelper = ExclusiveJetClusteringHelper(collections_no_gamma_no_leptons["PFParticles"], 2, "N2")
        df = jetClusteringHelper.define(df)

        jetFlavourHelper = JetFlavourHelper(
            collections_no_gamma_no_leptons,
            jetClusteringHelper.jets,
            jetClusteringHelper.constituents,
        )
        ## define observables for tagger
        df = jetFlavourHelper.define(df)

        ## tagger inference
        # df = jetFlavourHelper.inference(weaver_preproc, weaver_model, df)

        df = df.Define("y23", "std::sqrt(JetClusteringUtils::get_exclusive_dmerge(_jet_N2, 2))")  # dmerge from 3 to 2
        df = df.Define("y34", "std::sqrt(JetClusteringUtils::get_exclusive_dmerge(_jet_N2, 3))")  # dmerge from 4 to 3

        i = 2
        for j in range(1, 3):
            df = df.Define(f"jet{j}_nconst_N{i}", f"(float)jet_nconst_N{i}[{j-1}]")


        df = df.Define("jets_p4","JetConstituentsUtils::compute_tlv_jets({})".format(jetClusteringHelper.jets))
        df = df.Define("m_jj","JetConstituentsUtils::InvariantMass(jets_p4[0], jets_p4[1])")


        ###########
        ### CUT 5: jet mass cut (W* mass)
        ###########

        m_jj_min, m_jj_max = config_WW['cuts']['m_jj_range']

        df = df.Filter(f"{m_jj_min} < m_jj && m_jj < {m_jj_max}")  # W* mass cut


        #########
        ### Building variables
        #########

        df = df.Define("gamma_recoil", "FCCAnalyses::ReconstructedParticle::recoilBuilder(240)(photons_boosted)") 
        df = df.Define("gamma_recoil_m", "FCCAnalyses::ReconstructedParticle::get_mass(gamma_recoil)[0]") # recoil mass

        # df = df.Define("leptons", "FCCAnalyses::ZHfunctions::get_leptons(electrons_sel_iso, muons_sel_iso)")
        # df = df.Define("leptons_sorted", "FCCAnalyses::ZHfunctions::sort_by_energy(leptons)")
        # df = df.Define("lepton_p", "FCCAnalyses::ReconstructedParticle::get_p(leptons_sorted)[0]")
        # df = df.Define("lepton_pT", "FCCAnalyses::ReconstructedParticle::get_pt(leptons_sorted)[0]")
        df = df.Define("lepton", "muons_sel_iso.size() == 1 ? muons_sel_iso[0] : electrons_sel_iso[0]")
        df = df.Define("lepton_p", "FCCAnalyses::ReconstructedParticle::get_p(Vec_rp{lepton})[0]")
        df = df.Define("lepton_pT", "FCCAnalyses::ReconstructedParticle::get_pt(Vec_rp{lepton})[0]")

        df = df.Define("missP", "FCCAnalyses::ZHfunctions::missingParticle(240.0, ReconstructedParticles)")
        df = df.Define("miss_p", "FCCAnalyses::ReconstructedParticle::get_p(missP)[0]")
        df = df.Define("miss_pT", "FCCAnalyses::ReconstructedParticle::get_pt(missP)[0]")
        df = df.Define("miss_e", "FCCAnalyses::ReconstructedParticle::get_e(missP)[0]")

        df = df.Define("jet1", "jets_p4[0]")
        df = df.Define("jet2", "jets_p4[1]")
        df = df.Define("photons_sorted", "FCCAnalyses::ZHfunctions::sort_rp_by_energy(photons_boosted)")
        df = df.Define("photon_p", "FCCAnalyses::ReconstructedParticle::get_p(photons_sorted)[0]")  # only one photon after cuts
        df = df.Define("photon_pT", "FCCAnalyses::ReconstructedParticle::get_pt(photons_sorted)[0]")  # only one photon after cuts
        df = df.Define("photon_cos_theta","cos(FCCAnalyses::ReconstructedParticle::get_theta(photons_sorted))[0]")
        df = df.Define("photon", "photons_sorted[0]")  # only one photon after cuts

        df = df.Define("recoil_W", "FCCAnalyses::ZHfunctions::get_recoil_photon_and_jets(240.0, jet1, jet2, photon)")
        df = df.Define("recoil_W_m", "FCCAnalyses::ReconstructedParticle::get_mass(recoil_W)[0]")  # recoil mass of photon plus qq jets

        df = df.Define("Ws", "FCCAnalyses::ZHfunctions::build_WW(jet1, jet2, lepton, missP)")
        df = df.Define("W_qq", "Ws[0]") 
        df = df.Define("W_lv", "Ws[1]")

        df = df.Define("WW_unboosted", "FCCAnalyses::ZHfunctions::unboost_WW(Ws, photon, {})".format(ecm))
        df = df.Define("W_qq_unboosted", "WW_unboosted[0]") 
        df = df.Define("W_lv_unboosted", "WW_unboosted[1]")
        df = df.Define("W_qq_unboosted_theta", "FCCAnalyses::ReconstructedParticle::get_theta(Vec_rp{W_qq_unboosted})[0]")
        df = df.Define("W_lv_unboosted_theta", "FCCAnalyses::ReconstructedParticle::get_theta(Vec_rp{W_lv_unboosted})[0]")
        df = df.Define("W_qq_unboosted_costheta", "FCCAnalyses::ZHfunctions::get_costheta(W_qq_unboosted_theta)")
        df = df.Define("W_lv_unboosted_costheta", "FCCAnalyses::ZHfunctions::get_costheta(W_lv_unboosted_theta)")

        # look at more variables, see page 7: https://arxiv.org/pdf/2107.02686 

        # cos theta j1
        df = df.Define("jets_sorted_rp", "FCCAnalyses::ZHfunctions::get_rp_sorted_jets(jet1, jet2)")
        df = df.Define("jets_sorted_theta", "FCCAnalyses::ReconstructedParticle::get_theta(jets_sorted_rp)")
        df = df.Define("jets_sorted_phi", "FCCAnalyses::ReconstructedParticle::get_phi(jets_sorted_rp)")
        df = df.Define("jet1_costheta", "FCCAnalyses::ZHfunctions::get_costheta(jets_sorted_theta[0])")
        df = df.Define("jet2_costheta", "FCCAnalyses::ZHfunctions::get_costheta(jets_sorted_theta[1])")
        df = df.Define("jet1_cosphi", "FCCAnalyses::ZHfunctions::get_costheta(jets_sorted_phi[0])")
        df = df.Define("jet2_cosphi", "FCCAnalyses::ZHfunctions::get_costheta(jets_sorted_phi[1])")

        # pT of W_qq
        df = df.Define("W_qq_pT", "FCCAnalyses::ReconstructedParticle::get_pt(Ws)[0]")
        df = df.Define("W_lv_pT", "FCCAnalyses::ReconstructedParticle::get_pt(Ws)[1]")

        return df

    # define output branches to be saved
    def output():
        branchList = ["photon_p", "photons_boosted_n", "photon_cos_theta", "recopart_no_gamma_n",
                      "gamma_recoil_m", "num_isolated_leptons", "lepton_p", "lepton_pT", "y23", "y34",
                      "jet1_nconst_N2", "jet2_nconst_N2", "m_jj", "miss_p", "miss_pT", 
                      "recoil_W_m", "W_qq_unboosted_costheta", "W_lv_unboosted_costheta",
                      "jet1_costheta", "jet2_costheta", "jet1_cosphi", "jet2_cosphi",
                      "W_qq_pT", "W_lv_pT",] # isolation value photons?
        return branchList