import os, copy
import yaml
import argparse

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Set up argument parser
parser = argparse.ArgumentParser(description="Run a specific analysis: W(lv)W(qq) or W(qq)W(lv).")
parser.add_argument(
    "--config", "-c",
    type=str,
    default="lvqq",
    help="Choose from: qqlv, lvqq"
)
args, _ = parser.parse_known_args()  # <-- Ignore unknown args


config = load_config("config/config_240.yaml")
if args.config == "qqlv":
    config_WW = load_config("config/config_WW_qqlv_240.yaml")
elif args.config == "lvqq":
    config_WW = load_config("config/config_WW_lvqq_240.yaml")
else: 
    raise ValueError("Invalid config option. Choose from: qqlv, lvqq")

print("Configuration:")
print(config)

print("Configuration for WW:")
print(config_WW)



ecm = config['ecm']

# list of processes (mandatory)
processList = {}
for key, val in config['processList'].items():
    # change signal file 
    if key == f'p8_ee_Hgamma':
        entry = {'fraction': float(val['fraction'])}
        entry['inputDir'] = "/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/"
        entry['crossSection'] = float(val['crossSection']) * 0.2137 # H-> WW BR
        processList[f"mgp8_ee_ha_ecm{ecm}_hww"] = entry
    else:
        entry = {
            'fraction': float(val['fraction']),
        }
        if 'crossSection' in val:
            entry['crossSection'] = float(val['crossSection'])  # optional
            entry['inputDir'] = os.path.join(config['inputDirBase'], str(ecm))
        processList[f"{key}_ecm{ecm}"] = entry

print(processList)



# Production tag when running over EDM4Hep centrally produced events, this points to the yaml files for getting sample statistics (mandatory)
prodTag     = "FCCee/winter2023/IDEA/"

# Link to the dictonary that contains all the cross section informations etc... (mandatory)
procDict = "FCCee_procDict_winter2023_IDEA.json"

# additional/custom C++ functions, defined in header files (optional)
includePaths = ["../functions.h"]

#Optional: output directory, default is local running directory
outputDir   =  os.path.join(config['outputDir'], str(ecm),'histmaker/', config_WW['outputDir_sub'])
print(outputDir)

# optional: ncpus, default is 4, -1 uses all cores available
nCPUS       = -1

# scale the histograms with the cross-section and integrated luminosity
doScale = True
intLumi = config['intLumi']


# define some binning for various histograms
bins_a_p = (100, 0, 500) # 100 MeV bins
bins_a_n = (10, 0, 10) # 100 MeV bins
bins_count = (20, 0, 20)


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


# cuts
photon_iso_cone_radius_min, photon_iso_cone_radius_max = config['cuts']['photon_iso_cone_radius_range']
photon_iso_threshold = config['cuts']['photon_iso_threshold']
photon_energy_min, photon_energy_max = config['cuts']['photon_energy_range']
photon_cos_theta_max = config['cuts']['photon_cos_theta_max']
recoil_mass_min, recoil_mass_max = config['cuts']['recoil_mass_range']
signal_mass_min, signal_mass_max = config['cuts']['recoil_mass_signal_range']
min_n_reco_no_gamma = config['cuts']['min_n_reco_no_gamma']

# jet clustering and tagging

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




# build_graph function that contains the analysis logic, cuts and histograms (mandatory)
def build_graph(df, dataset):

  
    results = []
    df = df.Define("weight", "1.0")
    weightsum = df.Sum("weight")

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

    # plot iso values
    bins_iso = (500, 0, 3)
    results.append(df.Histo1D(("muons_iso", "", *bins_iso), "muons_iso"))
    results.append(df.Histo1D(("electrons_iso", "", *bins_iso), "electrons_iso"))


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

     
    #########
    ### CUT 1: Photons must be isolated
    #########
    
    df = df.Filter("photons_sel_iso.size()>0 ")  
    df = df.Define("cut1", "1")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut1"))
    
    results.append(df.Histo1D(("photons_p_cut_1", "",  130, 0, 130), "photons_iso_p"))
    results.append(df.Histo1D(("photons_n_cut_1", "", *bins_a_n), "photons_iso_n"))
    results.append(df.Histo1D(("photons_cos_theta_cut_1", "", 50, -1, 1), "photons_iso_cos_theta"))
 

    #sort in p  and select highest energetic one
    df = df.Define("iso_highest_p","FCCAnalyses::ZHfunctions::sort_by_energy(photons_sel_iso)")

    #energy cut
    df = df.Define("photons_boosted", f"FCCAnalyses::ReconstructedParticle::sel_p({photon_energy_min},{photon_energy_max})(iso_highest_p)") # looked okay from photons all
    #df = df.Define("photons_boosted", "FCCAnalyses::ReconstructedParticle::sel_p(60,100)(iso_highest_p)")

    df = df.Define("photons_boosted_p", "FCCAnalyses::ReconstructedParticle::get_p(photons_boosted)") # is this correct?
    df = df.Define("photons_boosted_n","FCCAnalyses::ReconstructedParticle::get_n(photons_boosted)") 
    df = df.Define("photons_boosted_cos_theta","cos(FCCAnalyses::ReconstructedParticle::get_theta(photons_boosted))")

    df = df.Define("recopart_no_gamma", "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles, photons_boosted)",)
    df = df.Define("recopart_no_gamma_n","FCCAnalyses::ReconstructedParticle::get_n(recopart_no_gamma)") 
   
 
    results.append(df.Histo1D(("recopart_no_gamma_n_cut_1", "", 60, 0, 60), "recopart_no_gamma_n"))


    #########
    ### CUT 2: Photons energy > 50
    #########
    
    df = df.Filter("photons_boosted.size()>0 ")  
    df = df.Define("cut2", "2")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut2"))
    
    results.append(df.Histo1D(("photons_p_cut_2", "",  130, 0, 130), "photons_boosted_p"))
    results.append(df.Histo1D(("photons_n_cut_2", "", *bins_a_n), "photons_boosted_n"))
    results.append(df.Histo1D(("photons_cos_theta_cut_2", "", 50, -1, 1), "photons_boosted_cos_theta"))


    #########
    ### CUT 3: Cos Theta cut
    #########
    df = df.Filter(f"ROOT::VecOps::All(abs(photons_boosted_cos_theta) < {photon_cos_theta_max}) ") 
   
    df = df.Define("cut3", "3")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut3"))
    
    results.append(df.Histo1D(("photons_p_cut_3", "", 130, 0, 130), "photons_boosted_p"))
    results.append(df.Histo1D(("photons_n_cut_3", "", *bins_a_n), "photons_boosted_n"))
    results.append(df.Histo1D(("photons_cos_theta_cut_3", "", 50, -1, 1), "photons_boosted_cos_theta"))


    # recoil plot
    df = df.Define("gamma_recoil", "FCCAnalyses::ReconstructedParticle::recoilBuilder(240)(photons_boosted)") 
    df = df.Define("gamma_recoil_m", "FCCAnalyses::ReconstructedParticle::get_mass(gamma_recoil)[0]") # recoil mass
    results.append(df.Histo1D(("gamma_recoil_m_cut_3", "", 170, 80, 250), "gamma_recoil_m"))

    
    #########
    ### CUT 4: require at least 6 reconstructed particles (except gamma)
    #########
    df = df.Filter(f" recopart_no_gamma_n > {min_n_reco_no_gamma}") 
    
    df = df.Define("cut4", "4")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut4"))
 
    results.append(df.Histo1D(("recopart_no_gamma_n_cut_4", "", 60, 0, 60), "recopart_no_gamma_n"))
    results.append(df.Histo1D(("gamma_recoil_m_cut_4", "", 170, 80, 250), "gamma_recoil_m"))



    #########
    ### CUT 5: gamma recoil cut
    #########
    df = df.Filter(f"{recoil_mass_min} < gamma_recoil_m && gamma_recoil_m < {recoil_mass_max}") 
    #df = df.Filter("115 < gamma_recoil_m && gamma_recoil_m < 170") 

    df = df.Define("cut5", "5")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut5"))

    results.append(df.Histo1D(("gamma_recoil_m_signal_cut", "", 40, 110, 150), "gamma_recoil_m"))
    #results.append(df.Histo1D(("gamma_recoil_m_signal_cut", "", 64, 116, 170), "gamma_recoil_m"))


    # NOTE: From here on, we add some extra cuts for H-> WW* -> lvqq 

    # Until I have the data, I will cut the MC data to only use events with MC H -> WW
    ########
    ### Cut 6: Only use events with MC H -> WW
    ########
    # df = df.Define("is_higgs_to_WW", "FCCAnalyses::ZHfunctions::get_higgs_to_WW(Particle, Particle1)")
    # df = df.Filter("is_higgs_to_WW == 1")
    # df = df.Define("cut6", "6")
    # results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut6"))

    ##########
    ### CUT 6: one isolated lepton
    ##########
    df = df.Define("num_isolated_leptons", "electrons_sel_iso.size() + muons_sel_iso.size()")
    results.append(df.Histo1D(("num_isolated_leptons", "", 10, 0, 10), "num_isolated_leptons"))

    df = df.Filter("num_isolated_leptons == 1")  # one isolated lepton
    df = df.Define("cut6", "6")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut6"))

    # have a look at the lepton
    df = df.Define("lepton", "muons_sel_iso.size() == 1 ? muons_sel_iso[0] : electrons_sel_iso[0]")
    df = df.Define("lepton_p", "FCCAnalyses::ReconstructedParticle::get_p(Vec_rp{lepton})[0]")
    results.append(df.Histo1D(("lepton_p", "", 100, 0, 200), "lepton_p"))

    
    # cluster 2 jets

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
    df = jetFlavourHelper.inference(weaver_preproc, weaver_model, df)

    df = df.Define("y23", "std::sqrt(JetClusteringUtils::get_exclusive_dmerge(_jet_N2, 2))")  # dmerge from 3 to 2
    df = df.Define("y34", "std::sqrt(JetClusteringUtils::get_exclusive_dmerge(_jet_N2, 3))")  # dmerge from 4 to 3
    results.append(df.Histo1D(("y23", "", 100, 0, 1), "y23"))

    i = 2
    for j in range(1, 3):
        df = df.Define(f"jet{j}_nconst_N{i}", f"jet_nconst_N{i}[{j-1}]")
        results.append(df.Histo1D((f"jet{j}_nconst_N{i}", "", 30, 0, 30), f"jet{j}_nconst_N{i}"))


    df = df.Define("jets_p4","JetConstituentsUtils::compute_tlv_jets({})".format(jetClusteringHelper.jets))
    df = df.Define("m_jj","JetConstituentsUtils::InvariantMass(jets_p4[0], jets_p4[1])")
    results.append(df.Histo1D(("m_jj", "", 100, 0, 200), "m_jj"))
    
    ###########
    ### CUT 7: jet mass cut (W* mass)
    ###########

    m_jj_min, m_jj_max = config_WW['cuts']['m_jj_range']

    df = df.Filter(f"{m_jj_min} < m_jj && m_jj < {m_jj_max}")  # W* mass cut
    df = df.Define("cut7", "7")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut7"))

    ##########
    ### CUT 8: missing momentum 
    ##########
    df = df.Define("missP", "FCCAnalyses::ZHfunctions::missingParticle(240.0, ReconstructedParticles)")
    df = df.Define("miss_p", "FCCAnalyses::ReconstructedParticle::get_p(missP)[0]")
    df = df.Define("miss_pT", "FCCAnalyses::ReconstructedParticle::get_pt(missP)[0]")
    df = df.Define("miss_e", "FCCAnalyses::ReconstructedParticle::get_e(missP)[0]")
    results.append(df.Histo1D(("miss_p", "", 100, 0, 200), "miss_p"))
    results.append(df.Histo1D(("miss_pT", "", 100, 0, 200), "miss_pT"))
    results.append(df.Histo1D(("miss_e", "", 100, 0, 200), "miss_e"))

    df = df.Filter(f"miss_p > {config_WW['cuts']['p_miss']}")  # missing momentum cut
    df = df.Define("cut8", "8")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut8"))

    ########
    ### CUT 8: missing transverse momentum
    ########
    # results.append(df.Histo1D(("miss_pT_cut_8", "", 100, 0, 200), "miss_pT"))
    # df = df.Filter("miss_pT > 10")  # missing transverse momentum cut
    # df = df.Define("cut8", "8")
    # results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut8"))


    ##########
    ### CUT 9: recoil of photon plus qq jets must be in W mass range
    ##########

    df = df.Define("jet1", "jets_p4[0]")
    df = df.Define("jet2", "jets_p4[1]")
    df = df.Define("photon", "photons_boosted[0]")  # only one photon after cuts

    df = df.Define("recoil_W", "FCCAnalyses::ZHfunctions::get_recoil_photon_and_jets(240.0, jet1, jet2, photon)")
    df = df.Define("recoil_W_m", "FCCAnalyses::ReconstructedParticle::get_mass(recoil_W)[0]")  # recoil mass of photon plus qq jets
    results.append(df.Histo1D(("recoil_W_m", "", 100, 0, 200), "recoil_W_m"))

    recoil_gammaqq_min, recoil_gammaqq_max = config_WW['cuts']['recoil_gammaqq_range']

    df = df.Filter(f"{recoil_gammaqq_min} < recoil_W_m && recoil_W_m < {recoil_gammaqq_max}")  # W mass range cut
    df = df.Define("cut9", "9")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut9"))

    ########
    ### CUT 10: number of constituents in jets
    ########

    results.append(df.Histo1D((f"jet1_nconst_N2_cut10", "", 30, 0, 30), f"jet1_nconst_N2"))
    results.append(df.Histo1D((f"jet2_nconst_N2_cut10", "", 30, 0, 30), f"jet2_nconst_N2"))

    df = df.Filter(f"jet1_nconst_N2 > {config_WW['cuts']['n_const_per_jet']} && jet2_nconst_N2 > {config_WW['cuts']['n_const_per_jet']}")  # at least 4 constituent in each jet
    df = df.Define("cut10", "10")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut10"))


    
    #########
    ### CUT 11: gamma recoil cut tight
    #########
    #df = df.Filter("123.5 < gamma_recoil_m && gamma_recoil_m < 126.5") 
    results.append(df.Histo1D(("gamma_recoil_m_tight_cut", "", 80, 110, 150), "gamma_recoil_m"))

    df = df.Filter(f"{signal_mass_min} < gamma_recoil_m && gamma_recoil_m < {signal_mass_max}") 
    df = df.Define("cut11", "11")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut11"))


    return results, weightsum