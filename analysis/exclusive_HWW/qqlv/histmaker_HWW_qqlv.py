import os, copy
import yaml
import argparse
from addons.TMVAHelper.TMVAHelper import TMVAHelperXGB

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Set up argument parser
parser = argparse.ArgumentParser(description="Run analysis for H->WW(lv)W(qq).")
parser.add_argument(
    "--energy", "-e",
    type=int,
    default=240,
    help="Choose from: 160, 240, 365. Default: 240"
)
args, _ = parser.parse_known_args()  # <-- Ignore unknown args


config = load_config(f"config/config_{args.energy}.yaml")
config_WW = load_config(f"config/config_WW_qqlv_{args.energy}.yaml")


print("Configuration:")
print(config)

print("Configuration for WW:")
print(config_WW)



ecm = config['ecm']

# list of processes (mandatory)
processList = {}
for key, val in config['processList'].items():
    if key == 'mgp8_ee_ha':
        frac = float(val['fraction']) 
        br_WW = 0.215  # branching ratio for H->WW
        xsec = {'160': 2.127e-5 * br_WW, '240': 8.773e-5 * br_WW, '365': 2.975e-5 * br_WW}.get(str(ecm), 0)
        if config_WW['do_inference']:
            frac = 0.7 # only use data that was not trained on 
        entry = {
            'fraction': frac,
            'crossSection': xsec
        }
        processList[f"{key}_ecm{ecm}_hww"] = entry
    else:
        entry = {
            'fraction': float(val['fraction']),
        }
        if 'crossSection' in val:
            entry['crossSection'] = float(val['crossSection'])  # optional
        if 'inputDir' in val:
            entry['inputDir'] = os.path.join(val['inputDir'], str(ecm))
        processList[f"{key}_ecm{ecm}"] = entry
    if key == 'wzp6_ee_aqqW':
        # correct xsec for WW* bkg
        xsec_aqqW = {'160': 2.328e-02, '240': 1.286e-01, '365': 1.131e-02}.get(str(ecm), 0)
        entry['crossSection'] = xsec_aqqW

print(processList)



# Production tag when running over EDM4Hep centrally produced events, this points to the yaml files for getting sample statistics (mandatory)
prodTag     = "FCCee/winter2023/IDEA/"

# Link to the dictonary that contains all the cross section informations etc... (mandatory)
procDict = "FCCee_procDict_winter2023_IDEA.json"

# additional/custom C++ functions, defined in header files (optional)
includePaths = ["../../functions.h"]

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
bins_count = (45, 0, 45)


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

    #isolation cut
    df = df.Define("photons_iso", f"FCCAnalyses::ZHfunctions::coneIsolation({photon_iso_cone_radius_min}, {photon_iso_cone_radius_max})(photons_all, ReconstructedParticles)")  # is this correct?
    df = df.Define("photons_sel_iso",f"FCCAnalyses::ZHfunctions::sel_iso({photon_iso_threshold})(photons_all, photons_iso)",) # and this??
   
    df = df.Define("photons_iso_p", "FCCAnalyses::ReconstructedParticle::get_p(photons_sel_iso)") 
    df = df.Define("photons_iso_n","FCCAnalyses::ReconstructedParticle::get_n(photons_sel_iso)")  #number of photons per event
    df = df.Define("photons_iso_cos_theta","cos(FCCAnalyses::ReconstructedParticle::get_theta(photons_sel_iso))")

     
    #########
    ### CUT 1: Photons must be isolated
    #########
    results.append(df.Histo1D(("photon_isolation", "", 50, 0, 10), "photons_iso"))
    
    df = df.Filter("photons_sel_iso.size()>0 ")  
    df = df.Define("cut1", "1")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut1"))

    #sort in p  and select highest energetic one
    df = df.Define("iso_highest_p","FCCAnalyses::ZHfunctions::sort_by_energy(photons_sel_iso)")
    df = df.Define("photons_boosted", f"FCCAnalyses::ReconstructedParticle::sel_p({photon_energy_min},{photon_energy_max})(iso_highest_p)") # looked okay from photons all

    df = df.Define("photons_boosted_p", "FCCAnalyses::ReconstructedParticle::get_p(photons_boosted)") # is this correct?
    df = df.Define("photons_boosted_n","(float)FCCAnalyses::ReconstructedParticle::get_n(photons_boosted)") 
    df = df.Define("photons_boosted_cos_theta","cos(FCCAnalyses::ReconstructedParticle::get_theta(photons_boosted))")

    # number reco particles
    df = df.Define("recopart_no_gamma", "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles, photons_boosted)",)
    df = df.Define("recopart_no_gamma_n","(float)FCCAnalyses::ReconstructedParticle::get_n(recopart_no_gamma)") 
 
    # recoil plot
    df = df.Define("gamma_recoil", f"FCCAnalyses::ReconstructedParticle::recoilBuilder({ecm})(photons_boosted)") 
    df = df.Define("gamma_recoil_m", "FCCAnalyses::ReconstructedParticle::get_mass(gamma_recoil)[0]") # recoil mass
    results.append(df.Histo1D(("gamma_recoil_m_cut1", "", 250, 0, 250), "gamma_recoil_m"))


    #########
    ### CUT 2: Photons energy 
    #########
    results.append(df.Histo1D(("photons_boosted_p", "", 80, int(photon_energy_min), int(photon_energy_max)), "photons_boosted_p"))
    
    df = df.Filter("photons_boosted.size()>0 ")  
    df = df.Define("cut2", "2")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut2"))

    #########
    ### CUT 3: Cos Theta cut
    #########
    results.append(df.Histo1D(("photons_cos_theta", "", 50, -1, 1), "photons_boosted_cos_theta"))

    df = df.Filter(f"ROOT::VecOps::All(abs(photons_boosted_cos_theta) < {photon_cos_theta_max}) ") 
    df = df.Define("cut3", "3")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut3"))

    
    #########
    ### CUT 4: require at least 6 reconstructed particles (except gamma)
    #########
    results.append(df.Histo1D(("recopart_no_gamma_n", "", 60, 0, 60), "recopart_no_gamma_n"))

    df = df.Filter(f" recopart_no_gamma_n > {min_n_reco_no_gamma}") 
    df = df.Define("cut4", "4")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut4"))

    #########
    ### CUT 5: gamma recoil cut
    #########
    results.append(df.Histo1D(("gamma_recoil_m", "", 250, 0, 250), "gamma_recoil_m"))

    df = df.Filter(f"{recoil_mass_min} < gamma_recoil_m && gamma_recoil_m < {recoil_mass_max}") 
    df = df.Define("cut5", "5")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut5"))

    results.append(df.Histo1D(("gamma_recoil_m_signal_cut", "", 40, 110, 150), "gamma_recoil_m"))
    #results.append(df.Histo1D(("gamma_recoil_m_signal_cut", "", 64, 116, 170), "gamma_recoil_m"))


    # NOTE: From here on, we add some extra cuts for H-> WW* -> qqlv

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
    df = df.Define("lepton_pT", "FCCAnalyses::ReconstructedParticle::get_pt(Vec_rp{lepton})[0]")
    results.append(df.Histo1D(("lepton_p", "", 100, 0, 200), "lepton_p"))
    results.append(df.Histo1D(("lepton_pT", "", 100, 0, 200), "lepton_pT"))

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
    # df = jetFlavourHelper.inference(weaver_preproc, weaver_model, df)

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
    df = df.Define("missP", f"FCCAnalyses::ZHfunctions::missingParticle({ecm}, ReconstructedParticles)")
    df = df.Define("miss_p", "FCCAnalyses::ReconstructedParticle::get_p(missP)[0]")
    df = df.Define("miss_pT", "FCCAnalyses::ReconstructedParticle::get_pt(missP)[0]")
    df = df.Define("miss_e", "FCCAnalyses::ReconstructedParticle::get_e(missP)[0]")
    results.append(df.Histo1D(("miss_p", "", 100, 0, 200), "miss_p"))
    results.append(df.Histo1D(("miss_pT", "", 100, 0, 200), "miss_pT"))
    results.append(df.Histo1D(("miss_e", "", 100, 0, 200), "miss_e"))

    # df = df.Filter(f"miss_p > {config_WW['cuts']['p_miss']}")  # missing momentum cut
    # df = df.Define("cut8", "8")
    # results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut8"))

    ########
    ### CUT 8: missing transverse momentum
    ########

    df = df.Filter(f"miss_pT > {config_WW['cuts']['pT_miss']}")  # missing transverse momentum cut
    df = df.Define("cut8", "8")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut8"))


    ##########
    ### CUT 9: recoil of photon plus qq jets must be in W mass range
    ##########

    df = df.Define("jet1", "jets_p4[0]")
    df = df.Define("jet2", "jets_p4[1]")
    df = df.Define("photon", "photons_boosted[0]")  # only one photon after cuts

    df = df.Define("recoil_W", f"FCCAnalyses::ZHfunctions::get_recoil_photon_and_jets({ecm}, jet1, jet2, photon)")
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



    # get the W bosons
    df = df.Define("Ws", "FCCAnalyses::ZHfunctions::build_WW(jet1, jet2, lepton, missP)")
    df = df.Define("W_qq", "Ws[0]") 
    df = df.Define("W_lv", "Ws[1]")
    df = df.Define("W_qq_theta", "FCCAnalyses::ReconstructedParticle::get_theta(Vec_rp{W_qq})[0]")
    df = df.Define("W_lv_theta", "FCCAnalyses::ReconstructedParticle::get_theta(Vec_rp{W_lv})[0]")
    df = df.Define("W_qq_costheta", "FCCAnalyses::ZHfunctions::get_costheta(W_qq_theta)")
    df = df.Define("W_lv_costheta", "FCCAnalyses::ZHfunctions::get_costheta(W_lv_theta)")
    results.append(df.Histo1D(("W_qq_theta", "", 50, 0, 3.14), "W_qq_theta"))
    results.append(df.Histo1D(("W_lv_theta", "", 50, 0, 3.14), "W_lv_theta"))
    results.append(df.Histo1D(("W_qq_costheta", "", 50, -1, 1), "W_qq_costheta"))
    results.append(df.Histo1D(("W_lv_costheta", "", 50, -1, 1), "W_lv_costheta"))

    df = df.Define("WW_unboosted", "FCCAnalyses::ZHfunctions::unboost_WW(Ws, photon, {})".format(ecm))
    df = df.Define("W_qq_unboosted", "WW_unboosted[0]") 
    df = df.Define("W_lv_unboosted", "WW_unboosted[1]")
    df = df.Define("W_qq_unboosted_theta", "FCCAnalyses::ReconstructedParticle::get_theta(Vec_rp{W_qq_unboosted})[0]")
    df = df.Define("W_lv_unboosted_theta", "FCCAnalyses::ReconstructedParticle::get_theta(Vec_rp{W_lv_unboosted})[0]")
    df = df.Define("W_qq_unboosted_costheta", "FCCAnalyses::ZHfunctions::get_costheta(W_qq_unboosted_theta)")
    df = df.Define("W_lv_unboosted_costheta", "FCCAnalyses::ZHfunctions::get_costheta(W_lv_unboosted_theta)")
    results.append(df.Histo1D(("W_qq_unboosted_theta", "", 50, 0, 3.14), "W_qq_unboosted_theta"))
    results.append(df.Histo1D(("W_lv_unboosted_theta", "", 50, 0, 3.14), "W_lv_unboosted_theta"))
    results.append(df.Histo1D(("W_qq_unboosted_costheta", "", 50, -1, 1), "W_qq_unboosted_costheta"))
    results.append(df.Histo1D(("W_lv_unboosted_costheta", "", 50, -1, 1), "W_lv_unboosted_costheta"))


    #### plot just more variable to have a look at! 
    results.append(df.Histo1D(("lepton_p_cut11", "", 100, 0, 200), "lepton_p"))
    results.append(df.Histo1D(("lepton_pT_cut11", "", 100, 0, 200), "lepton_pT"))
    results.append(df.Histo1D(("miss_p_cut11", "", 100, 0, 200), "miss_p"))
    results.append(df.Histo1D(("miss_pT_cut11", "", 100, 0, 200), "miss_pT"))
    
    df = df.Define("photons_sorted", "FCCAnalyses::ZHfunctions::sort_rp_by_energy(photons_boosted)")
    df = df.Define("photon_p", "FCCAnalyses::ReconstructedParticle::get_p(photons_sorted)[0]")  # only one photon after cuts
    df = df.Define("photon_pT", "FCCAnalyses::ReconstructedParticle::get_pt(photons_sorted)[0]")  # only one photon after cuts
    df = df.Define("photon_cos_theta","cos(FCCAnalyses::ReconstructedParticle::get_theta(photons_sorted))[0]")
    results.append(df.Histo1D(("photon_p_cut11", "", 80, int(photon_energy_min), int(photon_energy_max)), "photon_p"))
    results.append(df.Histo1D(("photon_pT_cut11", "", 80, 0, int(photon_energy_max)), "photon_pT"))
    results.append(df.Histo1D(("photon_cos_theta_cut11", "", 50, -1, 1), "photon_cos_theta"))

    df = df.Define("jets_sorted_rp", "FCCAnalyses::ZHfunctions::get_rp_sorted_jets(jet1, jet2)")
    df = df.Define("jets_sorted_theta", "FCCAnalyses::ReconstructedParticle::get_theta(jets_sorted_rp)")
    df = df.Define("jet1_costheta", "FCCAnalyses::ZHfunctions::get_costheta(jets_sorted_theta[0])")
    df = df.Define("jet2_costheta", "FCCAnalyses::ZHfunctions::get_costheta(jets_sorted_theta[1])")
    results.append(df.Histo1D(("jet1_costheta_cut11", "", 50, -1, 1), "jet1_costheta"))
    results.append(df.Histo1D(("jet2_costheta_cut11", "", 50, -1, 1), "jet2_costheta"))

    # pT of W_qq
    df = df.Define("W_qq_pT", "FCCAnalyses::ReconstructedParticle::get_pt(Ws)[0]")
    df = df.Define("W_lv_pT", "FCCAnalyses::ReconstructedParticle::get_pt(Ws)[1]")
    results.append(df.Histo1D(("W_qq_pT_cut11", "", 100, 0, 200), "W_qq_pT"))
    results.append(df.Histo1D(("W_lv_pT_cut11", "", 100, 0, 200), "W_lv_pT"))


    #########
    ### CUT 11: lepton pT cut
    #########
    results.append(df.Histo1D(("lepton_pT_cut11", "", 100, 0, 200), "lepton_pT"))
    df = df.Filter(f"lepton_pT > {config_WW['cuts']['pT_lepton']}")  # lepton pT cut
    df = df.Define("cut11", "11")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut11"))



    do_inference = config_WW.get('do_inference', False)
    if do_inference:
        # build variables for the MVA

        # cos theta j1
        df = df.Define("jets_sorted_phi", "FCCAnalyses::ReconstructedParticle::get_phi(jets_sorted_rp)")
        df = df.Define("jet1_cosphi", "FCCAnalyses::ZHfunctions::get_costheta(jets_sorted_phi[0])")
        df = df.Define("jet2_cosphi", "FCCAnalyses::ZHfunctions::get_costheta(jets_sorted_phi[1])")


        # inference with TMVAHelperXGB

        # inference with TMVAHelperXGB
        bdt_name = config_WW['BDT']
        tmva_helper = TMVAHelperXGB(f"outputs/{int(ecm)}/BDT/qqlv/{bdt_name}.root", "bdt_model") # read the XGBoost training
    
        df = tmva_helper.run_inference(df, col_name="mva_score") # by default, makes a new column mva_score
        df = df.Define("mva_score_signal", "mva_score[0]")
        df = df.Define("mva_score_bkg", "mva_score[1]")
        bins_mva = (100, 0, 1)
        results.append(df.Histo1D(("mva_score_signal", "", *bins_mva), "mva_score_signal"))
        results.append(df.Histo1D(("mva_score_bkg", "", *bins_mva), "mva_score_bkg"))


        ##########
        ### CUT 12: MVA score cut
        ##########
        mva_cut_value = config_WW['cuts']['mva_score_cut']
        df = df.Filter("mva_score_signal > {}".format(mva_cut_value))  # MVA score cut
        df = df.Define("cut12", "12")
        results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut12"))

        # # do a scan
        # mva_cut_values = config_WW['cuts']['mva_score_cut']
        # for i, mva_cut_value in enumerate(mva_cut_values):
        #     df_cut = df.Filter("mva_score_signal > {}".format(mva_cut_value))
        #     df_cut = df_cut.Define(f"cut{i+12}", f"{i+12}")
        #     results.append(df_cut.Histo1D(("cutFlow", "", *bins_count), f"cut{i+12}"))

        #########
        ### CUT 13: gamma recoil cut tight
        #########
        results.append(df.Histo1D(("gamma_recoil_m_tight_cut", "", 80, 110, 150), "gamma_recoil_m"))
        results.append(df.Histo1D(("gamma_recoil_m_some_cut", "", 60, 110, 150), "gamma_recoil_m"))
        results.append(df.Histo1D(("gamma_recoil_m_last_cut", "", 40, 110, 150), "gamma_recoil_m"))

        df = df.Filter(f"{signal_mass_min} < gamma_recoil_m && gamma_recoil_m < {signal_mass_max}")
        df = df.Define("cut13", "13")
        results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut13"))


    else:
        #########
        ### CUT 12: gamma recoil cut tight
        #########
        results.append(df.Histo1D(("gamma_recoil_m_tight_cut", "", 80, 110, 150), "gamma_recoil_m"))
        results.append(df.Histo1D(("gamma_recoil_m_last_cut", "", 40, 110, 150), "gamma_recoil_m"))

        df = df.Filter(f"{signal_mass_min} < gamma_recoil_m && gamma_recoil_m < {signal_mass_max}")
        df = df.Define("cut12", "12")
        results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut12"))


    return results, weightsum