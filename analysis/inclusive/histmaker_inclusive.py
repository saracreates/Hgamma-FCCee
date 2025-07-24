import os, copy
import yaml
import argparse

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)



config = load_config("config/config_240.yaml")

print("Configuration:")
print(config)



ecm = config['ecm']
scaling_factor = config['scaling_factor']

# list of processes (mandatory)
processList = {}
processList = {}
for key, val in config['processList'].items():
    entry = {
        'fraction': float(val['fraction']),
    }
    if 'crossSection' in val:
        entry['crossSection'] = float(val['crossSection'])  # optional
        entry['inputDir'] = os.path.join(config['inputDirBase'], str(ecm))
    processList[f"{key}_ecm{ecm}"] = entry

"""
for key, val in config['processList'].items():
    processList[key + f"_ecm{ecm}"] = {
        'fraction': float(val['fraction']),
        'crossSection': float(val['crossSection']),
        'inputDir': os.path.join(config['inputDirBase'], str(ecm))
    }
"""
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
outputDir   =  os.path.join(config['outputDir'], str(ecm),'histmaker/recoil')
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


##?| name of collections in EDM root files
collections = {
    "GenParticles": "Particle",
    "MCRecoMap": "MCRecoAssociations",
    "PFParticles": "ReconstructedParticles",
    "PFTracks": "EFlowTrack",
    "PFPhotons": "EFlowPhoton",
    "PFNeutralHadrons": "EFlowNeutralHadron",
    # "TrackState": "EFlowTrack_1",
    "TrackState": "_EFlowTrack_trackStates",
    "TrackerHits": "TrackerHits",
    "CalorimeterHits": "CalorimeterHits",
    # "dNdx": "EFlowTrack_2",
    "dNdx": "_EFlowTrack_dxQuantities",
    "PathLength": "EFlowTrack_L",
    "Bz": "magFieldBz",
    "Electrons": "Electron",
    "Muons": "Muon",
}


#cuts
photon_iso_cone_radius_min, photon_iso_cone_radius_max = config['cuts']['photon_iso_cone_radius_range']
photon_iso_threshold = config['cuts']['photon_iso_threshold']
photon_energy_min, photon_energy_max = config['cuts']['photon_energy_range']
photon_cos_theta_max = config['cuts']['photon_cos_theta_max']
recoil_mass_min, recoil_mass_max = config['cuts']['recoil_mass_range']
signal_mass_min, signal_mass_max = config['cuts']['recoil_mass_signal_range']
min_n_reco_no_gamma = config['cuts']['min_n_reco_no_gamma']




# build_graph function that contains the analysis logic, cuts and histograms (mandatory)
def build_graph(df, dataset):

  
    results = []
    df = df.Define("weight", "1.0")
    weightsum = df.Sum("weight")

    if config['apply_scaling']:
        print("rescale photon momentum resolution")
        ## define MC/Reco links, needed for smearing
        df = (df.Alias("mc_reco_0", "{}#0.index".format(collections["MCRecoMap"])).Alias(
                "mc_reco_1", "{}#1.index".format(collections["MCRecoMap"]))
            # matching between the RecoParticles and the MCParticles:
            .Define(
                "reco_mc_index",
                "ReconstructedParticle2MC::getRP2MC_index(mc_reco_0,mc_reco_1,{})".format(collections["PFParticles"]),
            )
            )
    
        #apply smearing
        df = df.Define(
            "smeared_PFParticles",
            f'FCCAnalyses::SmearObjects::SmearedReconstructedParticle({scaling_factor}, 22, 1, false)('
            + collections["PFParticles"] + ', reco_mc_index, ' + collections["GenParticles"] + ')')
   

        df = df.Alias("Photon0", "Photon#0.index")
        df = df.Define(
            "photons_all",
            "FCCAnalyses::ReconstructedParticle::get(Photon0, smeared_PFParticles)",
        )
    else:
        df = df.Alias("Photon0", "Photon#0.index")
        df = df.Define(
            "photons_all",
            "FCCAnalyses::ReconstructedParticle::get(Photon0, ReconstructedParticles)",
        )

    df = df.Alias("Electron0", "Electron#0.index")
    df = df.Define(
            "electrons_all",
            "FCCAnalyses::ReconstructedParticle::get(Electron0, ReconstructedParticles)",
        )

    


    df = df.Define("photons_p", "FCCAnalyses::ReconstructedParticle::get_p(photons_all)") 
    df = df.Define("photons_n","FCCAnalyses::ReconstructedParticle::get_n(photons_all)")  #number of photons per event
    df = df.Define("photons_cos_theta","cos(FCCAnalyses::ReconstructedParticle::get_theta(photons_all))")
    

    df = df.Define("electrons_p", "FCCAnalyses::ReconstructedParticle::get_p(electrons_all)") 
    df = df.Define("electrons_n","FCCAnalyses::ReconstructedParticle::get_n(electrons_all)")  #number of photons per event
    df = df.Define("electrons_cos_theta","cos(FCCAnalyses::ReconstructedParticle::get_theta(electrons_all))")



    # order the cos theta values, and return arrays, when filter require length 2 for cut!

    # get cos theta from electrons
    df = df.Define("electrons_ordered_cos_theta","FCCAnalyses::ZHfunctions::ee_costheta_max(electrons_cos_theta)")
   
    #print and check
    #df = df.Define("photons_print", "FCCAnalyses::ZHfunctions::print_momentum(electrons_all)")
    #results.append(df.Histo1D(("photons_print", "", 100, 0, 100), "photons_print"))


    #########
    ### CUT 0: all events
    #########
    df = df.Define("cut0", "0")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut0"))


    #Baseline selection
    results.append(df.Histo1D(("photons_p_cut_0", "", 130, 0, 130), "photons_p"))
    results.append(df.Histo1D(("photons_n_cut_0", "", *bins_a_n), "photons_n"))
    results.append(df.Histo1D(("photons_cos_theta_cut_0", "", 50, -1, 1), "photons_cos_theta"))

    results.append(df.Histo1D(("electrons_p_baseline", "", 130, 0, 130), "electrons_p"))
    results.append(df.Histo1D(("electrons_n_baseline", "", *bins_a_n), "electrons_n"))
    results.append(df.Histo1D(("electrons_cos_theta", "", 50, -1, 1), "electrons_ordered_cos_theta"))

   

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
    results.append(df.Histo1D(("electrons_cos_theta_cut_1", "", 50, -1, 1), "electrons_ordered_cos_theta"))
 
    

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

    
    #########
    ### CUT 2: Photons energy > 50
    #########
    
    df = df.Filter("photons_boosted.size()>0 ")  
    df = df.Define("cut2", "2")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut2"))
    
    results.append(df.Histo1D(("photons_p_cut_2", "",  130, 0, 130), "photons_boosted_p"))
    results.append(df.Histo1D(("photons_n_cut_2", "", *bins_a_n), "photons_boosted_n"))
    results.append(df.Histo1D(("photons_cos_theta_cut_2", "", 50, -1, 1), "photons_boosted_cos_theta"))
    results.append(df.Histo1D(("electrons_cos_theta_cut_2", "", 50, -1, 1), "electrons_ordered_cos_theta"))
 

    
     #########
    ### CUT 3: Cos Theta cut
    #########
    df = df.Filter(f"ROOT::VecOps::All(abs(photons_boosted_cos_theta) < {photon_cos_theta_max}) ") 
   
    df = df.Define("cut3", "3")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut3"))
    
    results.append(df.Histo1D(("photons_p_cut_3", "", 130, 0, 130), "photons_boosted_p"))
    results.append(df.Histo1D(("photons_n_cut_3", "", *bins_a_n), "photons_boosted_n"))
    results.append(df.Histo1D(("photons_cos_theta_cut_3", "", 50, -1, 1), "photons_boosted_cos_theta"))
    results.append(df.Histo1D(("electrons_cos_theta_cut_3", "", 50, -1, 1), "electrons_ordered_cos_theta"))


    df = df.Define("recopart_no_gamma", "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles, photons_boosted)",)
    df = df.Define("recopart_no_gamma_n","FCCAnalyses::ReconstructedParticle::get_n(recopart_no_gamma)") 
   
 
    results.append(df.Histo1D(("recopart_no_gamma_n_cut_0", "", 60, 0, 60), "recopart_no_gamma_n"))

    """
    #########
    ### CUT 4: Cos Theta cut on ee to reduce bhabhar
    #########
    df = df.Filter("electrons_ordered_cos_theta.size()<2 || (abs(electrons_ordered_cos_theta)[0]<0.8 && abs(electrons_ordered_cos_theta)[1]<0.8)") 
    
    df = df.Define("cut4", "4")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut4"))
 
    results.append(df.Histo1D(("photons_p_cut_4", "", 130, 0, 130), "photons_boosted_p"))
    results.append(df.Histo1D(("photons_n_cut_4", "", *bins_a_n), "photons_boosted_n"))
    results.append(df.Histo1D(("photons_cos_theta_cut_4", "", 50, -1, 1), "photons_boosted_cos_theta"))
    results.append(df.Histo1D(("electrons_cos_theta_cut_4", "", 50, -1, 1), "electrons_ordered_cos_theta"))
    """

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
   
    #########
    ### CUT 6: gamma recoil cut tight
    #########
    #df = df.Filter("123.5 < gamma_recoil_m && gamma_recoil_m < 126.5") 
    df = df.Filter(f"{signal_mass_min} < gamma_recoil_m && gamma_recoil_m < {signal_mass_max}") 

    df = df.Define("cut6", "6")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut6"))

    results.append(df.Histo1D(("gamma_recoil_m_tight_cut", "", 70, 80, 150), "gamma_recoil_m"))

   
    #define further variables for plotting
    #df = df.Define("photons_all_p", "FCCAnalyses::ReconstructedParticle::get_p(photons_all)")
    #df = df.Define("photons_boosted_p", "FCCAnalyses::ReconstructedParticle::get_p(photons_boosted)")
    #df = df.Define("photons_boosted_n","FCCAnalyses::ReconstructedParticle::get_n(photons_boosted)")  #number of photons per event
    
   
    #select highest energetic photon

    ########################
    # Final histograms
    ########################
    #results.append(df.Histo1D(("photons_all_p", "", 100, 0, 100), "photons_all_p"))
   # results.append(df.Histo1D(("photons_boosted_p", "", *bins_a_p), "photons_boosted_p"))
    #results.append(df.Histo1D(("photons_n", "", *bins_a_n), "photons_n"))
    #results.append(df.Histo1D(("photons_boosted_n", "", *bins_a_n), "photons_boosted_n"))
    
    #results.append(df.Histo1D(("zmumu_recoil_m", "", *bins_recoil), "zmumu_recoil_m"))   # see how recoil determined
   
    #need to select the highest energetic photon

    return results, weightsum