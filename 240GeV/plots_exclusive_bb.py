# list of processes (mandatory)
processList = {
    'p8_ee_Hgamma_ecm240':    {'fraction':1, 'crossSection': 8.20481e-05}, 
    'p8_ee_qqgamma_ecm240':    {'fraction':1, 'crossSection': 6.9},  #what are the exact values here?
    'p8_ee_ccgamma_ecm240':    {'fraction':1, 'crossSection': 2.15},  #what are the exact values here?
    'p8_ee_bbgamma_ecm240':    {'fraction':1, 'crossSection': 2.35},  #what are the exact values here?
    'p8_ee_ZH_ecm240':    {'fraction':1, 'crossSection': 0.2},
    'p8_ee_WW_ecm240':    {'fraction':1},  
    'p8_ee_ZZ_ecm240':    {'fraction':1},
    'p8_ee_eegamma_ecm240':    {'fraction':1, 'crossSection': 190},  #what are the exact values here?
    'p8_ee_tautaugamma_ecm240':    {'fraction':1, 'crossSection': 0.77},  #what are the exact values here?
    'p8_ee_mumugamma_ecm240':    {'fraction':1, 'crossSection': 0.8},  #what are the exact values here?
}

# Production tag when running over EDM4Hep centrally produced events, this points to the yaml files for getting sample statistics (mandatory)
#prodTag = "FCCee/winter2023/IDEA/"

# Link to the dictonary that contains all the cross section informations etc... (mandatory)
procDict = "FCCee_procDict_winter2023_IDEA.json"


# Optional: output directory, default is local running directory
#outputDir   = "./outputs/histmaker/flavor"
outputDir   = "./outputs/histmaker/flavor/extended"

# Define the input dir (optional)
#inputDir    = "./outputs/treemaker/flavor/"
inputDir    = "./outputs/treemaker/flavor/extended"

includePaths = ["../../tutorial/functions.h"]

# optional: ncpus, default is 4, -1 uses all cores available
nCPUS = -1

# scale the histograms with the cross-section and integrated luminosity
doScale = True
intLumi = 10800000  # 5 /ab


# define some binning for various histograms
bins_a_p = (100, 0, 500) # 100 MeV bins
bins_a_n = (10, 0, 10) # 100 MeV bins

bins_count = (10, 0, 10)

bins_recojetsize = (5, 0, 5)
bins_score = (50, 0, 1)
bins_score_sum = (100, 0, 2)
bins_m_jj = (150, 0, 150)

collections = {
    "GenParticles": "Particle",
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


# build_graph function that contains the analysis logic, cuts and histograms (mandatory)
def build_graph(df, dataset):

    results = []
    df = df.Define("weight", "1.0")
    weightsum = df.Sum("weight")
    


    #########
    ### CUT 0: all events
    #########
    df = df.Define("cut0", "0")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut0"))

    df = df.Define("recojet_isB_size", "recojet_isB.size()")
    results.append(df.Histo1D(("recojet_isB_size", "", *bins_recojetsize), "recojet_isB_size"))


    #get the recoil mass:
    #isolation cut
    df = df.Define("photons_iso", "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.5)(photons_all, ReconstructedParticles)")  # is this correct?
    df = df.Define("photons_sel_iso","FCCAnalyses::ZHfunctions::sel_iso(0.2)(photons_all, photons_iso)",) # and this??
  
    
    df = df.Filter("photons_sel_iso.size()>0 ")
      
    #sort in p  and select highest energetic one
    df = df.Define("iso_highest_p","FCCAnalyses::ZHfunctions::sort_by_energy(photons_sel_iso)")

    #energy cut
    df = df.Define("photons_boosted", "FCCAnalyses::ReconstructedParticle::sel_p(60,100)(iso_highest_p)")
    
    df = df.Filter("photons_boosted.size()>0 ") 
        
    df = df.Define("photons_boosted_cos_theta","cos(FCCAnalyses::ReconstructedParticle::get_theta(photons_boosted))")
    df = df.Filter("ROOT::VecOps::All(abs(photons_boosted_cos_theta) < 0.9) ") 

    df = df.Define("cut1", "1")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut1"))
      
    df = df.Define("gamma_recoil", "FCCAnalyses::ReconstructedParticle::recoilBuilder(240)(photons_boosted)") 
    df = df.Define("gamma_recoil_m", "FCCAnalyses::ReconstructedParticle::get_mass(gamma_recoil)[0]")
    results.append(df.Histo1D(("gamma_recoil_m", "", 170, 80, 250), "gamma_recoil_m"))

    
    #########
    df = df.Define("recojet_isB0", "recojet_isB[0]")
    results.append(df.Histo1D(("recojet_isB0", "", *bins_score), "recojet_isB0"))

    df = df.Define("recojet_isB1", "recojet_isB[1]")
    results.append(df.Histo1D(("recojet_isB1", "", *bins_score), "recojet_isB1"))

    df = df.Define("scoresum_B", "recojet_isB[0] + recojet_isB[1]")
    results.append(df.Histo1D(("scoresum_B", "", *bins_score_sum), "scoresum_B"))

    df = df.Define("scorediv_B", "recojet_isB[0] / recojet_isB[1]")
    results.append(df.Histo1D(("scorediv_B", "", 30,0,3), "scorediv_B"))

    df = df.Define("recojet_isG0", "recojet_isG[0]")
    results.append(df.Histo1D(("recojet_isG0", "", *bins_score), "recojet_isG0"))

    df = df.Define("recojet_isG1", "recojet_isG[1]")
    results.append(df.Histo1D(("recojet_isG1", "", *bins_score), "recojet_isG1"))

    df = df.Define("scoresum_G", "recojet_isG[0] + recojet_isG[1]")
    results.append(df.Histo1D(("scoresum_G", "", *bins_score_sum), "scoresum_G"))

    df = df.Define("recojet_isU0", "recojet_isU[0]")
    results.append(df.Histo1D(("recojet_isU0", "", *bins_score), "recojet_isU0"))

    df = df.Define("recojet_isU1", "recojet_isU[1]")
    results.append(df.Histo1D(("recojet_isU1", "", *bins_score), "recojet_isU1"))

    df = df.Define("scoresum_U", "recojet_isU[0] + recojet_isU[1]")
    results.append(df.Histo1D(("scoresum_U", "", *bins_score_sum), "scoresum_U"))

    df = df.Define("recojet_isS0", "recojet_isS[0]")
    results.append(df.Histo1D(("recojet_isS0", "", *bins_score), "recojet_isS0"))

    df = df.Define("recojet_isS1", "recojet_isS[1]")
    results.append(df.Histo1D(("recojet_isS1", "", *bins_score), "recojet_isS1"))

    df = df.Define("scoresum_S", "recojet_isS[0] + recojet_isS[1]")
    results.append(df.Histo1D(("scoresum_S", "", *bins_score_sum), "scoresum_S"))

    df = df.Define("recojet_isC0", "recojet_isC[0]")
    results.append(df.Histo1D(("recojet_isC0", "", *bins_score), "recojet_isC0"))

    df = df.Define("recojet_isC1", "recojet_isC[1]")
    results.append(df.Histo1D(("recojet_isC1", "", *bins_score), "recojet_isC1"))

    df = df.Define("scoresum_C", "recojet_isC[0] + recojet_isC[1]")
    results.append(df.Histo1D(("scoresum_C", "", *bins_score_sum), "scoresum_C"))

    df = df.Define("recojet_isD0", "recojet_isD[0]")
    results.append(df.Histo1D(("recojet_isD0", "", *bins_score), "recojet_isD0"))

    df = df.Define("recojet_isD1", "recojet_isD[1]")
    results.append(df.Histo1D(("recojet_isD1", "", *bins_score), "recojet_isD1"))

    df = df.Define("scoresum_D", "recojet_isD[0] + recojet_isD[1]")
    results.append(df.Histo1D(("scoresum_D", "", *bins_score_sum), "scoresum_D"))

    df = df.Define("recojet_isTAU0", "recojet_isTAU[0]")
    results.append(df.Histo1D(("recojet_isTAU0", "", *bins_score), "recojet_isTAU0"))

    df = df.Define("recojet_isTAU1", "recojet_isTAU[1]")
    results.append(df.Histo1D(("recojet_isTAU1", "", *bins_score), "recojet_isTAU1"))

    df = df.Define("scoresum_Tau", "recojet_isTAU[0] + recojet_isTAU[1]")
    results.append(df.Histo1D(("scoresum_Tau", "", *bins_score_sum), "scoresum_Tau"))


    #########
    ### CUT 2: filter on b score
    #########
    
    df = df.Filter("scoresum_B>1 ") 
   
    #df = df.Filter("scoresum_G>1") 
    #df = df.Filter("scoresum_Tau>1 ")

    df = df.Define("cut2", "2")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut2"))

    results.append(df.Histo1D(("gamma_recoil_m_signal_cut", "", 170, 80, 250), "gamma_recoil_m"))

    #########
    ### CUT 3 cut & count
    #########
    
    df = df.Filter("123.5 < gamma_recoil_m && gamma_recoil_m < 126.5") 

    df = df.Define("cut3", "3")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut3"))

    results.append(df.Histo1D(("gamma_recoil_m_cut_3", "", 70, 80, 150), "gamma_recoil_m"))

    return results, weightsum