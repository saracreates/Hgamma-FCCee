def generate_datacard(process_paths, output_file, hist_name, uncertainty="1.05"):
    with open(output_file, "w") as f:
        # Header
        f.write("imax *\njmax *\nkmax *\n---------------\n")
        
        # Shapes
        for process, path in process_paths.items():
            f.write(f"shapes {process} * {path} {hist_name}\n")

        f.write("---------------\n---------------\n")
        f.write("#bin            bin1\nobservation     -1\n------------------------------\n")
        
        # Exclude data_obs from bin, process, and rate lines
        processes = [p for p in process_paths.keys() if p != "data_obs"]
        
        # Bin and process definitions
        f.write("bin          " + " ".join(["bin1"] * len(processes)) + "\n")
        f.write("process      " + " ".join(processes) + "\n")
        f.write("process      " + " ".join(map(str, range(len(processes)))) + "\n")
        f.write("rate         " + " ".join(["-1"] * len(processes)) + "\n")
        
        f.write("--------------------------------\n")
        # f.write("#bkg lnU      -              1.5\n")
        # f.write("#HWW_norm rateParam bin1 WW 1\n")
        # f.write("#ZZ_norm rateParam bin1 ZZ 1\n")
        # f.write("#Zqq_norm rateParam bin1 Zqq 1\n\n")
        
        # Systematic uncertainties (excluding data_obs and the first process, which is the signal)
        for i, process in enumerate(processes[1:]):  # Skip the first process (signal)
            f.write(f"{process}_norm lnN " + "- " * (i + 1) + uncertainty + " " + "- " * (len(processes) - i - 2) + "\n")


### ALTER SCRIPT HERE
ecm = 240 # 160, 240 or 365
data_path = f"/afs/cern.ch/work/s/saaumill/public/analyses/Hgamma-FCCee/outputs/{ecm}/histmaker/lvqq/"

# ONLY if needed - change user defined settings
cardname = f"datacard_Hgamma_HWW_{ecm}ecm.txt" # name of the output datacard
uncertainty = "1.01" # systematic uncertainty to apply to all backgrounds (except data_obs and signal)
if ecm == 160:
    hist_name = "gamma_recoil_m_very_tight_cut" # name of the histogram you want to fit on
else: 
    # hist_name = "gamma_recoil_m_tight_cut" # 0.5 GeV bins
    hist_name = "gamma_recoil_m_last_cut" # 1 GeV bins


process_paths = {
    "HWW": data_path + f"mgp8_ee_ha_ecm{ecm}_hww.root", # signal!
    "bba": data_path + f"wzp6_ee_bba_ecm{ecm}.root",
    "cca": data_path + f"wzp6_ee_cca_ecm{ecm}.root",
    "qqa": data_path + f"wzp6_ee_qqa_ecm{ecm}.root",
    "eea": data_path + f"wzp6_ee_eea_ecm{ecm}.root",
    "mumua": data_path + f"wzp6_ee_mumua_ecm{ecm}.root",
    "tautaua": data_path + f"wzp6_ee_tautaua_ecm{ecm}.root",
    "WW": data_path + f"p8_ee_WW_ecm{ecm}.root",
    "ZZ": data_path + f"p8_ee_ZZ_ecm{ecm}.root",
    "data_obs": data_path + f"mgp8_ee_ha_ecm{ecm}_hww.root"  # using signal as a placeholder for data
}

# modification for 240 ecm
if ecm == 240:
    # Add ZH as bkg
    process_paths["ZH"] = data_path + f"mgp8_ee_zh_ecm{ecm}.root"

# calls main function
generate_datacard(process_paths, cardname, hist_name, uncertainty)