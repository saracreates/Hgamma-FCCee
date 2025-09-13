#!/usr/bin/env python
"""
Create plots out of the histograms produced in previous stages
"""
import sys
import os
import os.path
import ntpath
import importlib
import copy
import re
import logging
import ROOT  # type: ignore

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)


LOGGER = logging.getLogger("FCCAnalyses.plot")


# _____________________________________________________________________________
def removekey(d: dict, key: str) -> dict:
    """
    Remove dictionary element.
    """
    r = dict(d)
    del r[key]
    return r


def sorted_dict_values(dic: dict) -> list:
    """'
    Sort values in the dictionary.
    """
    keys = sorted(dic)
    return [dic[key] for key in keys]


def formatStatUncHist(hists, name, hstyle=3254):
    hist_tot = hists[0].Clone(name + "_unc")
    for h in hists[1:]:
        hist_tot.Add(h)
    hist_tot.SetFillColor(ROOT.kBlack)
    hist_tot.SetMarkerSize(0)
    hist_tot.SetLineWidth(0)
    hist_tot.SetFillStyle(hstyle)
    return hist_tot


# _____________________________________________________________________________
def determine_lumi_scaling(config: dict[str, any], infile: object, initial_scale: float = 1.0) -> float:
    """
    Determine whether to (re)scale histograms in the file to luminosity.
    """
    scale: float = initial_scale

    # Check if histograms were already scaled to lumi
    try:
        scaled: bool = infile.scaled.GetVal()
    except AttributeError:
        LOGGER.error(
            "Input file does not contain scaling " "information!\n  %s\nAborting...",
            infile.GetName(),
        )
        sys.exit(3)

    print("found scaling", scaled)
    if scaled:
        try:
            int_lumi_in_file: float = infile.intLumi.GetVal()
        except AttributeError:
            LOGGER.error(
                "Can not load integrated luminosity " "value from the input file!\n  %s\n" "Aborting...",
                infile.GetName(),
            )

        if config["int_lumi"] != int_lumi_in_file:
            LOGGER.warning(
                "Histograms are already scaled to different "
                "luminosity value!\n"
                "Luminosity in the input file is %s pb-1 and "
                "luminosity requested in plots script is %s pb-1.",
                int_lumi_in_file,
                config["int_lumi"],
            )
            if config["do_scale"]:
                LOGGER.warning(
                    "Rescaling from %s pb-1 to %s pb-1...",
                    int_lumi_in_file,
                    config["int_lumi"],
                )
                scale *= config["int_lumi"] / int_lumi_in_file

    else:
        if config["do_scale"]:
            scale = scale * config["int_lumi"]

    return scale


# _____________________________________________________________________________
def load_hists(var: str, label: str, sel: str, config: dict[str, any], rebin: int) -> tuple[dict[str, any], dict[str:any]]:
    """
    Load all histograms needed for the plot
    """

    try:
        signal = config["plots"][label]["signal"]
    except KeyError:
        signal = {}

    try:
        backgrounds = config["plots"][label]["backgrounds"]
    except KeyError:
        backgrounds = {}

    hsignal = {}
    for s in signal:
        hsignal[s] = []
        for filepathstem in signal[s]:
            infilepath = config["input_dir"] + filepathstem + "_" + sel + "_histo.root"
            if not os.path.isfile(infilepath):
                LOGGER.info('File "%s" not found!\nSkipping it...', infilepath)
                continue

            with ROOT.TFile(infilepath, "READ") as infile:
                hist = copy.deepcopy(infile.Get(var))
                hist.SetDirectory(0)

                scale = determine_lumi_scaling(config, infile, config["scale_sig"])
            hist.Scale(scale)
            hist.Rebin(rebin)

            if len(hsignal[s]) == 0:
                hsignal[s].append(hist)
            else:
                hist.Add(hsignal[s][0])
                hsignal[s][0] = hist

    hbackgrounds = {}
    for b in backgrounds:
        hbackgrounds[b] = []
        for filepathstem in backgrounds[b]:
            infilepath = config["input_dir"] + filepathstem + "_" + sel + "_histo.root"
            if not os.path.isfile(infilepath):
                LOGGER.info('File "%s" not found!\nSkipping it...', infilepath)
                continue

            with ROOT.TFile(infilepath) as infile:
                hist = copy.deepcopy(infile.Get(var))
                hist.SetDirectory(0)

                scale = determine_lumi_scaling(config, infile, config["scale_bkg"])
            hist.Scale(scale)
            hist.Rebin(rebin)

            if len(hbackgrounds[b]) == 0:
                hbackgrounds[b].append(hist)
            else:
                hist.Add(hbackgrounds[b][0])
                hbackgrounds[b][0] = hist

    for s in hsignal:
        if len(hsignal[s]) == 0:
            hsignal = removekey(hsignal, s)

    for b in hbackgrounds:
        if len(hbackgrounds[b]) == 0:
            hbackgrounds = removekey(hbackgrounds, b)

    return hsignal, hbackgrounds


# _____________________________________________________________________________
def mapHistosFromHistmaker(config: dict[str, any], hist_name: str, param, hist_cfg):

    rebin = hist_cfg["rebin"] if "rebin" in hist_cfg else 1
    # LOGGER.info("Get histograms for %s", hist_name)

    signal = param.procs.get("signal", {})
    backgrounds = param.procs.get("backgrounds", {})
    scaleSig = hist_cfg["scaleSig"] if "scaleSig" in hist_cfg else 1
    normalize = hist_cfg.get("density", False)

    hsignal = {}
    for s in signal:
        hsignal[s] = []
        for f in signal[s]:
            fin = f"{param.inputDir}/{f}.root"
            if not os.path.isfile(fin):
                LOGGER.info('File "%s" not found!\nSkipping it...', fin)
                continue

            with ROOT.TFile(fin) as tf:
                # print(f"Getting histogram {hist_name} from {fin}")
                h = tf.Get(hist_name)
                hh = copy.deepcopy(h)
                hh.SetDirectory(0)
            # LOGGER.info("ScaleSig: %g", scaleSig)
            hh.Scale(param.intLumi * scaleSig)
            hh.Rebin(rebin)
            if normalize:  # Normalize histogram if required
                integral = hh.Integral()
                if integral > 0:
                    hh.Scale(1.0 / integral)
            if len(hsignal[s]) == 0:
                hsignal[s].append(hh)
            else:
                hh.Add(hsignal[s][0])
                hsignal[s][0] = hh

    hbackgrounds = {}
    for b in backgrounds:
        hbackgrounds[b] = []
        for f in backgrounds[b]:
            fin = f"{param.inputDir}/{f}.root"
            if not os.path.isfile(fin):
                LOGGER.info('File "%s" not found!\nSkipping it...', fin)
                continue

            with ROOT.TFile(fin) as tf:
                # print(f"Getting histogram {hist_name} from {fin}")
                h = tf.Get(hist_name)
                hh = copy.deepcopy(h)
                hh.SetDirectory(0)
            hh.Scale(param.intLumi)
            hh.Rebin(rebin)
            if normalize:  # Normalize histogram if required
                integral = hh.Integral()
                if integral > 0:
                    hh.Scale(1.0 / integral)
            if len(hbackgrounds[b]) == 0:
                hbackgrounds[b].append(hh)
            else:
                hh.Add(hbackgrounds[b][0])
                hbackgrounds[b][0] = hh

    for s in list(hsignal.keys()):
        if len(hsignal[s]) == 0:
            del hsignal[s]

    for b in list(hbackgrounds.keys()):
        if len(hbackgrounds[b]) == 0:
            del hbackgrounds[b]

    if not hsignal:
        LOGGER.error("No signal input files found!\nAborting...")
        sys.exit(3)

    return hsignal, hbackgrounds


# _____________________________________________________________________________
def runPlots(
    config: dict[str, any],
    args,
    var,
    sel,
    script_module,
    hsignal,
    hbackgrounds,
    extralab,
):

    # Below are settings for separate signal and background legends
    if config["split_leg"]:
        legsize = 0.04 * (len(hsignal))
        legsize2 = 0.04 * (len(hbackgrounds))
        leg = ROOT.TLegend(0.15, 0.60 - legsize, 0.50, 0.62)
        leg2 = ROOT.TLegend(0.60, 0.60 - legsize2, 0.88, 0.62)

        if config["leg_position"][0] is not None and config["leg_position"][2] is not None:
            leg.SetX1(config["leg_position"][0])
            leg.SetX2((config["leg_position"][0] + config["leg_position"][2]) / 2)
            leg2.SetX2((config["leg_position"][0] + config["leg_position"][2]) / 2)
            leg2.SetX2(config["leg_position"][0])
        if config["leg_position"][1] is not None:
            leg.SetY1(config["leg_position"][1])
            leg2.SetY1(config["leg_position"][1])
        if config["leg_position"][3] is not None:
            leg.SetY2(config["leg_position"][3])
            leg2.SetY2(config["leg_position"][3])

        leg2.SetFillColor(0)
        leg2.SetFillStyle(0)
        leg2.SetLineColor(0)
        leg2.SetShadowColor(10)
        leg2.SetTextSize(config["legend_text_size"])
        leg2.SetTextFont(42)
    else:
        legsize = 0.04 * (len(hbackgrounds) + len(hsignal))
        leg = ROOT.TLegend(0.60, 0.86 - legsize, 0.96, 0.88)
        leg2 = None

        if config["leg_position"][0] is not None:
            leg.SetX1(config["leg_position"][0])
        if config["leg_position"][1] is not None:
            leg.SetY1(config["leg_position"][1])
        if config["leg_position"][2] is not None:
            leg.SetX2(config["leg_position"][2])
        if config["leg_position"][3] is not None:
            leg.SetY2(config["leg_position"][3])

    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetLineColor(0)
    leg.SetShadowColor(10)
    leg.SetTextSize(config["legend_text_size"])
    leg.SetTextFont(42)

    for s in hsignal:
        leg.AddEntry(hsignal[s][0], script_module.legend[s], "l")

    for b in hbackgrounds:
        if config["split_leg"]:
            leg2.AddEntry(hbackgrounds[b][0], script_module.legend[b], "f")
        else:
            leg.AddEntry(hbackgrounds[b][0], script_module.legend[b], "f")

    yields = {}
    for s in hsignal:
        yields[s] = [
            script_module.legend[s],
            hsignal[s][0].Integral(0, -1),
            hsignal[s][0].GetEntries(),
        ]
    for b in hbackgrounds:
        yields[b] = [
            script_module.legend[b],
            hbackgrounds[b][0].Integral(0, -1),
            hbackgrounds[b][0].GetEntries(),
        ]

    histos = []
    colors = []

    nsig = len(hsignal)
    nbkg = len(hbackgrounds)

    for sig in hsignal:
        histos.append(hsignal[sig][0])
        colors.append(script_module.colors[sig])

    for bkg in hbackgrounds:
        histos.append(hbackgrounds[bkg][0])
        colors.append(script_module.colors[bkg])

    lt = "FCC-hh Simulation (Delphes)"
    rt = f"#sqrt{{s}} = {script_module.energy:.1f} TeV,   " f'{config["int_lumi_label"]}'

    if "ee" in script_module.collider:
        lt = "FCC-ee Simulation"
        rt = f"#sqrt{{s}} = {script_module.energy:.1f} GeV,   " f'{config["int_lumi_label"]}'

    customLabel = ""
    try:
        customLabel = script_module.customLabel
    except AttributeError:
        LOGGER.debug("No custom label, using nothing...")

    if "AAAyields" in var:
        drawStack(
            config,
            var,
            "events",
            leg,
            lt,
            rt,
            script_module.formats,
            script_module.outdir + "/" + sel,
            False,
            True,
            histos,
            colors,
            script_module.ana_tex,
            extralab,
            customLabel,
            nsig,
            nbkg,
            leg2,
            yields,
            config["plot_stat_unc"],
        )
        return

    if "stack" in script_module.stacksig:
        if "lin" in script_module.yaxis:
            drawStack(
                config,
                var + "_stack_lin",
                "events",
                leg,
                lt,
                rt,
                script_module.formats,
                script_module.outdir + "/" + sel,
                False,
                True,
                histos,
                colors,
                script_module.ana_tex,
                extralab,
                customLabel,
                nsig,
                nbkg,
                leg2,
                yields,
                config["plot_stat_unc"],
            )
        if "log" in script_module.yaxis:
            drawStack(
                config,
                var + "_stack_log",
                "events",
                leg,
                lt,
                rt,
                script_module.formats,
                script_module.outdir + "/" + sel,
                True,
                True,
                histos,
                colors,
                script_module.ana_tex,
                extralab,
                customLabel,
                nsig,
                nbkg,
                leg2,
                yields,
                config["plot_stat_unc"],
            )
        if "lin" not in script_module.yaxis and "log" not in script_module.yaxis:
            LOGGER.info("Unrecognized option in formats, should be " "['lin','log']")

    if "nostack" in script_module.stacksig:
        if "lin" in script_module.yaxis:
            drawStack(
                config,
                var + "_nostack_lin",
                "events",
                leg,
                lt,
                rt,
                script_module.formats,
                script_module.outdir + "/" + sel,
                False,
                False,
                histos,
                colors,
                script_module.ana_tex,
                extralab,
                customLabel,
                nsig,
                nbkg,
                leg2,
                yields,
                config["plot_stat_unc"],
            )
        if "log" in script_module.yaxis:
            drawStack(
                config,
                var + "_nostack_log",
                "events",
                leg,
                lt,
                rt,
                script_module.formats,
                script_module.outdir + "/" + sel,
                True,
                False,
                histos,
                colors,
                script_module.ana_tex,
                extralab,
                customLabel,
                nsig,
                nbkg,
                leg2,
                yields,
                config["plot_stat_unc"],
            )
        if "lin" not in script_module.yaxis and "log" not in script_module.yaxis:
            LOGGER.info("Unrecognised option in formats, should be " "['lin','log']")
    if "stack" not in script_module.stacksig and "nostack" not in script_module.stacksig:
        LOGGER.info("Unrecognized option in stacksig, should be " "['stack','nostack']")


# _____________________________________________________________________________
def runPlotsHistmaker(config: dict[str, any], args, hist_name: str, param, hist_cfg):

    output = hist_cfg["output"]
    hsignal, hbackgrounds = mapHistosFromHistmaker(config, hist_name, param, hist_cfg)

    # only plot requested processes
    if "processes" in hist_cfg:
        hsignal = {k: v for k, v in hsignal.items() if k in hist_cfg["processes"]}
        hbackgrounds = {k: v for k, v in hbackgrounds.items() if k in hist_cfg["processes"]}

    # treat everything as signal if density true
    # merge two dicts into signal and empty background
    normalize = hist_cfg.get("density", False)
    if normalize:
        hsignal.update(hbackgrounds)
        hbackgrounds = {}

    if hasattr(param, "splitLeg"):
        splitLeg = param.splitLeg
    else:
        splitLeg = False

    if hasattr(param, "plotStatUnc"):
        plotStatUnc = param.plotStatUnc
    else:
        plotStatUnc = False

    # Below are settings for separate signal and background legends
    if splitLeg:
        legsize = 0.04 * (len(hsignal))
        legsize2 = 0.04 * (len(hbackgrounds))
        legCoord = [0.15, 0.60 - legsize, 0.50, 0.62]
        leg2 = ROOT.TLegend(0.60, 0.60 - legsize2, 0.88, 0.62)
        leg2.SetFillColor(0)
        leg2.SetFillStyle(0)
        leg2.SetLineColor(0)
        leg2.SetShadowColor(10)
        leg2.SetTextSize(config["legend_text_size"])
        leg2.SetTextFont(42)
    else:
        legsize = 0.04 * (len(hbackgrounds) + len(hsignal))
        legCoord = [0.55, 0.86 - legsize, 0.92, 0.88]
        try:
            legCoord = param.legendCoord
        except AttributeError:
            LOGGER.debug("No legCoord, using default one...")
            legCoord = [0.62, 0.86 - legsize, 0.86, 0.88]
        leg2 = None

    leg = ROOT.TLegend(
        (config["leg_position"][0] if config["leg_position"][0] is not None else legCoord[0]),
        (config["leg_position"][1] if config["leg_position"][1] is not None else legCoord[1]),
        (config["leg_position"][2] if config["leg_position"][2] is not None else legCoord[2]),
        (config["leg_position"][3] if config["leg_position"][3] is not None else legCoord[3]),
    )
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetLineColor(0)
    leg.SetShadowColor(10)
    leg.SetTextSize(config["legend_text_size"])
    leg.SetTextFont(42)

    for s in hsignal:
        signal_legend = param.legend[s]
        if hist_cfg.get("scaleSig", None):
            signal_legend = f"{param.legend[s]} x {int(hist_cfg['scaleSig'])}"
        leg.AddEntry(hsignal[s][0], signal_legend, "l")

    for b in hbackgrounds:
        if splitLeg:
            leg2.AddEntry(hbackgrounds[b][0], param.legend[b], "f")
        else:
            leg.AddEntry(hbackgrounds[b][0], param.legend[b], "f")

    yields = {}
    for s in hsignal:
        yields[s] = [
            param.legend[s],
            hsignal[s][0].Integral(0, -1),
            hsignal[s][0].GetEntries(),
        ]
    for b in hbackgrounds:
        yields[b] = [
            param.legend[b],
            hbackgrounds[b][0].Integral(0, -1),
            hbackgrounds[b][0].GetEntries(),
        ]

    if hist_name == "cutFlow":
        cutFlowHist = hsignal[list(hsignal.keys())[0]][0]

        funcs = cutFlowHist.GetListOfFunctions()
        for item in funcs:
            if item.InheritsFrom("TObjString"):
                saved_selections = item.GetString().Data()
                print("Selections were:")
                print(saved_selections)
                hist_cfg["xtitle"] = saved_selections.split("\n")
                break

        extract_cutflow_data(param, hist_name, hist_cfg, hsignal, hbackgrounds)

    histos = []
    colors = []

    nsig = len(hsignal)
    nbkg = len(hbackgrounds)

    for s in hsignal:
        histos.append(hsignal[s][0])
        colors.append(param.colors[s])

    for b in hbackgrounds:
        histos.append(hbackgrounds[b][0])
        colors.append(param.colors[b])

    xtitle = hist_cfg["xtitle"] if "xtitle" in hist_cfg else ""
    ytitle = hist_cfg["ytitle"] if "ytitle" in hist_cfg else "Events"
    xmin = hist_cfg["xmin"] if "xmin" in hist_cfg else -1
    xmax = hist_cfg["xmax"] if "xmax" in hist_cfg else -1
    ymin = hist_cfg["ymin"] if "ymin" in hist_cfg else -1
    ymax = hist_cfg["ymax"] if "ymax" in hist_cfg else -1
    stack = hist_cfg["stack"] if "stack" in hist_cfg else False
    logy = hist_cfg["logy"] if "logy" in hist_cfg else False
    extralab = hist_cfg["extralab"] if "extralab" in hist_cfg else ""

    intLumiab = param.intLumi / 1e06
    intLumi = f"L = {intLumiab:.0f} ab^{{-1}}"
    if hasattr(param, "intLumiLabel"):
        intLumi = getattr(param, "intLumiLabel")

    lt = "FCC-hh Simulation (Delphes)"
    rt = f"#sqrt{{s}} = {param.energy} TeV, {intLumi}"

    if "ee" in param.collider:
        lt = "       FCC-ee Simulation" # so that 10^3 does not collide 
        rt = f"#sqrt{{s}} = {param.energy} GeV,   {intLumi}"

    customLabel = ""
    try:
        customLabel = param.customLabel
    except AttributeError:
        LOGGER.debug("No customLabel, using nothing...")

    if stack:
        if logy:
            drawStack(
                config,
                output,
                ytitle,
                leg,
                lt,
                rt,
                param.formats,
                param.outdir,
                True,
                True,
                histos,
                colors,
                param.ana_tex,
                extralab,
                customLabel,
                nsig,
                nbkg,
                leg2,
                yields,
                plotStatUnc,
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                xtitle=xtitle,
            )
        else:
            drawStack(
                config,
                output,
                ytitle,
                leg,
                lt,
                rt,
                param.formats,
                param.outdir,
                False,
                True,
                histos,
                colors,
                param.ana_tex,
                extralab,
                customLabel,
                nsig,
                nbkg,
                leg2,
                yields,
                plotStatUnc,
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                xtitle=xtitle,
            )

    else:
        if logy:
            drawStack(
                config,
                output,
                ytitle,
                leg,
                lt,
                rt,
                param.formats,
                param.outdir,
                True,
                False,
                histos,
                colors,
                param.ana_tex,
                extralab,
                customLabel,
                nsig,
                nbkg,
                leg2,
                yields,
                plotStatUnc,
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                xtitle=xtitle,
            )
        else:
            drawStack(
                config,
                output,
                ytitle,
                leg,
                lt,
                rt,
                param.formats,
                param.outdir,
                False,
                False,
                histos,
                colors,
                param.ana_tex,
                extralab,
                customLabel,
                nsig,
                nbkg,
                leg2,
                yields,
                plotStatUnc,
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                xtitle=xtitle,
            )


# ________________________________________________________________________________________
def runPlotsHistmaker2D(config: dict[str, any], args, hist_name: str, param, hist_cfg):

    # Extract parameters from the configuration dictionary
    output_base = hist_cfg.get("output", "output")
    xtitle = hist_cfg.get("xtitle", "")
    ytitle = hist_cfg.get("ytitle", hist_cfg.get("ytile", ""))  # fallback if key is 'ytile'

    # Retrieve axis ranges only if they exist; if not, leave them None
    xmin = hist_cfg.get("xmin", None)
    xmax = hist_cfg.get("xmax", None)
    ymin = hist_cfg.get("ymin", None)
    ymax = hist_cfg.get("ymax", None)

    # Retrieve z range if specified
    zmin = hist_cfg.get("zmin", None)
    zmax = hist_cfg.get("zmax", None)

    logz = hist_cfg.get("logz", False)
    processes = hist_cfg.get("processes", [])

    signal, backgrounds = mapHistosFromHistmaker(config, hist_name, param, hist_cfg)

    LOGGER.info("Get histograms for %s", hist_name)

    for proc in processes:
        output_name = f"{output_base}_{proc}"

        hist = signal.get(proc, None)
        if not hist:
            hist = backgrounds.get(proc, None)

        # If hist is a list of histograms, sum them up
        if isinstance(hist, list):
            if not hist:  # empty list check
                print(f"Warning: No histograms for {hist_name} and process {proc}")
                continue
            sumHist = hist[0].Clone()
            sumHist.SetDirectory(0)
            for h in hist[1:]:
                sumHist.Add(h)
            hist = sumHist

        if not hist:
            print(f"Warning: histogram {hist_name} not found for process {proc}")
            continue

        # check integral integral
        nBinsX = hist.GetNbinsX()
        nBinsY = hist.GetNbinsY()
        integral = hist.Integral(0, nBinsX + 1, 0, nBinsY + 1)
        # print("Integral (including under/overflow):", integral)

        # Create a canvas
        c = ROOT.TCanvas("c", "c", 800, 800)
        c.SetLeftMargin(0.15)
        c.SetRightMargin(0.15)

        # Set logz if required
        if logz:
            c.SetLogz()

        # Adjust axis titles
        hist.GetXaxis().SetTitle(xtitle)
        hist.GetYaxis().SetTitle(ytitle)
        hist.SetTitle("")  # remove general title if desired

        # Set axis ranges if specified
        if xmin is not None and xmax is not None:
            hist.GetXaxis().SetRangeUser(xmin, xmax)
        if ymin is not None and ymax is not None:
            hist.GetYaxis().SetRangeUser(ymin, ymax)

        # Set z range if specified
        if zmin is not None:
            hist.SetMinimum(zmin)
        if zmax is not None:
            hist.SetMaximum(zmax)

        # Draw the histogram with "COLZ" option
        hist.Draw("COLZ")

        # Prepare luminosity label
        intLumiab = param.intLumi / 1e06
        intLumi = f"L = {intLumiab:.0f} ab^{{-1}}"
        if hasattr(param, "intLumiLabel"):
            intLumi = getattr(param, "intLumiLabel")

        leftText = "FCC-hh Simulation (Delphes)"
        rightText = f"#sqrt{{s}} = {param.energy:.1f} TeV,   {intLumi}"

        if "ee" in param.collider:
            leftText = "FCC-ee Simulation"
            rightText = f"#sqrt{{s}} = {param.energy:.1f} GeV,   {intLumi}"

        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextAlign(11)
        latex.SetTextSize(0.03)

        text = leftText
        latex.DrawLatex(0.15, 0.91, text)

        latex.SetTextAlign(31)
        latex.SetTextSize(0.03)

        text = "#bf{" + rightText + "}"
        latex.DrawLatex(0.90, 0.91, text)

        # Update canvas and save the plot
        c.Update()

        for f in param.formats:
            out_file = os.path.join(param.outdir, output_name) + "." + f
            c.SaveAs(out_file)

        c.Close()

    return 0


# _____________________________________________________________________________
def extract_cutflow_data(param, hist_name, hist_cfg, hsignal, hbackgrounds):
    import pandas as pd
    import numpy as np
    import os
    import subprocess

    pd.set_option("display.max_colwidth", None)
    if hist_name != "cutFlow":
        raise ValueError('This function only works for "cutFlow".')

    def round_to_significant(x, sig_digits=4):
        if x == 0:
            return 0
        return int(round(x, -int(np.floor(np.log10(abs(x)))) + (sig_digits - 1)))

    def sanitize_latex(text):
        # Replace '#' and 'ˆ' if needed
        if not isinstance(text, str):
            return text
        text = text.replace("#", "\\")
        text = text.replace("ˆ", "^")
        text = text.replace(" ", "\,")
        return text

    data = []
    signal_names = list(hsignal.keys())
    background_names = list(hbackgrounds.keys())

    for i, label in enumerate(hist_cfg["xtitle"]):
        process_yields = {}

        total_signal_yield = sum(hsignal[s][0].Integral(i + 1, i + 1) for s in signal_names)
        total_background_yield = sum(hbackgrounds[b][0].Integral(i + 1, i + 1) for b in background_names)

        for s in signal_names:
            yield_val = hsignal[s][0].Integral(i + 1, i + 1)
            initial_yield = hsignal[s][0].Integral(1, 1)
            efficiency = (yield_val / initial_yield * 100) if (i > 0 and initial_yield != 0) else 100.0
            process_yields[s] = f"{round_to_significant(yield_val)} ({efficiency:.1f})%"

        for b in background_names:
            yield_val = hbackgrounds[b][0].Integral(i + 1, i + 1)
            initial_yield = hbackgrounds[b][0].Integral(1, 1)
            efficiency = (yield_val / initial_yield * 100) if (i > 0 and initial_yield != 0) else 100.0
            # Notice you have parentheses around efficiency here, leaving it as is:
            process_yields[b] = f"{round_to_significant(yield_val)} ({efficiency:.1f})%"

        if total_signal_yield > 0:
            # formula for precision: sqrt((S + B) / S) only valid for S + B, see https://www.pp.rhul.ac.uk/~cowan/stat/medsig/medsigNote.pdf 
            # prec_value_old = 100.0 * np.sqrt(total_signal_yield + total_background_yield) / total_signal_yield
            try:
                prec_value = 100.0 * (1/(np.sqrt(2 * ((total_signal_yield + total_background_yield)* np.log(1 + (total_signal_yield/total_background_yield))-total_signal_yield))))
                # print(f"Precision for {label}. \t \t Old calculation: {prec_value_old:.1f}%, \t new calculation: {prec_value:.1f}%")
                prec_str = f"{prec_value:.1f}%"
            except ZeroDivisionError:
                prec_str = "inf"
        else:
            prec_str = "inf"

        row = {"Selection Cut": label}
        row.update(process_yields)
        row["prec (\\%)"] = prec_str
        data.append(row)

    df = pd.DataFrame(data)

    # Reorder columns so that 'prec (%)' is last
    cols = list(df.columns)
    if "prec (%)" in cols:
        cols.remove("prec (%)")
        cols.append("prec (%)")
    df = df[cols]

    print(df.to_string(index=False))

    # Apply sanitize_latex only to "Selection Cut" column
    df["Selection Cut"] = df["Selection Cut"].apply(sanitize_latex)

    # Replace '%' with '\%' across the entire DataFrame
    df = df.replace("%", r"\%", regex=True)

    # Wrap the selection cuts in math mode with \rm
    df["Selection Cut"] = df["Selection Cut"].apply(lambda x: f"$\\rm {x}$")

    # create output directory if does not exist
    if not os.path.exists(param.outdir):
        os.makedirs(param.outdir)

    output_table = os.path.join(param.outdir, hist_name + "_cutflow.tex")
    pdf_file = os.path.join(param.outdir, hist_name + "_cutflow.pdf")

    latex_preamble = r"""\documentclass[7pt]{article}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\usepackage{pdflscape} % add this
\usepackage[margin=1in]{geometry}

\begin{document}
\pagestyle{empty} % No page numbers, headers, or footers
\begin{landscape} % start rotated environment
\small
"""
    latex_postamble = r"""
\end{landscape} % end rotated environment
\end{document}
"""

    table_latex = df.to_latex(index=False, escape=False)
    with open(output_table, "w") as f:
        f.write(latex_preamble)
        f.write(table_latex)
        f.write(latex_postamble)

    ### prints the table in latex style to the consolesss
    os.system("cat " + output_table)

    pdflatex_cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", param.outdir, output_table]
    pdf_generated = False

    try:
        for _ in range(2):
            result = subprocess.run(pdflatex_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            if result.returncode != 0:
                print("pdflatex failed to compile the PDF. Output:")
                print(result.stdout.decode("utf-8", errors="ignore"))
                print(result.stderr.decode("utf-8", errors="ignore"))
                break
        if os.path.isfile(pdf_file):
            pdf_generated = True
    except subprocess.TimeoutExpired:
        print("pdflatex timed out. Check if there are infinite loops or problematic packages.")

    if pdf_generated:
        if os.path.isfile(output_table):
            os.remove(output_table)
        base_name = os.path.splitext(os.path.basename(output_table))[0]
        for ext in [".log", ".aux", ".out"]:
            aux_file = os.path.join(param.outdir, base_name + ext)
            if os.path.isfile(aux_file):
                os.remove(aux_file)
    else:
        print("No PDF generated. LaTeX sources are left for debugging.")

    return df


# _____________________________________________________________________________
def drawStack(
    config,
    name,
    ylabel,
    legend,
    leftText,
    rightText,
    formats,
    directory,
    logY,
    stacksig,
    histos,
    colors,
    ana_tex,
    extralab,
    customLabel,
    nsig,
    nbkg,
    legend2=None,
    yields=None,
    plotStatUnc=False,
    xmin=-1,
    xmax=-1,
    ymin=-1,
    ymax=-1,
    xtitle="",
):

    canvas = ROOT.TCanvas(name, name, 800, 800)
    canvas.SetLogy(logY)
    canvas.SetTicks(1, 1)
    canvas.SetLeftMargin(0.14)
    canvas.SetRightMargin(0.08)

    sumhistos = histos[0].Clone()
    iterh = iter(histos)
    next(iterh)

    unit = "GeV"
    if "TeV" in str(histos[0].GetXaxis().GetTitle()):
        unit = "TeV"

    if unit in str(histos[0].GetXaxis().GetTitle()):
        bwidth = sumhistos.GetBinWidth(1)
        if bwidth.is_integer():
            ylabel += f" / {bwidth} {unit}"
        else:
            ylabel += f" / {bwidth:.2f} {unit}"

    nbins = 1 if not isinstance(xtitle, list) else len(xtitle)
    h_dummy = ROOT.TH1D("h_dummy", "", nbins, 0, nbins)
    if nbins == 1:
        h_dummy.GetXaxis().SetTitle(histos[0].GetXaxis().GetTitle() if xtitle == "" else xtitle)
        h_dummy.GetYaxis().SetTitleOffset(1.95)
        h_dummy.GetXaxis().SetTitleOffset(1.2 * h_dummy.GetXaxis().GetTitleOffset())
    else:  # for cutflow plots
        for i, label in enumerate(xtitle):
            h_dummy.GetXaxis().SetBinLabel(i + 1, label)
        h_dummy.GetXaxis().LabelsOption("u")
        h_dummy.GetXaxis().SetLabelSize(1.1 * h_dummy.GetXaxis().GetLabelSize())
        h_dummy.GetXaxis().SetLabelOffset(1.5 * h_dummy.GetXaxis().GetLabelOffset())
        #h_dummy.GetXaxis().SetRangeUser(-0.5, len(xtitle) - 0.5)
    h_dummy.GetYaxis().SetTitle(ylabel)

    for h in iterh:
        sumhistos.Add(h)

    if logY:
        canvas.SetLogy(1)

    # define stacked histo
    hStack = ROOT.THStack("hstack", "")
    hStackBkg = ROOT.THStack("hstackbkg", "")
    hStackSig = ROOT.THStack("hstacksig", "")
    BgMCHistYieldsDic = {}

    # first plot backgrounds (sorted by the yields)
    for i in range(nsig, nsig + nbkg):
        h = histos[i]
        h.SetLineWidth(1)
        h.SetLineColor(ROOT.kBlack)
        h.SetFillColor(colors[i])
        if h.Integral() > 0:
            BgMCHistYieldsDic[h.Integral()] = h
        else:
            BgMCHistYieldsDic[-1 * nbkg] = h
    # sort stack by yields (smallest to largest)
    BgMCHistYieldsDic = sorted_dict_values(BgMCHistYieldsDic)
    for h in BgMCHistYieldsDic:
        hStack.Add(h)
        hStackBkg.Add(h)

    # add the signal histograms
    for i in range(nsig):
        h = histos[i]
        h.SetLineWidth(3)
        h.SetLineColor(colors[i])
        h.SetMarkerColor(colors[i])
        hStack.Add(h)
        hStackSig.Add(h)

    if xmin != -1 and xmax != -1:
        h_dummy.GetXaxis().SetLimits(xmin, xmax)


    h_dummy.Draw("HIST")
    if stacksig:
        hStack.Draw("HIST SAME")
        if plotStatUnc:
            # sig+bkg uncertainty
            hUnc_sig_bkg = formatStatUncHist(hStack.GetHists(), "sig_bkg")
            hUnc_sig_bkg.Draw("E2 SAME")
    else:
        hStackBkg.Draw("HIST SAME")
        hStackSig.Draw("HIST SAME NOSTACK")
        if plotStatUnc:
            # bkg-only uncertainty
            if hStackBkg.GetNhists() != 0:
                hUnc_bkg = formatStatUncHist(hStackBkg.GetHists(), "bkg_only")
                hUnc_bkg.Draw("E2 SAME")
            for sHist in hStackSig.GetHists():
                # sigs uncertainty
                hUnc_sig = formatStatUncHist([sHist], "sig", 3245)
                hUnc_sig.Draw("E2 SAME")

    # x limits
    if xmin == -1:
        h_tmp = hStack.GetStack().Last()
        xmin = h_tmp.GetBinLowEdge(1)
    if xmax == -1:
        h_tmp = hStack.GetStack().Last()
        xmax = h_tmp.GetBinLowEdge(h_tmp.GetNbinsX() + 1)

    h_dummy.GetXaxis().SetLimits(xmin, xmax)

    # if cutFlow table
    # if len(xtitle) > 1:
    #     print("set x axis differnetly")
    #     h_dummy.GetXaxis().SetLimits(-0.5, len(xtitle) - 0.5)





    # y limits
    def get_minmax_range(hists, xmin, xmax):
        hist_tot = hists[0].Clone(name + "_unc")
        for h in hists[1:]:
            hist_tot.Add(h)
        vals = []
        for i in range(0, hist_tot.GetNbinsX() + 1):
            if hist_tot.GetBinLowEdge(i) > xmin or hist_tot.GetBinLowEdge(i + 1) < xmax:
                if hist_tot.GetBinContent(i) != 0:
                    vals.append(hist_tot.GetBinContent(i))
        if len(vals) == 0:
            return 1e-5, 1
        return min(vals), max(vals)

    if stacksig:
        ymin_, ymax_ = get_minmax_range(hStack.GetHists(), xmin, xmax)
    else:
        if hStackSig.GetNhists() != 0 and hStackBkg.GetNhists() != 0:
            ymin_sig, ymax_sig = get_minmax_range(hStackSig.GetHists(), xmin, xmax)
            ymin_bkg, ymax_bkg = get_minmax_range(hStackBkg.GetHists(), xmin, xmax)
            ymin_ = min(ymin_sig, ymin_bkg)
            ymax_ = max(ymax_sig, ymax_bkg)
        elif hStackSig.GetNhists() == 0:
            ymin_, ymax_ = get_minmax_range(hStackBkg.GetHists(), xmin, xmax)
        elif hStackBkg.GetNhists() == 0:
            ymin_, ymax_ = get_minmax_range(hStackSig.GetHists(), xmin, xmax)
    if ymin == -1:
        ymin = ymin_ * 0.1 if logY else 0
    if ymax == -1:
        ymax = ymax_ * 1000.0 if logY else 1.4 * ymax_
    if ymin <= 0 and logY:
        LOGGER.error("Log scale can't start at: %i", ymin)
        sys.exit(3)
    h_dummy.SetMaximum(ymax)
    h_dummy.SetMinimum(ymin)

    legend.Draw()
    if legend2 is not None:
        legend2.Draw()

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextAlign(11)
    latex.SetTextSize(0.03)

    text = leftText
    latex.DrawLatex(0.15, 0.91, text)

    latex.SetTextAlign(31)
    latex.SetTextSize(0.03)

    text = "#bf{" + rightText + "}"
    latex.DrawLatex(0.90, 0.91, text)

    text = "#it{" + customLabel + "}"
    latex.SetTextAlign(12)
    latex.SetNDC(ROOT.kTRUE)
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.18, 0.85, text)

    rightText = re.split(",", rightText)
    text = "#bf{#it{" + rightText[0] + "}}"

    latex.SetTextAlign(12)
    latex.SetNDC(ROOT.kTRUE)
    latex.SetTextSize(0.04)
    # latex.DrawLatex(0.18, 0.81, text)

    rightText[1] = rightText[1].replace("   ", "")
    text = "#bf{#it{" + rightText[1] + "}}"
    latex.SetTextSize(0.035)
    # latex.DrawLatex(0.18, 0.76, text)

    text = "#bf{" + ana_tex + "}"
    latex.SetTextSize(0.035)
    latex.DrawLatex(0.20, 0.83, text)

    text = "#bf{#it{" + extralab + "}}"
    latex.SetTextSize(0.025)
    latex.DrawLatex(0.18, 0.66, text)

    if config["scale_sig"] != 1.0:
        text = "#bf{#it{Signal Scaling = " + f'{config["scale_sig"]:.3g}' + "}}"
        latex.SetTextSize(0.025)
        latex.DrawLatex(0.18, 0.63, text)

    if config["scale_bkg"] != 1.0:
        text = "#bf{#it{Background Scaling = " + f'{config["scale_bkg"]:.3g}' + "}}"
        latex.SetTextSize(0.025)
        latex.DrawLatex(0.18, 0.63, text)

    canvas.RedrawAxis()
    canvas.GetFrame().SetBorderSize(12)
    canvas.Modified()
    canvas.Update()

    if "AAAyields" in name:
        dummyh = ROOT.TH1F("", "", 1, 0, 1)
        dummyh.SetStats(0)
        dummyh.GetXaxis().SetLabelOffset(999)
        dummyh.GetXaxis().SetLabelSize(0)
        dummyh.GetYaxis().SetLabelOffset(999)
        dummyh.GetYaxis().SetLabelSize(0)
        dummyh.Draw("AH")
        legend.Draw()

        latex.SetNDC()
        latex.SetTextAlign(31)
        latex.SetTextSize(0.04)

        text = "#it{" + leftText + "}"
        latex.DrawLatex(0.90, 0.92, text)

        text = "#bf{#it{" + rightText[0] + "}}"
        latex.SetTextAlign(12)
        latex.SetNDC(ROOT.kTRUE)
        latex.SetTextSize(0.04)
        latex.DrawLatex(0.18, 0.83, text)

        text = "#bf{#it{" + rightText[1] + "}}"
        latex.SetTextSize(0.035)
        latex.DrawLatex(0.18, 0.78, text)

        text = "#bf{#it{" + ana_tex + "}}"
        latex.SetTextSize(0.04)
        latex.DrawLatex(0.18, 0.73, text)

        text = "#bf{#it{" + extralab + "}}"
        latex.SetTextSize(0.025)
        latex.DrawLatex(0.18, 0.68, text)

        text = "#bf{#it{Signal Scaling = " + f'{config["scale_sig"]:.3g}' + "}}"
        latex.SetTextSize(0.04)
        latex.DrawLatex(0.18, 0.57, text)

        text = "#bf{#it{Background Scaling = " + f'{config["scale_bkg"]:.3g}' + "}}"
        latex.SetTextSize(0.04)
        latex.DrawLatex(0.18, 0.52, text)

        dy = 0
        text = "#bf{#it{" + "Process" + "}}"
        latex.SetTextSize(0.035)
        latex.DrawLatex(0.18, 0.45, text)

        text = "#bf{#it{" + "Yields" + "}}"
        latex.SetTextSize(0.035)
        latex.DrawLatex(0.5, 0.45, text)

        text = "#bf{#it{" + "Raw MC" + "}}"
        latex.SetTextSize(0.035)
        latex.DrawLatex(0.75, 0.45, text)

        for y in yields:
            text = "#bf{#it{" + yields[y][0] + "}}"
            latex.SetTextSize(0.035)
            latex.DrawLatex(0.18, 0.4 - dy * 0.05, text)

            stry = str(yields[y][1])
            stry = stry.split(".", maxsplit=1)[0]
            text = "#bf{#it{" + stry + "}}"
            latex.SetTextSize(0.035)
            latex.DrawLatex(0.5, 0.4 - dy * 0.05, text)

            stry = str(yields[y][2])
            stry = stry.split(".", maxsplit=1)[0]
            text = "#bf{#it{" + stry + "}}"
            latex.SetTextSize(0.035)
            latex.DrawLatex(0.75, 0.4 - dy * 0.05, text)

            dy += 1
        # canvas.Modified()
        # canvas.Update()

    print_canvas(canvas, name, formats, directory)


# _____________________________________________________________________________
def print_canvas(canvas, name, formats, directory):
    """
    Saving canvas in multiple formats.
    """

    if not formats:
        LOGGER.error("No output formats specified!\nAborting...")
        sys.exit(3)

    if not os.path.exists(directory):
        os.system("mkdir -p " + directory)

    for f in formats:
        out_file = os.path.join(directory, name) + "." + f
        canvas.SaveAs(out_file)


# _____________________________________________________________________________
def run(args):
    """
    Run over all the plots.
    """
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kWarning

    module_path = os.path.abspath(args.script_path)
    module_dir = os.path.dirname(module_path)
    base_name = os.path.splitext(ntpath.basename(args.script_path))[0]

    # Load plot script as module
    sys.path.insert(0, module_dir)
    script_module = importlib.import_module(base_name)

    # Merge script and command line arguments into one configuration object
    # Also check the script attributes
    config: dict[str, any] = {}

    # Input directory
    config["input_dir"] = os.getcwd()
    if hasattr(script_module, "indir"):
        config["input_dir"] = script_module.indir
    if hasattr(script_module, "inputDir"):
        config["input_dir"] = script_module.inputDir
    if args.input_dir is not None:
        config["input_dir"] = args.input_dir

    # Output directory
    config["output_dir"] = os.getcwd()
    if hasattr(script_module, "outdir"):
        config["output_dir"] = script_module.outdir
    if hasattr(script_module, "outputDir"):
        config["output_dir"] = script_module.outputDir
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    # Integrated luminosity
    config["int_lumi"] = 1.0
    if hasattr(script_module, "intLumi"):
        config["int_lumi"] = script_module.intLumi
    else:
        LOGGER.debug("No integrated luminosity provided, using 1.0 pb-1.")
    LOGGER.info("Integrated luminosity: %g pb-1", config["int_lumi"])

    # Whether to scale histograms to luminosity
    config["do_scale"] = 1.0
    if hasattr(script_module, "doScale"):
        config["do_scale"] = script_module.doScale
    else:
        LOGGER.debug("No scaling to luminosity requested, scaling won't be " "done.")
        config["do_scale"] = False
    if config["do_scale"]:
        LOGGER.info("Histograms will be scaled to luminosity.")

    # Scale factor to apply to all signal histograms
    config["scale_sig"] = 1.0
    if hasattr(script_module, "scaleSig"):
        config["scale_sig"] = script_module.scaleSig
    else:
        LOGGER.debug("No scale factor for signal provided, using 1.0.")
    LOGGER.info("Scale factor for signal: %g", config["scale_sig"])

    # Scale factor to apply to all background histograms
    config["scale_bkg"] = 1.0
    if hasattr(script_module, "scaleBkg"):
        config["scale_bkg"] = script_module.scaleBkg
    else:
        LOGGER.debug("No scale factor for background provided, using 1.0.")
    LOGGER.info("Scale factor for background: %g", config["scale_sig"])

    # Check if we have plots (staged analysis) or histos (histmaker)
    config["plots"]: dict[str, any] = {}
    config["hists"]: dict[str, any] = {}
    config["ana_type"]: str = "none"
    if hasattr(script_module, "plots"):
        config["plots"] = script_module.plots
        config["ana_type"]: str = "staged"
    if hasattr(script_module, "hists"):
        config["hists"] = script_module.hists
        config["ana_type"]: str = "histmaker"

    if config["ana_type"] == "none":
        LOGGER.error("No plot definitions found!\nAborting...")
        sys.exit(3)

    # Splitting legend into two columns
    config["split_leg"] = False
    if hasattr(script_module, "splitLeg"):
        config["split_leg"] = script_module.splitLeg

    config["leg_position"] = [None, None, None, None]
    if hasattr(script_module, "legendCoord"):
        config["leg_position"] = script_module.legendCoord
    if args.legend_x_min is not None:
        config["leg_position"][0] = args.legend_x_min
    if args.legend_y_min is not None:
        config["leg_position"][1] = args.legend_y_min
    if args.legend_x_max is not None:
        config["leg_position"][2] = args.legend_x_max
    if args.legend_y_max is not None:
        config["leg_position"][3] = args.legend_y_max

    config["plot_stat_unc"] = False
    if hasattr(script_module, "plotStatUnc"):
        config["plot_stat_unc"] = script_module.plotStatUnc

    config["legend_text_size"] = 0.035
    if hasattr(script_module, "legendTextSize"):
        config["legend_text_size"] = script_module.legendTextSize
    if args.legend_text_size is not None:
        config["legend_text_size"] = args.legend_text_size

    # Label for the integrated luminosity
    config["int_lumi_label"] = None
    if hasattr(script_module, "intLumiLabel"):
        config["int_lumi_label"] = script_module.intLumiLabel
    if config["int_lumi_label"] is None:
        if config["int_lumi"] >= 1e6:
            int_lumi_label = config["int_lumi"] / 1e6
            config["int_lumi_label"] = f"L = {int_lumi_label:.2g} ab^{{-1}}"
        elif config["int_lumi"] >= 1e3:
            int_lumi_label = config["int_lumi"] / 1e3
            config["int_lumi_label"] = f"L = {int_lumi_label:.2g} fb^{{-1}}"
        else:
            config["int_lumi_label"] = f'L = {config["int_lumi"]:.2g} pb^{{-1}}'

    # Handle plots for the Histmaker analyses and exit
    if config["ana_type"] == "histmaker":
        LOGGER.info("Plotting histograms from histmaker step...")
        for hist_name, hist_cfg in script_module.hists.items():
            runPlotsHistmaker(config, args, hist_cfg["input"], script_module, hist_cfg)
        # check if there are 2D histograms to plot
        try:
            for hist_name, hist_cfg in script_module.hists2D.items():
                runPlotsHistmaker2D(config, args, hist_cfg["input"], script_module, hist_cfg)
        except AttributeError:
            pass
        sys.exit()

    counter = 0
    LOGGER.info("Plotting staged analysis plots...")
    for var_index, var in enumerate(script_module.variables):
        for label, sels in script_module.selections.items():
            for sel in sels:
                rebin_tmp = 1
                if hasattr(script_module, "rebin"):
                    if len(script_module.rebin) == len(script_module.variables):
                        rebin_tmp = script_module.rebin[var_index]

                LOGGER.info("  var: %s     label: %s     selection: %s", var, label, sel)

                hsignal, hbackgrounds = load_hists(var, label, sel, config, rebin=rebin_tmp)
                runPlots(
                    config,
                    args,
                    var + "_" + label,
                    sel,
                    script_module,
                    hsignal,
                    hbackgrounds,
                    script_module.extralabel[sel],
                )
                if counter == 0:
                    runPlots(
                        config,
                        args,
                        "AAAyields_" + label,
                        sel,
                        script_module,
                        hsignal,
                        hbackgrounds,
                        script_module.extralabel[sel],
                    )
        counter += 1


def do_plots(parser):
    """
    Run plots generation
    """

    args, _ = parser.parse_known_args()

    if args.command != "plots":
        LOGGER.error("Wrong sub-command!\nAborting...")

    if not os.path.isfile(args.script_path):
        LOGGER.error('Plots script "%s" not found!\nAborting...', args.script_path)
        sys.exit(3)

    run(args)