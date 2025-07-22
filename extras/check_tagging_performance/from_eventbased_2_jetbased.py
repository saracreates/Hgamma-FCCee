import sys
from array import array
from ROOT import TFile, TTree
from examples.FCCee.weaver.config import variables_pfcand, variables_jet, flavors

debug = False 

if len(sys.argv) < 2:
    print(" Usage: from_eventbased_2_jetbased.py input_file output_file n_start n_events")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]
n_start = int(sys.argv[3])
n_final = int(sys.argv[4])
n_events = n_final - n_start

# Opening the input file containing the tree (output of stage1.py)
infile = TFile.Open(input_file, "READ")

ev = infile.Get("events")
numberOfEntries = ev.GetEntries()
if debug:
    print("-> number of entries in input file: {}".format(numberOfEntries))

## basic checks
if n_final > n_start + numberOfEntries:
    print("ERROR: requesting too many events. This file only has {}".format(numberOfEntries))
    sys.exit()

## define variables for output tree
maxn = 500

## output jet-wise tree
out_root = TFile(output_file, "RECREATE")
t = TTree("tree", "tree with jets")

jet_array = dict()

flavors = ["U", "D", "S", "B", "C", "G", "TAU"]
for f in flavors:
    b1 = "recojet_is{}".format(f.upper())
    b2 = "score_recojet_is{}".format(f.upper())
    jet_array[b1] = array("i", [0])
    jet_array[b2] = array("f", [0.0])
    t.Branch(b1, jet_array[b1], "{}/I".format(b1))
    t.Branch(b2, jet_array[b2], "{}/F".format(b2))

# Loop over all events
for entry in range(n_start, n_final):
    # Load selected branches with data from specified event

    ev.GetEntry(entry)

    ## loop over jets
    njets = len(getattr(ev, "score_recojet_isU"))
    if debug:
        print("-> processing event {} with {} jets".format(entry, njets))
    for j in range(njets):

        ## fill jet-based quantities
        for f in flavors:
            name = "recojet_is{}".format(f.upper())
            jet_array[name][0] = getattr(ev, name)
            jet_array["score_" + name][0] = float(getattr(ev, "score_" + name)[j])
            if debug:
                print("   jet:", j, name, jet_array[name][0])
                print("   jet:", j, "score_" + name, jet_array["score_" + name][0])

        ## fill tree at every jet
        t.Fill()

# write tree
t.SetDirectory(out_root)
t.Write()