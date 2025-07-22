# $H\gamma$ analysis at FCC-ee

We look at $e^+ e^- \rightarrow H \gamma$ at 160 GeV, 240 GeV and 365 GeV at FCC-ee. 

## Installation

To run the analysis, we need to set-up the [FCCAnalyses](https://github.com/HEP-FCC/FCCAnalyses/tree/pre-edm4hep1) framework with the `pre-edm4hep1` branch.  

```
git clone --branch pre-edm4hep1 git@github.com:HEP-FCC/FCCAnalyses.git
cd FCCAnalyses
source ./setup.sh
fccanalysis build -j 8
```

## Example usage

After sourcing FCCAnalyses, run the analyses with 

```
fccanalysis run analysis/histmaker_inclusive.py
```

and create the plots with 
```
export PYTHONPATH=/afs/cern.ch/work/s/saaumill/public/MyFCCAnalyses/extras:$PYTHONPATH
export PATH=/cvmfs/sft.cern.ch/lcg/external/texlive/2020/bin/x86_64-linux:$PATH
fccanalysis plots analysis/plots_inclusive.py
```
