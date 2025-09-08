# $H\gamma$ analysis at FCC-ee

We look at $e^+ e^- \rightarrow H \gamma$ at 160 GeV, 240 GeV and 365 GeV at FCC-ee and investigate following final state:
- $H\gamma$: inclusive
- $H(jj) \gamma$: $H \rightarrow bb$ and $gg$ and $\tau \tau$
- $H(WW) \gamma$: with $W \rightarrow qq$ and $W \rightarrow \ell \nu$

## Installation

To run the analysis, we need to set-up the [FCCAnalyses](https://github.com/HEP-FCC/FCCAnalyses/tree/pre-edm4hep1) framework with the `pre-edm4hep1` branch.  

```
git clone --branch pre-edm4hep1 git@github.com:HEP-FCC/FCCAnalyses.git
cd FCCAnalyses
source ./setup.sh
fccanalysis build -j 8
```

## Usage

After sourcing FCCAnalyses, run the analyses with 

### Inclusive study

```
fccanalysis run analysis/inclusive/histmaker_inclusive.py
```

and create the plots with 
```
export PYTHONPATH=/afs/cern.ch/work/s/saaumill/public/MyFCCAnalyses/extras:$PYTHONPATH
export PATH=/cvmfs/sft.cern.ch/lcg/external/texlive/2020/bin/x86_64-linux:$PATH
fccanalysis plots analysis/inclusive/plots_inclusive.py
```

### $H \rightarrow WW$

```
fccanalysis run analysis/exclusive_HWW/histmaker_HWW_lvqq.py  
export PYTHONPATH=/afs/cern.ch/work/s/saaumill/public/MyFCCAnalyses/extras:$PYTHONPATH
export PATH=/cvmfs/sft.cern.ch/lcg/external/texlive/2020/bin/x86_64-linux:$PATH
fccanalysis plots analysis/exclusive_HWW/plots_HWW_lvqq.py 
```

Choose `lvqq` for $W(\ell \nu) W*(qq)$ or `qqlv` for $W(qq)W(\ell \nu)$.

### $H \righarrow jj$

```
fccanalysis run analysis/exclusive_Hjj/histmaker_Hjj.py --flavor B
export PYTHONPATH=/afs/cern.ch/work/s/saaumill/public/analyses/Hgamma-FCCee/extras:$PYTHONPATH
export PATH=/cvmfs/sft.cern.ch/lcg/external/texlive/2020/bin/x86_64-linux:$PATH
fccanalysis plots analysis/exclusive_Hjj/plots_Hjj.py --flavor B
```

And choose a flavor from $b,g$. For $\tau$ there are files with `_Htautau`. 
