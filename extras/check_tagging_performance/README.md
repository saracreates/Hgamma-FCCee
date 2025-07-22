# Checking the tagging performance

We want to check the tagging performance of a neural network trained on H(jj)Z($\nu \nu$) data evaluated on $H(jj) \gamma$. 

## $H \gamma$ data

Source the FCCAnalyses pre-edm4hep1 branch and run 
```
fccanalysis run extras/check_tagging_performance/treemaker_jettags.py
```

Then, create a flat tree (jet-based) with 

```
python3 extras/check_tagging_performance/from_eventbased_2_jetbased.py outputs/jettags/Hgamma/p8_ee_Hgamma_ecm240.root outputs/jettags/Hgamma/p8_ee_Hgamma_ecm240_jetbased.root 0 50000
```

and then the ROC curve with
```
python3 extras/check_tagging_performance/rocs.py
``` 

