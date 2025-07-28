# Boosted Decision Tree for $H(WW) \rightarrow \ell \nu q q$

The last background left after the cut-and-count approach is WW. We have a lot of statistics for this process, so we train a BDT to distinguish between the signal HWW and WW. We train the binary decision tree with:
- 1800 k events of WW -> After preselection cuts we have 0.2% of the data left: 360k
- 400 k events of HWW

We choose following preselection cuts: 
- isolated photon
- momentum of the photon
- $\cos(\theta)$ of the photon
- 1 isolated lepton
- $m_jj$ within $W^*$ range

After these selction cuts
- 14 % of the HWW data is left 
- 0.017 % of the WW data is left

The max data I want to train on, is 50% of the whole datasets available so there is statistics left for the inference. 
- HWW: 1mio /2 = 500 k -> 70 k after selections
- WW: 374 mio /2 = 187 k -> 32k after selections

For a balanced dataset, we use the same amount of data for each class. So we use
- 1/4 of the HWW data for training
- 1/2 of the WW data for training

And train on ~30-35k events per class. 

## Commands to train the BDT: 

```
# create training data
fccanalysis run analysis/exclusive_HWW/BDT/preselection.py

# train bdt
python3 analysis/exclusive_HWW/BDT/train_bdt.py

# evaluate the bdt
python3 analysis/exclusive_HWW/BDT/evaluate_bdt.py
```

Now you can use the BDT score in the histmaker. 

