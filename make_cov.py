import ROOT as r
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mplhep as hep
#import root_numpy as rnp
import sys
import uproot 
import os 
# Define the patch function
def _matrix_mask(data, mask=None):
    """Ensure that data and mask are compatible and add the mask"""
    import numpy as np

    if mask is None:
        mask = np.zeros(data.shape, bool)
    elif np.array(mask, bool).shape != data.shape:
        raise ValueError("Mask must have the same shape as data.")
    mask = np.asarray(mask)
    if mask.dtype.kind != "b":
        mask = mask.astype(bool)
    return mask

# Monkey-patch seaborn's _matrix_mask function
sns.matrix._matrix_mask = _matrix_mask

def plot_cov(fitDiagnostics_file="fitDiagnostics.root", fit="fit_s", out="covariance_matrix.png", include=None, data=False, year=2017):
    assert include in [None, "all", "tf"]
    rf = r.TFile.Open(fitDiagnostics_file)
    h2 = rf.Get("fit_s").correlationHist()
    TH2 = uproot.open(fitDiagnostics_file)[f"covariance_{fit}"].values()
    print(TH2)
    #TH2 = uproot.open()
    #TH2 = rnp.hist2array(h2)

    labs = []
    for i in range(h2.GetXaxis().GetNbins() + 2):
        lab = h2.GetXaxis().GetBinLabel(i)
        labs.append(lab)
    labs = labs[1:-1]  # Remove over/under flows

    if include == "all":
        sel_labs = [lab for lab in labs]
    elif include == "tf":
        sel_labs = [lab for lab in labs if not (lab.startswith("qcdparam") or "mcstat" in lab)]
    else:
        sel_labs = [lab for lab in labs if not (lab.startswith("qcdparam") or "mcstat" in lab or lab.startswith("tf"))]
    sel_ixes = [labs.index(lab) for lab in sel_labs]

    # Get only values we want
    def extract(arr2d, ix):
        x, y = np.meshgrid(ix, ix)
        return arr2d[x, y]

    cov_mat = extract(np.flip(TH2, axis=1), sel_ixes)

    # Plot it
    fig, ax = plt.subplots(figsize=(12, 10))
    g = sns.heatmap(cov_mat, xticklabels=sel_labs, yticklabels=sel_labs, cmap="RdBu", vmin=-1, vmax=1, ax=ax)
    hep.cms.label(fontsize=23, year=year, data=data)
    g.set_xticklabels(g.get_xticklabels(), rotation=30, horizontalalignment="right")
    g.set_yticklabels(g.get_yticklabels(), rotation=30, horizontalalignment="right")
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    plt.minorticks_off()

    opath = os.path.dirname(fitDiagnostics_file)
    fig.savefig(opath+f"/covariance_{fit}.png", bbox_inches="tight")

plot_cov(fitDiagnostics_file=sys.argv[1],fit=sys.argv[2],year=sys.argv[3])
