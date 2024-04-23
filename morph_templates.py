import ROOT as r

from scipy.interpolate import interp1d
import numpy as np
import hist
import argparse
import uproot
import scipy.stats
parser = argparse.ArgumentParser(description="Rhalphalib setup.")
parser.add_argument(
    "--template_file", action="store", type=str, required=True, help="Path to template file ."
)

parser.add_argument(
    "--year", action="store", type=str, required=True, help="Path to template file ."
)

args = parser.parse_args()

f = uproot.open(args.template_file) 

class AffineMorphTemplate(object):
    def __init__(self, hist):
        """
        hist: a numpy-histogram-like tuple of (sumw, edges)
        """
        self.sumw = hist[0]
        self.edges = hist[1]
        self.varname = hist[2]
        self.centers = self.edges[:-1] + np.diff(self.edges) / 2
        self.norm = self.sumw.sum()
        self.mean = (self.sumw * self.centers).sum() / self.norm
        self.cdf = interp1d(
            x=self.edges,
            y=np.r_[0, np.cumsum(self.sumw / self.norm)],
            kind="linear",
            assume_sorted=True,
            bounds_error=False,
            fill_value=(0, 1),
        )

    def get(self, shift=0.0, smear=1.0):
        """
        Return a shifted and smeard histogram
        i.e. new edges = edges * smear + shift
        """
        if not np.isclose(smear, 1.0):
            shift += self.mean * (1 - smear)
        smeard_edges = (self.edges - shift) / smear
        # print(smeard_edges)
        return np.diff(self.cdf(smeard_edges)) * self.norm, self.edges, self.varname

def moment_morph(hists, params, param_interp, components=False, vertical_only=False):
    # print(params, param_interp)
    '''
    Pythonic implementation of Moment morphing for `hist` histograms (incl variances): 
        - http://cds.cern.ch/record/1958015/files/arXiv:1410.7388.pdf 
        - https://arxiv.org/abs/1410.7388
    '''
    dists_nom = [scipy.stats.rv_histogram((h.values(), h.axes[0].edges), density=True) for h in hists]
    dists_var = [scipy.stats.rv_histogram((h.variances(), h.axes[0].edges), density=True) for h in hists]
    out_nom = []
    out_var = []
    # do twice - for nominals and variances 
    for dists, out in zip([dists_nom, dists_var], [out_nom, out_var]):
        n = len(params)
        M = np.empty((n,n))
        for i in range(0, n):
            for j in range(0, n):
                elem = (params[i]-params[0])**j
                M[i, j] = elem
        Minv = np.linalg.inv(M)
        # Minv = abs(Minv)
        cis = []
        for i in range(0, n):
            ci = 0
            for j in range(0, n):
                ci += (param_interp - params[0])**j * Minv[j, i]
            cis.append(ci)
            
        means = [dist.moment(1) for dist in dists]
        sigs = [np.sqrt(dist.stats(moments='v')/2) for dist in dists]
        mup = np.sum([c * mu for c, mu in zip(cis, means)])
        sigp = np.sum([c * sig for c, sig in zip(cis, sigs)])
        
        xps, slopes, offsets = [], [], []
        for i in range(0, n):
            xj = dists[i]._hbins
            aij = sigs[i]/sigp
            bij = means[i] - aij*mup
            xp = xj*aij + bij
            slopes.append(aij)
            offsets.append(bij)
        
            xps.append(xp)
        # print(cis)
        for i, (ci, dist, xp) in enumerate(zip(cis, dists, xps)):
            centers = xp[:-1] + np.diff(xp)/2
            # For rescale
            binwidths = hists[i].axes[0].widths
            norm = dist._histogram[0].sum()
            if vertical_only:
                out.append(ci *  dist.pdf(hists[i].axes[0].centers) * norm * binwidths)
            else:
                out.append(ci *  dist.pdf(centers) * norm * binwidths)

    # Reconstruct hists
    if not components:
        rh = hists[0].copy()
        rh.view().value = np.sum(out_nom, axis=0)
        rh.view().variance = np.sum(out_var, axis=0)
        try:  # If param value in axis name, rename
            rh.axes[0].label = rh.axes[0].label.replace(str(params[0]), str(param_interp))
        except:
            pass
        return rh
    else:
        to_return = []
        for i in range(len(hists)):
            rh = hists[i].copy()
            rh.view().value = out_nom[i]
            rh.view().variance = out_var[i]
            to_return.append(rh)
        return to_return   

odict = {}
params = [50, 75, 100, 125, 150, 200, 250]
for smp in ["zpqq", "zpbb"]:
    for ptbin in [0, 1, 2, 3, 4]:
        #for region in ["fail", "pass_lowbvl", "pass_highbvl"]:
        for region in ["fail_T","pass_T_bvl_fail_L","pass_T_bvl_pass_L","pass_T_bvl_fail_T","pass_T_bvl_pass_T","pass_T_bvl_fail_VT","pass_T_bvl_pass_VT"]:
    
            for interp in np.arange(50, 255, 5):
                nearest = sorted([params[np.argsort(np.abs(params-interp))[0]], params[np.argsort(np.abs(params-interp))[1]]])
                mass_hists = []
                # for param in params:
                # print(nearest)
                for param in nearest:
                    mass_hists.append(f[f"SR_{smp}{param}_ptbin{ptbin}_pnmd2prong_{region}"].to_hist())
                out = moment_morph(mass_hists, params=nearest, param_interp=interp)
                odict[f"SR_{smp}{param}_ptbin{ptbin}_pnmd2prong_{region}"] = out

                for syst in sys_names:
                    if "muo" in syst: continue
                    if "HEM" in syst and args.year != "2018" : continue
                    if "L1Pre" in syst and args.year == "2018" : continue
                    #print(isamp in ["zqq","dy"])
                    if syst in ['W_d2kappa_EW', 'W_d3kappa_EW'] and not isamp in ["wqq","wlnu"]: continue
                    if syst in ['Z_d2kappa_EW', 'Z_d3kappa_EW'] and not isamp in ["zqq","dy"]: continue
                    if syst in ['d1kappa_EW','d1K_NLO','d2K_NLO','d3K_NLO'] and isamp not in ["wqq","wlnu","zqq","dy",]: continue
                    syst_name_up = sys_name_updown[syst][0]
                    syst_name_down = sys_name_updown[syst][1]
    
                    if "year" in syst_name_up:
                        syst_name_up = syst_name_up.replace('year',args.year)
                        syst_name_down = syst_name_down.replace('year',args.year)
                    for param in nearest:
                        mass_hists.append(f[f"SR_{smp}{param}_ptbin{ptbin}_pnmd2prong_{region}__{syst}"].to_hist())
    
                    # out = moment_morph(mass_hists, params=params, param_interp=interp)
                    out = moment_morph(mass_hists, params=nearest, param_interp=interp)
                    odict[f"SR_{smp}{param}_ptbin{ptbin}_pnmd2prong_{region}__{syst}"] = out
    
                    # break
    
root_file = uproot.recreate(args.template_file.replace(".root","_interpolated.root"))
for k, v in odict.items():
    if k not in f.keys():
        root_file[k] = v
    else:
        print("Skipping", k)
root_file.close()

