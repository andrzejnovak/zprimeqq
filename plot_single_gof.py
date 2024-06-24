import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np
import ROOT as r
import mplhep as hep
import argparse 

plt.style.use(hep.style.ROOT)

def get_vals(fname):
    rfile = r.TFile.Open(fname)
    rtree = rfile.Get("limit")
    vals = []
    for i in range(rtree.GetEntries()):
        rtree.GetEntry(i)
        mu = rtree.limit
        vals.append(mu)
    return vals

def gofplot(datafile, mcfile, year=2017, savename='fplotX', nbins=130, algo='saturated',opath="."):
    gofs = np.array(get_vals(mcfile))
    gof_data = get_vals(datafile)[0]
    
    print("XXXXXXX")
    print(gof_data)
    print(np.array([np.around(np.mean(gofs) + x * np.std(gofs), 3) for x in [-3,-2,-1,0,1,2,3]]))
    print("XXXXXXX")

    from scipy.stats import chi2
    x_lim = np.max(gofs) * 1.2
    x_low = np.min(gofs) * 0.9
    x = np.linspace(x_low, x_lim, 200)
    bins = np.linspace(0, x_lim, 50)
    width = bins[1] - bins[0]

    fig, ax = plt.subplots()
    trans = ax.get_xaxis_transform()
    #trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    if algo == 'saturated':
        ax.plot(x, len(gofs) * width * chi2.pdf(x, np.mean(gofs)), color='red', label='Toy average GOF = {:.2f}'.format(np.mean(gofs)))
    h, _, _ = ax.hist(gofs, bins, facecolor='none', edgecolor='black', histtype='stepfilled', lw=2,
            label="Toys, N = {}".format(len(gofs)))
    ax.hist(gofs[gofs > gof_data], bins, facecolor='steelblue', edgecolor='gray', histtype='stepfilled', alpha=0.3,
            label='p-value = {}'.format(round(float(len(gofs[gofs > gof_data]))/len(gofs), 3)));
    print("P-value", round(float(len(gofs[gofs > gof_data]))/len(gofs), 3))
    print("gof_data",gof_data)
    ax.annotate("", xy=(gof_data, 0), xycoords=trans,
                xytext=(gof_data, 0.25), textcoords=trans,
                arrowprops=dict(lw=4, color='b', arrowstyle="->,head_length=1.5,head_width=0.5"),
                )
    ax.plot([], [], color='blue', lw=2, label="Observed = {:.2f}".format(gof_data))

    ax.legend()
    hep.cms.label(llabel='Private Work', data=True, year=year, ax=ax)
    ax.set_xlim(np.mean(gofs)-np.std(gofs)*4, np.mean(gofs)+np.std(gofs)*5)
    ax.set_ylim(0, max(h) * 1.4)
    if algo == 'saturated':
        xlab = r"$-2log(\lambda)$"
    else:
        xlab = "KS"
    ax.set_xlabel(xlab , x=1, ha='right')
    ax.set_ylabel("Pseudoexperiments", y=1, ha='right')
    plt.savefig('{}/{}.pdf'.format(opath,savename), dpi=300, transparent=True, bbox_inches='tight')
    plt.savefig('{}/{}.png'.format(opath,savename), dpi=300, transparent=True, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make plots for ftest/gof files.')
    parser.add_argument('datagof')
    parser.add_argument('mcgofs')
    parser.add_argument("--year", choices=['2016', '2017', '2018', "Run2"], default="")
    parser.add_argument("--algo", choices=['saturated', 'KS'], default="saturated")
    parser.add_argument("--opath", default="test")
    args = parser.parse_args()

sname = "gof_" + args.year + "_" + args.algo
gofplot(args.datagof, args.mcgofs, year=args.year, savename=sname, algo=args.algo, opath=args.opath)



