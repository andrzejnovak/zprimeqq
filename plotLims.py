import ROOT
from optparse import OptionParser
from operator import add
import math
import sys
import time
from array import array
import os
import glob
import argparse
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as mticker
from matplotlib.ticker import LogLocator, FuncFormatter


#hep.style.use("CMS") # string aliases work too
plt.style.use(hep.style.CMS)
parser = argparse.ArgumentParser(description='Rhalphalib setup.')
parser.add_argument("--ipath", action='store', type=str, required=True, help="base path to read input.")
parser.add_argument('--observed', dest='observed', action='store_true',default=False, help='show observed')
parser.add_argument('--gq', dest='gq', action='store_true',default=False, help='gq')
parser.add_argument('--asimov', dest='asimov', action='store_true',default=False, help='limits on asimov dataset')
parser.add_argument('--lumi', dest='lumi', action='store',type=float,help='luminosity')
parser.add_argument('--year', dest='year', action='store',type=str,help='luminosity')
parser.add_argument('--rb', dest='rb', action='store_true',help='b-limit')

args = parser.parse_args()

masses = [55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,160,165,170,175,180,185,190,195,200]

def format_func(value, tick_number):
    # format the tick as plain, not scientific
    return f"{value:.0f}" if value > 1 else f"{value:.2f}"



def setDict():
	sample_xsec = {} # cross section used to normalize sample (in combine)
	theory_xsec = {} # real theory cross section to compare (no BR)
	theory_inclusive_xsec = {} # inclusive theory cross section to compare (no BR)
	br = {} # branching ratio to qq
	legend_entry = {}
	legend_entry['Zpqq'] = "Z', g_{q}=0.25, H_{T}>500 GeV"

	# == 1 since samples already re-weighted
	theory_xsec['Zpqq'] = ROOT.TGraph(16) # HT > 500
	theory_xsec['Zpqq'].SetPoint(0,  50, 1.466e+01)
	theory_xsec['Zpqq'].SetPoint(1,  75, 1.521e+01)
	theory_xsec['Zpqq'].SetPoint(2, 100, 1.433e+01)
	theory_xsec['Zpqq'].SetPoint(3, 115, 1.505e+01)
	theory_xsec['Zpqq'].SetPoint(4, 125, 1.456e+01)
	theory_xsec['Zpqq'].SetPoint(5, 150, 1.444e+01)
	theory_xsec['Zpqq'].SetPoint(6, 175, 1.362e+01)
	theory_xsec['Zpqq'].SetPoint(7, 200, 1.338e+01)
	theory_xsec['Zpqq'].SetPoint(8, 225, 1.370e+01)
	theory_xsec['Zpqq'].SetPoint(9, 250, 1.278e+01)
	theory_xsec['Zpqq'].SetPoint(10,275, 1.423e+01)
	theory_xsec['Zpqq'].SetPoint(11,300, 1.398e+01)
	theory_xsec['Zpqq'].SetPoint(12,350, 1.511e+01)
	theory_xsec['Zpqq'].SetPoint(13,400, 1.625e+01)
	theory_xsec['Zpqq'].SetPoint(14,450, 1.838e+01)
	theory_xsec['Zpqq'].SetPoint(15,500, 3.120e+01)

	sample_xsec['Zpqq'] = theory_xsec['Zpqq']


	return theory_xsec, sample_xsec

def read_limits(f):
    #print(f)
    tfile = ROOT.TFile(f)
    ttree = tfile.Get("limit")
    limits = []
    limit = array('d', [0])  # 'd' for double, initializes with one element set to 0
    ttree.SetBranchAddress("limit",limit)
    for entry in range(ttree.GetEntries()):
        ttree.GetEntry(entry)
        limits.append(limit[0])
    print(f,limits)
    return limits

class SimpleFormatter(mticker.LogFormatterSciNotation):
    def __call__(self, value, pos=None): 
        print("Hello")
        if value != 0:
            return f"{value:.2f}"
        else:
            return "0"

def non_scientific(y, pos):
    """ Custom formatter to replace scientific notation """
    return f'{y:.2f}'


def get_graphs():
    theory_xsec, sample_xsec = setDict()
    lo2 = []
    lo1 = []
    med = []
    hi1 = []
    hi2 = []
    obs = [] 
    for mass in masses:
        if args.rb:
            limits = read_limits(f"{args.ipath}/m{mass}/m{mass}_model/higgsCombiner_b.AsymptoticLimits.mH120.root")
        else:
            limits = read_limits(f"{args.ipath}/m{mass}/m{mass}_model/higgsCombine.Test.AsymptoticLimits.mH120.root")
        fac = sample_xsec['Zpqq'].Eval(mass,0,'S')
        theory = theory_xsec['Zpqq'].Eval(mass,0,'S')
        fac,theory=1.,1.
        theory = theory * 4. * 4.
        lo2.append(math.sqrt(limits[0]*fac/theory))
        lo1.append(math.sqrt(limits[1]*fac/theory))
        med.append(math.sqrt(limits[2]*fac/theory))
        hi1.append(math.sqrt(limits[3]*fac/theory))
        hi2.append(math.sqrt(limits[4]*fac/theory))
        if args.observed:
            assert(len(limits)>5)
            obs.append(math.sqrt(limits[5]*fac/theory))
    fig,ax = plt.subplots(figsize=(8,8))
    ax.set_yscale("log")    
    hep.cms.label(year=args.year,data=False if args.asimov else True,lumi=args.lumi,fontsize=17)
    #ax.yaxis.set_major_formatter(SimpleFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(non_scientific))
    ax.yaxis.set_minor_formatter(FuncFormatter(non_scientific))
    #ax.yaxis.set_major_locator(plt.MaxNLocator(2))
    #ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
    #ax.yaxis.set_minor_locator(LogLocator(base=10, numticks=5))
    ax.set_yticks([0.04,0.05,0.07,0.1,0.14,0.2,0.5,1.0],[0.04,0.05,0.07,0.1,0.14,0.2,0.5,1.0])
    ax.fill_between(masses,lo2,hi2,color='gold',label="Expected $\pm 2 \sigma$")
    ax.fill_between(masses,lo1,hi1,color='limegreen',label="Expected $\pm 1 \sigma$")
    ax.plot(masses,med,color='black',linestyle="--",linewidth=1.5,label="Expected")
    if args.observed:
        ax.plot(masses,obs,color='black',linestyle="-",linewidth=1.5,label="Observed")
    if args.rb:
        ax.set_ylabel("$g'_{b}$")
    else:
        ax.set_ylabel("$g'_{q}$")
    ax.set_xlabel("Z' mass (GeV)")
    ax.set_ylim(min(lo2)*0.7,max(hi2)*2)
    ax.set_xlim(min(masses),max(masses)) 
    
    plt.tight_layout()
    plt.legend(loc="upper left",fontsize=15)
    #plt.savefig(f"{args.ipath}/limits.png")
    #plt.savefig(f"{args.ipath}/limits.pdf")
    plt.savefig("{ipath}/limits_logy_{r_b}.png".format(ipath=args.ipath,r_b="r_b" if args.rb else "r"))
    plt.savefig("{ipath}/limits_logy_{r_b}.pdf".format(ipath=args.ipath,r_b="r_b" if args.rb else "r"))
get_graphs() 
