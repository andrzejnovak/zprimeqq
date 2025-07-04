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
from matplotlib.ticker import LogLocator, MultipleLocator, FuncFormatter
import numpy as np
from matplotlib.ticker import FormatStrFormatter

#hep.style.use("CMS") # string aliases work too
plt.style.use(hep.style.CMS)
parser = argparse.ArgumentParser(description='Rhalphalib setup.')
parser.add_argument("--ipath", action='store', type=str, required=True, help="base path to read input.")
parser.add_argument('--observed', dest='observed', action='store_true',default=False, help='show observed')
parser.add_argument('--gq', dest='gq', action='store_true',default=False, help='Plot gq')
parser.add_argument('--xsec', dest='xsec', action='store_true',default=False, help='Plot xsec')
parser.add_argument('--asimov', dest='asimov', action='store_true',default=False, help='limits on asimov dataset')
parser.add_argument('--lumi', dest='lumi', action='store',type=float,help='luminosity')
parser.add_argument('--year', dest='year', action='store',type=str,help='luminosity')
parser.add_argument('--rb', dest='rb', action='store_true',help='b-limit')
parser.add_argument('--rp', dest='rp', action='store_true',help='phi-limit')
parser.add_argument('--r', dest='r', action='store_true',help='zprime-limit')

args = parser.parse_args()
masses = np.arange(50,305,5)#[50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,160,165,170,175,180,185,190,195,200,205,210,215,220,225,230,235,240,245,250,]

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

        theory_inclusive_xsec['Zpqq'] =  ROOT.TGraph(10)
        theory_inclusive_xsec['Zpqq'].SetPoint(0,  50, 1.394e+05)
        theory_inclusive_xsec['Zpqq'].SetPoint(1,  60, 8.419e+04)
        theory_inclusive_xsec['Zpqq'].SetPoint(2,  75, 4.481e+04) 
        theory_inclusive_xsec['Zpqq'].SetPoint(3,  90, 2.641e+04)
        theory_inclusive_xsec['Zpqq'].SetPoint(4, 100, 1.939e+04)
        theory_inclusive_xsec['Zpqq'].SetPoint(5, 110, 1.462e+04)
        theory_inclusive_xsec['Zpqq'].SetPoint(6, 135, 7870)
        theory_inclusive_xsec['Zpqq'].SetPoint(7, 150, 5707)
        theory_inclusive_xsec['Zpqq'].SetPoint(8, 165, 4254)
        theory_inclusive_xsec['Zpqq'].SetPoint(9, 180, 3233)
        theory_inclusive_xsec['Zpqq'].SetPoint(10,200, 2320)
        theory_inclusive_xsec['Zpqq'].SetPoint(11,250, 1131)
        theory_inclusive_xsec['Zpqq'].SetPoint(12,300, 620)


        return theory_xsec, sample_xsec, theory_inclusive_xsec

def read_limits(f):
    tfile = ROOT.TFile(f)
    ttree = tfile.Get("limit")
    limits = []
    limit = array('d', [0])  # 'd' for double, initializes with one element set to 0
    ttree.SetBranchAddress("limit",limit)
    for entry in range(ttree.GetEntries()):
        ttree.GetEntry(entry)
        limits.append(limit[0])
    return limits

class SimpleFormatter(mticker.LogFormatterSciNotation):
    def __call__(self, value, pos=None): 
        if value != 0:
            return f"{value:.2f}"
        else:
            return "0"

def non_scientific(y, pos, ndigits=2):
    """ Custom formatter to replace scientific notation """
    return f'{y:.{ndigits}f}'.format(ndigits=ndigits)


def get_graphs():
    theory_xsec, sample_xsec, theory_inclusive_xsec = setDict()
    lo2 = []
    lo1 = []
    med = []
    hi1 = []
    hi2 = []
    obs = [] 
    kFactor = 1.218
    filled_masses = [] 
    for mass in masses:
      try:
        if args.rb:
            limits = read_limits(f"{args.ipath}/m{mass}/m{mass}_model/higgsCombiner_b.AsymptoticLimits.mH120.root")
        if args.r:
            limits = read_limits(f"{args.ipath}/m{mass}/m{mass}_model/higgsCombiner.AsymptoticLimits.mH120.root")
        elif args.rp:
            limits = read_limits(f"{args.ipath}/m{mass}/m{mass}_model/higgsCombiner_p.AsymptoticLimits.mH120.root")
        fac = sample_xsec['Zpqq'].Eval(mass,0,'S')
        theory = theory_xsec['Zpqq'].Eval(mass,0,'S')
        theoryRatio = theory_inclusive_xsec['Zpqq'].Eval(mass,0,'S') / theory_xsec['Zpqq'].Eval(mass,0,'S')
        assert(len(limits)>1)
        if args.gq:
            if args.r:
                factor = 1. /4. / 4.
            elif args.rp:
                factor = 1. #* 4. * 4
            lo2.append(math.sqrt(limits[0] * factor))
            lo1.append(math.sqrt(limits[1] * factor))
            med.append(math.sqrt(limits[2] * factor))
            hi1.append(math.sqrt(limits[3] * factor))
            hi2.append(math.sqrt(limits[4] * factor))
            if args.observed:
                assert(len(limits)>5)
                obs.append(math.sqrt(limits[5] * factor))
            filled_masses.append(mass)
        elif args.xsec:
            lo2.append(limits[0] * fac * theoryRatio * kFactor)
            lo1.append(limits[1] * fac * theoryRatio * kFactor)
            med.append(limits[2] * fac * theoryRatio * kFactor)
            hi1.append(limits[3] * fac * theoryRatio * kFactor)
            hi2.append(limits[4] * fac * theoryRatio * kFactor)
            filled_masses.append(mass)
            if args.observed:
                assert(len(limits)>5)
                obs.append(limits[5] * fac * theoryRatio * kFactor)
        
      except:
        print("No mass {}".format(mass))
        pass
    fig,ax = plt.subplots(figsize=(8,7))
    hep.cms.label(data=False if args.asimov else True,lumi=int(args.lumi),fontsize=19)
    #ax.fill_between(filled_masses,lo2,hi2,color='gold',label="Expected $\pm 2 \sigma$")
    #ax.fill_between(filled_masses,lo1,hi1,color='limegreen',label="Expected $\pm 1 \sigma$")
    ax.fill_between(filled_masses,lo2,hi2,color='#85D1FBff',label="Expected $\pm 2 \sigma$")
    ax.fill_between(filled_masses,lo1,hi1,color='#FFDF7Fff',label="Expected $\pm 1 \sigma$")
    ax.plot(filled_masses,med,color='black',linestyle="--",linewidth=1.5,label="Expected")
    if args.observed:
        #ax.plot(filled_masses,obs,color='black',linestyle="-",linewidth=1.5,label="Observed")
        ax.plot(filled_masses,obs,color='k',marker='o',linewidth=1.5,label="Observed")
    print("masses",filled_masses)
    print("median limit",med)
    print("observed limit", obs)
    if args.gq:
        if args.rb:
            ax.set_ylabel("$g'_{b}$")
        elif args.r:
            ax.set_ylabel("$g_{q}$")
            ax.set_xlabel("Z' mass [GeV]")
        elif args.rp:
            ax.set_ylabel("$g_{q\Phi}$")
            ax.set_xlabel("$\Phi$ mass [GeV]")
      
    elif args.xsec:
        if args.rb:
            ax.set_ylabel("$\sigma_B$ [pb]")
        else:
            ax.set_ylabel("$\sigma_q$ [pb]")
    #ax.set_ylim(min(lo2)*0.7,max(hi2)*2)
    if args.r:
        ax.set_ylim(0.02,0.4)
        ax.set_yscale("log")    
    if args.rp:
        ax.set_yscale("linear")    
        ax.set_ylim(0,10)
    elif args.xsec:
        ax.set_ylim(1e2, 5e4)
    ax.set_xlim(min(filled_masses),max(filled_masses)) 
    
    if args.gq:
        plt.legend(loc="upper left",fontsize=15)
    elif args.xsec:
        plt.legend(loc="upper right",fontsize=15)
    #plt.savefig(f"{args.ipath}/limits.png")
    #plt.savefig(f"{args.ipath}/limits.pdf")
    if args.r or args.rb:
        y_ticks = [0.02, 0.04, 0.07, 0.14]
        ax.set_yticks(y_ticks,y_ticks,minor=True)
        y_ticks = [0.1, 0.2, 0.4]
        ax.set_yticks(y_ticks,y_ticks)
        #ax.set_yticklabels([f'{tick:g}' for tick in y_ticks])  # Convert to non-scientific notation
        #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        #ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
    # Ensure that major and minor ticks are displayed
    #ax.yaxis.set_minor_locator(MultipleLocator(0.01))  # Adjust the minor locator as needed
    #ax.yaxis.set_major_locator(MultipleLocator(0.02))  # Adjust the major locator as needed
    plt.tight_layout()
    plt.savefig("{ipath}/limits_logy_{r_p}_{r_b}_{xsec}.png".format(ipath=args.ipath,r_p = "r_p" if args.rp else "", r_b="r_b" if args.rb else "r",xsec="xsec" if args.xsec else "gq"))
    plt.savefig("{ipath}/limits_logy_{r_p}_{r_b}_{xsec}.pdf".format(ipath=args.ipath,r_p = "r_p" if args.rp else "", r_b="r_b" if args.rb else "r",xsec="xsec" if args.xsec else "gq"))
    if args.rp:
    
        fig,ax = plt.subplots(figsize=(8,7))
        hep.cms.label(data=False if args.asimov else True,lumi=int(args.lumi),fontsize=19)
        #ax.fill_between(filled_masses,np.array(lo2)/1.5,np.array(hi2)/1.5,color='gold',label="Expected $\pm 2 \sigma$")
        #ax.fill_between(filled_masses,np.array(lo1)/1.5,np.array(hi1)/1.5,color='limegreen',label="Expected $\pm 1 \sigma$")
        ax.fill_between(filled_masses,np.array(lo2)/1.5,np.array(hi2)/1.5,color='#85D1FBff',label="Expected $\pm 2 \sigma$")
        ax.fill_between(filled_masses,np.array(lo1)/1.5,np.array(hi1)/1.5,color='#FFDF7Fff',label="Expected $\pm 1 \sigma$")
        ax.plot(filled_masses,np.array(med)/1.5,color='black',linestyle="--",linewidth=1.5,label="Expected")
        if args.observed:
            #ax.plot(filled_masses,np.array(obs)/1.5,color='black',linestyle="-",linewidth=1.5,label="Observed")
            ax.plot(filled_masses,np.array(obs)/1.5,color='black',marker='o',linewidth=1.5,label="Observed")
        ax.set_ylabel("$g_{q\mathrm{A}}$")
        ax.set_xlabel("$\mathrm{A}$ mass [GeV]")

        ax.set_xlim(min(filled_masses),max(filled_masses)) 
        ax.set_yscale("linear")    
        ax.set_ylim(0,10)
        plt.legend(loc="upper left",fontsize=15)
        plt.tight_layout()
        plt.savefig("{ipath}/limits_logy_A.pdf".format(ipath=args.ipath))
        plt.savefig("{ipath}/limits_logy_A.png".format(ipath=args.ipath))
get_graphs() 
