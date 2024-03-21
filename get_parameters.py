import sys
import numpy as np
import ROOT
import argparse

parser = argparse.ArgumentParser(description='Rhalphalib setup.')
parser.add_argument("--workspace",action="store",type=str,required=True)
parser.add_argument("--degsrho",action="store",type=int,required=True)
parser.add_argument("--degspt",action="store",type=int,required=True)
parser.add_argument("--year",action="store",type=int,required=True)
args = parser.parse_args()

tf = ROOT.TFile(args.workspace,"READ")
w = tf.Get("w")
w.loadSnapshot("MultiDimFit")

bestfit = {}
_initsMC = np.zeros(shape=(args.degspt+1,args.degsrho+1))
_initsMCUncs = np.zeros(shape=(args.degspt+1,args.degsrho+1))
for ipt in range(args.degspt+1):
    for irho in range(args.degsrho+1):
         _initsMC[ipt,irho] = w.var(f"tf2017_dataResidual_pt_par{ipt}_rho_par{irho}").getVal()
         _initsMCUncs[ipt,irho] = w.var(f"tf2017_dataResidual_pt_par{ipt}_rho_par{irho}").getError()
np.save(f"data/inits_MC_{args.year}",_initsMC)
print(_initsMC, _initsMCUncs)
