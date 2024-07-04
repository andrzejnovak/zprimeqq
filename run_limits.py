import ROOT
import glob
import matplotlib.pyplot as plt
import sys 
import numpy as np
import mplhep as hep 
import argparse 
plt.style.use(hep.style.CMS)
from scipy import optimize
import pylab as py
import uproot
import subprocess
import os
import time 
 
plt.ioff()

parser = argparse.ArgumentParser(description='Rhalphalib setup.')
parser.add_argument("--postfix", action='store', type=str, required=True, help="special tag for tests.")
parser.add_argument('--year', dest='year', action='store',choices=["2016","2016APV","2017","2018","combination"],type=str,help='year')
parser.add_argument('--make', dest='make', action='store_true',help='Make space')
parser.add_argument('--build', dest='build', action='store_true',help='Build space')
parser.add_argument('--run', dest='run', action='store_true',help='Run toys')
parser.add_argument('--plot', dest='plot', action='store_true',help='Plot limit')
parser.add_argument('--significance', dest='significance', action='store_true',help='calc significance')
parser.add_argument('--collect', dest='collect', action='store_true',help='Collect toys')
parser.add_argument('--debug', dest='debug', action='store_true',help='Debug')
parser.add_argument('--r_hat', dest='r_hat', action='store',type=float,help='expected signal strength')
parser.add_argument('--r', dest='r', action='store_true',help='Run r')
parser.add_argument('--r_q', dest='r_q', action='store_true',help='Run r_q')
parser.add_argument('--r_b', dest='r_b', action='store_true',help='Run r_b')
parser.add_argument('--init', dest='init', action='store_true',help='Initialize')
parser.add_argument('--asimov', dest='asimov', action='store_true',help='Is asimov')
parser.add_argument('--frequentist', dest='frequentist', action='store_true',help='Is frequentist')
parser.add_argument('--sigmass', dest='sigmass', action='store',type=int,help='signal mass')
parser.add_argument('--condor', dest='condor', action='store_true',help='Run on condor')
parser.add_argument('--lowbvl', dest='lowbvl', action='store_true',help='Run on lowbvl only')
parser.add_argument('--highbvl', dest='highbvl', action='store_true',help='Run on highbvl only')
parser.add_argument('--gq', dest='gq', action='store_true',help='Plot gq limit')
parser.add_argument('--xsec', dest='xsec',action='store_true',help='Plot xsec limit')
parser.add_argument('--decorr_scale_cat', dest='decorr_scale_cat',action='store_true',help='Decorrelate scale by category')
parser.add_argument('--decorr_scale_pt', dest='decorr_scale_pt',action='store_true',help='Decorrelate scale by pt')
parser.add_argument('--mask_outlier', dest='mask_outlier',action='store_true',help='Mask outlying data points')
parser.add_argument('--shift_sf_err', dest='shift_sf_err',action='store',default=1.,type=float,help='SHIFT_SF_ERR')
parser.add_argument('-p', dest='p', action='store_true',help='Parallel')
#parser.add_argument('--injected_signal', dest='injected_signal', action='store',type=float,help='injected signal',required=True)
#parser.add_argument('--param', dest='param', choices=["r","r_b","r_q",], action='store',type=str,help='Parameter of interest')
#parser.add_argument('--type', dest='type', choices=["toys","data"],type=str,help="toys or data")
args = parser.parse_args()


def gauss(x, a, b, c):
    return a * py.exp(-(x - b)**2.0 / (2 * c**2))

cmssw_str='''#!/bin/sh\nulimit -s unlimited\nset -e\ncd /eos/home-j/jekrupa/fitting/CMSSW_14_1_0_pre4/src\nexport SCRAM_ARCH=el9_amd64_gcc12\nsource /cvmfs/cms.cern.ch/cmsset_default.sh\neval `scramv1 runtime -sh`'''


ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(ROOT.kTRUE)

commands = []
if args.year == "2016APV":
    OPATH = f"results/limits/{args.postfix}/pnmd2prong/ipt2,0_irho2,0/m{args.sigmass}/m{args.sigmass}_model/"
if args.year == "2016":
    OPATH = f"results/limits/{args.postfix}/pnmd2prong/ipt2,2_irho2,2/m{args.sigmass}/m{args.sigmass}_model/"
elif args.year == "2017":
    OPATH = f"results/limits/{args.postfix}/pnmd2prong/ipt2,1_irho3,0/m{args.sigmass}/m{args.sigmass}_model/"
elif args.year == "2018":
    OPATH = f"results/limits/{args.postfix}/pnmd2prong/ipt1,0_irho1,0/m{args.sigmass}/m{args.sigmass}_model/"
elif args.year == "combination":
    OPATH = f"results/limits/combination/{args.postfix}/m{args.sigmass}/m{args.sigmass}_model/"
    
OPATH=os.path.abspath(OPATH)

templates = {
    "2016APV": "/eos/project/c/contrast/public/cl/www/zprime/bamboo/7May24-2016APV-SR/results/TEMPLATES.root",
    "2016"   : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/7May24-2016-SR/results/TEMPLATES.root",
    "2017"   : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/7May24-2017-SR/results/TEMPLATES.root",
    "2018"   : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/24Apr24-2018-SR/results/TEMPLATES.root",
}

templates_mu = {
    "2016APV" : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/19Apr24-2016APV-CR1/results/TEMPLATES_30May24.root",
    "2016"    : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/19Apr24-2016-CR1/results/TEMPLATES_30May24.root",
    "2017"    : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/19Apr24-2017-CR1/results/TEMPLATES_30May24.root",
    "2018"    : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/19Apr24-2018-CR1/results/TEMPLATES_30May24.root",
}

tf_orders = {
    "2016APV" : " --ipt 2,0 --irho 2,0 --iptMC 0,2 --irhoMC 1,3 ",
    "2016" : " --ipt 2,2 --irho 2,2 --iptMC 0,2 --irhoMC 1,3 ",
    "2017" : " --ipt 2,1 --irho 3,0 --iptMC 0,2 --irhoMC 1,4 ",
    "2018" : " --ipt 1,0 --irho 1,0 --iptMC 2,2 --irhoMC 3,4 ",
}

if args.run:
    taskname="limit_"+str(args.sigmass)+"_"+args.postfix
if args.build:
    taskname="build_"+str(args.sigmass)+"_"+args.postfix
if args.r_b:
    taskname="_rb"
if args.condor:
    condor_str = """ --memory 4000 --job-mode condor --sub-opts=\'+JobFlavour = \"workday\"\nRequestCpus = 2\n+MaxRuntime = 120000\' --task-name {taskname} """.format(taskname=taskname)
overall_cmd = ""
if args.asimov:
    overall_cmd += " -t -1 "
if args.frequentist:
    overall_cmd += " -t -1 --toysFrequentist " 

if args.r:
    overall_cmd += " --redefineSignalPOIs r -d inclusive_workspace.root "
elif args.r_b:
    overall_cmd += " --redefineSignalPOIs r_b --setParameters r_q=0,r_b=0 --freezeParameters r_q -d model_combined.root -n r_b "
elif args.r_q:
    overall_cmd += " --redefineSignalPOIs r_q -d model_combined.root -n r_q "

if args.make:
    cmd = f"python3 rhalphalib_zprime.py --opath results/limits/{args.postfix} --tagger pnmd2prong --sigmass {args.sigmass} --muonCR --MCTF --tworeg --year {args.year} --do_systematics {tf_orders[args.year]} {'--pseudo' if 'scale_full_lumi' in templates[args.year] else ''} {'--ftest --lowbvl' if args.lowbvl else ''} {'--ftest --highbvl' if args.highbvl else ''} {'--decorr_scale_cat' if args.decorr_scale_cat else ''} {'--decorr_scale_pt' if args.decorr_scale_pt else ''} --collapse {'--mask_outlier' if args.mask_outlier else ''} --shift_sf_err {args.shift_sf_err}" #-vv"
    commands.append(cmd)

#print(f"cd {OPATH}")
base_dir=os.getcwd()
if args.build:
    #os.chdir(OPATH)
    cmd = """ echo '{cmssw_str}'  > {opath}/{taskname}.sh """.format(cmssw_str=cmssw_str,opath=OPATH,taskname=taskname)
    commands.append(cmd)
    cmd = """ echo "cd {opath}" >> {opath}/{taskname}.sh """.format(opath=OPATH,taskname=taskname)
    commands.append(cmd)
    if not args.year == "combination":
        cmd = """ echo "bash {opath}/build.sh" >> {opath}/{taskname}.sh """.format(opath=OPATH,taskname=taskname)
        commands.append(cmd)
        cmd = """ echo "text2workspace.py -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel  --PO verbose --PO \'map=.*/*{sigmass}:r[1,-5,5]\' {opath}/model_combined.txt -o {opath}/inclusive_workspace.root " >> {opath}/{taskname}.sh """.format(sigmass=args.sigmass, opath=OPATH,taskname=taskname)
        commands.append(cmd)
    else:
        cmd = "cat {opath}/build.sh >> {opath}/{taskname}.sh".format(opath=OPATH,taskname=taskname)
        commands.append(cmd)
    cmd = "cp condor_templ.sub {opath}/{taskname}.sub".format(opath=OPATH,taskname=taskname)
    commands.append(cmd)
    cmd = """sed -i 's|XXX|{opath}/{taskname}.sh|g' {opath}/{taskname}.sub""".format(opath=OPATH,taskname=taskname)
    commands.append(cmd)
    if args.condor:
        cmd = "condor_submit -spool {opath}/{taskname}.sub".format(opath=OPATH,taskname=taskname)
        commands.append(cmd)
    else:
        cmd = """bash {opath}/{taskname}.sh""".format(opath=OPATH,taskname=taskname)
        commands.append(cmd) 
    
if args.significance:
    os.chdir(OPATH)
    cmd = "combine -M Significance -d model_combined.root "
    cmd += overall_cmd
    commands.append(cmd)

if args.run:
    #cmd = f"cd {OPATH}"#results/bias_tests/{args.opath}/pnmd2prong_0p01/ipt0_irho0/m{args.sigmass}/m{args.sigmass}_model/"
    #commands.append(cmd)
    #os.chdir(OPATH)
    cmd = """ echo "combineTool.py -M AsymptoticLimits --cminDefaultMinimizerStrategy 0 --cminFallbackAlgo Minuit2,0:0.4 --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 """.format(opath=OPATH)
    cmd += overall_cmd
    if args.condor:
        cmd += condor_str
        cmd += """ " > {opath}/{taskname}.sh """.format(opath=OPATH,taskname=taskname)
        commands.append(cmd)
        cmd = "bash {opath}/{taskname}.sh".format(opath=OPATH,taskname=taskname)
        commands.append(cmd)
        cmd = "mv condor_{taskname}.* {opath}".format(taskname=taskname,opath=OPATH)
        commands.append(cmd)
        cmd = "sed -i 's|executable = condor_{taskname}.sh|executable = {opath}/condor_{taskname}.sh|g' {opath}/condor_{taskname}.sub".format(taskname=taskname,opath=OPATH)
        commands.append(cmd)
        #if args.year=="combination":
        cmd = "sed -i 's|cd {pwd}|cd /{opath}/|g' {opath}/condor_{taskname}.sh".format(taskname=taskname,pwd=os.environ["PWD"],opath=OPATH)
        #else:
        #    cmd = "sed -i 's|cd {pwd}|cd {pwd}/{opath}/|g' {opath}/condor_{taskname}.sh".format(taskname=taskname,pwd=os.environ["PWD"],opath=OPATH)
        commands.append(cmd)
        cmd = "condor_submit -spool {opath}/condor_{taskname}.sub".format(opath=OPATH,taskname=taskname)
        commands.append(cmd)


lumi={
  "2016APV" : 19.52,
  "2016" : 16.81,
  "2017" : 41.5,
  "2018" : 59.72,
  "combination" : "137.6",
}

year = args.year

if year == "combination":
    year = "Run 2"
if args.plot:
    #usage: plotLims.py [-h] --ipath IPATH [--observed] [--gq] [--asimov] [--lumi LUMI] [--year YEAR] [--rb]
    cmd = "python3 plotLims.py --ipath {ipath} {observed} {xsec} {gq} --year {year} --lumi {lumi} {rb} {asimov} {decorr}".format(
        ipath="/".join(OPATH.split("/")[:-2]),
        year=args.year,lumi=lumi[args.year],
        rb="--rb" if args.r_b else "",
        asimov="--asimov" if args.asimov else "", 
        observed="--observed" if not args.asimov else "",
        xsec="--xsec" if args.xsec else "", 
        gq="--gq" if args.gq else "",
        decorr="--decorr_scale_cat" if args.decorr_scale_cat else "",
    )
    commands.append(cmd) 
if args.debug:
    for cmd in commands:
        print(cmd)
    import sys
    sys.exit()
processes = []
for cmd in commands:
    if args.p:
        processes.append(subprocess.Popen(cmd, shell=True))
    else:
        processes.append(subprocess.Popen(cmd, shell=True).wait())

#while sum([p.wait() is not None for p in processes]) < len(processes):
#    try:
#        time.sleep(1)
#        print([p.poll() is not None for p in processes])
#        print([p.wait() for p in processes])
#    except KeyboardInterrupt:
#        term = [p.terminate() for p in processes]

#print("TIME:", time.strftime("%H:%M:%S", time.gmtime(time.time()-start)))

