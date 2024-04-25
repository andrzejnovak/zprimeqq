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
parser.add_argument('--year', dest='year', action='store',type=str,help='year')
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
parser.add_argument('-p', dest='p', action='store_true',help='Parallel')
#parser.add_argument('--injected_signal', dest='injected_signal', action='store',type=float,help='injected signal',required=True)
#parser.add_argument('--param', dest='param', choices=["r","r_b","r_q",], action='store',type=str,help='Parameter of interest')
#parser.add_argument('--type', dest='type', choices=["toys","data"],type=str,help="toys or data")

args = parser.parse_args()


def gauss(x, a, b, c):
    return a * py.exp(-(x - b)**2.0 / (2 * c**2))




ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(ROOT.kTRUE)

commands = []

OPATH=f"results/limits/{args.postfix}/pnmd2prong/ipt0_irho0/m{args.sigmass}/m{args.sigmass}_model/"
print(OPATH)
templates = {
    "2016APV": "/eos/project/c/contrast/public/cl/www/zprime/bamboo/24Apr24-2016APV-SR/results/TEMPLATES.root",
    "2016"   : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/24Apr24-2016-SR/results/TEMPLATES.root",
    "2017"   : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/24Apr24-2017-SR/results/TEMPLATES.root",
    "2018"   : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/24Apr24-2018-SR/results/TEMPLATES.root",
    #"2017" : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/4Apr24-SR-fulldataset/results/TEMPLATES.root",
}

templates_mu = {
    "2016APV" : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/19Apr24-2016APV-CR1/results/TEMPLATES.root",
    "2016"    : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/19Apr24-2016-CR1/results/TEMPLATES.root",
    "2017"    : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/19Apr24-2017-CR1/results/TEMPLATES.root",
    "2018"    : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/19Apr24-2018-CR1/results/TEMPLATES.root",
}

tf_orders = {
    "2016APV" : " --ipt 0 --irho 0 --iptMC 2 --irhoMC 4 ",
    "2016" : " --ipt 0 --irho 0 --iptMC 2 --irhoMC 4 ",
    "2017" : " --ipt 0 --irho 0 --iptMC 2 --irhoMC 4 ",
    "2018" : " --ipt 0 --irho 0 --iptMC 2 --irhoMC 4 ",
}

taskname="limit_"+str(args.sigmass)+"_"+args.postfix
if args.r_b:
    taskname += "_rb"
condor_str = """ --memory 4000 --job-mode condor --sub-opts=\'+JobFlavour = \"workday\"\nRequestCpus = 2\n+MaxRuntime = 6000\' --task-name {taskname} """.format(taskname=taskname)
overall_cmd = ""
if args.asimov:
    overall_cmd += " -t -1 "
if args.frequentist:
    overall_cmd += " -t 1 --toysFrequentist " 

if args.r:
    overall_cmd += " --redefineSignalPOIs r -d inclusive_workspace.root "
elif args.r_b:
    overall_cmd += " --redefineSignalPOIs r_b --setParameters r_q=0,r_b=0 --freezeParameters r_q -d model_combined.root -n r_b "
elif args.r_q:
    overall_cmd += " --redefineSignalPOIs r_q -d model_combined.root -n r_q "

if args.make:
    cmd = f"python3 rhalphalib_zprime.py --opath results/limits/{args.postfix} --tagger pnmd2prong --sigmass {args.sigmass} --root_file {templates[args.year]} --root_file_mu {templates_mu[args.year]} --muonCR --MCTF --tworeg --year {args.year} --do_systematics {tf_orders[args.year]} {'--pseudo' if 'scale_full_lumi' in templates[args.year] else ''}"
    commands.append(cmd)

#print(f"cd {OPATH}")
base_dir=os.getcwd()
if args.build:
    os.chdir(OPATH)
    cmd = "bash build.sh"
    commands.append(cmd)
    if args.r:
        cmd = "text2workspace.py -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel  --PO verbose --PO 'map=.*/*{sigmass}:r[1,-5,5]' model_combined.txt -o inclusive_workspace.root".format(sigmass=args.sigmass)
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
    cmd = """ echo "combineTool.py -M AsymptoticLimits """.format(opath=OPATH)
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
        cmd = "sed -i 's|cd {pwd}|cd {pwd}/{opath}/|g' {opath}/condor_{taskname}.sh".format(taskname=taskname,pwd=os.environ["PWD"],opath=OPATH)
        commands.append(cmd)
        cmd = "condor_submit -spool {opath}/condor_{taskname}.sub".format(opath=OPATH,taskname=taskname)
        commands.append(cmd)


lumi={
  "2016APV" : 19.52,
  "2016" : 16.81,
  "2017" : 41.5,
  "2018" : 59.72,
}
if args.plot:
    #usage: plotLims.py [-h] --ipath IPATH [--observed] [--gq] [--asimov] [--lumi LUMI] [--year YEAR] [--rb]
    cmd = "python3 plotLims.py --ipath {ipath} {observed} --gq --year {year} --lumi {lumi} {rb} {asimov}".format(ipath="/".join(OPATH.split("/")[:5]),year=args.year,lumi=lumi[args.year],rb="--rb" if args.r_b else "",asimov="--asimov" if args.asimov else "", observed="--observed" if not args.asimov else "")
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

while sum([p.wait() is not None for p in processes]) < len(processes):
    try:
        time.sleep(1)
        print([p.poll() is not None for p in processes])
        print([p.wait() for p in processes])
    except KeyboardInterrupt:
        term = [p.terminate() for p in processes]

    print("TIME:", time.strftime("%H:%M:%S", time.gmtime(time.time()-start)))

