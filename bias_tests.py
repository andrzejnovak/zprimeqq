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
plt.ioff()

parser = argparse.ArgumentParser(description='Rhalphalib setup.')
parser.add_argument("--postfix", action='store', type=str, required=True, help="special tag for tests.")
parser.add_argument('--year', dest='year', action='store',type=str,help='year')
parser.add_argument('--make', dest='make', action='store_true',help='Make space')
parser.add_argument('--build', dest='build', action='store_true',help='Build space')
parser.add_argument('--run', dest='run', action='store_true',help='Run toys')
parser.add_argument('--significance', dest='significance', action='store_true',help='calc significance')
parser.add_argument('--collect', dest='collect', action='store_true',help='Collect toys')
parser.add_argument('--debug', dest='debug', action='store_true',help='Debug')
parser.add_argument('--condor', dest='condor', action='store_true',help='Run condor')
parser.add_argument('--r_hat', dest='r_hat', action='store',type=float,help='expected signal strength')
parser.add_argument('--r', dest='r', action='store_true',help='Run r')
parser.add_argument('--r_q', dest='r_q', action='store_true',help='Run r_q')
parser.add_argument('--r_b', dest='r_b', action='store_true',help='Run r_b')
parser.add_argument('--init', dest='init', action='store_true',help='Initialize')
parser.add_argument('--lumi', dest='lumi', action='store',type=float,help='luminosity')
parser.add_argument('--sigmass', dest='sigmass', action='store',type=int,help='signal mass')
#parser.add_argument('--injected_signal', dest='injected_signal', action='store',type=float,help='injected signal',required=True)
#parser.add_argument('--param', dest='param', choices=["r","r_b","r_q",], action='store',type=str,help='Parameter of interest')
#parser.add_argument('--type', dest='type', choices=["toys","data"],type=str,help="toys or data")

args = parser.parse_args()


def gauss(x, a, b, c):
    return a * py.exp(-(x - b)**2.0 / (2 * c**2))



taskname="limit_"+str(args.sigmass)+"_"+args.postfix
condor_str = """ --job-mode condor --sub-opts=\'+JobFlavour = \"longlunch\"\' --task-name {taskname} --job-dir results/limits/test_v0/pnmd2prong_0p01/ipt0_irho0/m125/m125_model/ """.format(taskname=taskname)
overall_cmd = ""


ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(ROOT.kTRUE)

commands = []

OPATH=f"results/bias_tests/{args.postfix}/pnmd2prong_0p01/ipt0_irho0/m{args.sigmass}/m{args.sigmass}_model/"
if args.r_q:
    overall_cmd = f" --redefineSignalPOIs r_q --setParameters r_b=0,r_q={args.r_hat} --freezeParameters r_b -n {args.r_hat}_rq_bias -d model_combined.root "
elif args.r:
    overall_cmd = f" --redefineSignalPOIs r --expectSignal={args.r_hat} -n {args.r_hat}_r_bias -d inclusive_workspace.root "
elif args.r_b:
    overall_cmd = f" --redefineSignalPOIs r_b --setParameters r_q=0,r_b={args.r_hat} --freezeParameters r_b -n {args.r_hat}_rb_bias -d model_combined.root "


if args.make:
    cmd = f"python3 rhalphalib_zprime.py --ipt 0 --irho 0 --opath results/bias_tests/{args.postfix} --tagger pnmd2prong_0p01 --sigmass {args.sigmass} --root_file /eos/project/c/contrast/public/cl/www/zprime/bamboo/4Apr24-SR-fulldataset/results/TEMPLATES.root --root_file_mu /eos/project/c/contrast/public/cl/www/zprime/bamboo/22Mar24-split-CR1-3/results/TEMPLATES.root --muonCR --MCTF --iptMC 2 --irhoMC 4 --tworeg --year 2017 -vv --do_systematics"

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
    cmd = "combine -M Significance -t -1 "
    cmd += overall_cmd
    commands.append(cmd)

if args.run:
    cmd = """ echo "combineTool.py -M FitDiagnostics -t 2 --toysFrequentist -s 1:500:1 """
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



if args.collect:
    if args.r:
        name="r_bias"
    elif args.r_b:
        name="rb_bias"
    elif args.r_q:
        name="rq_bias"
    #higgsCombine0.2_r_bias.FitDiagnostics.mH120.43.root
    toyfiles = glob.glob("{OPATH}/higgsCombine{r_hat}_{name}.FitDiagnostics.mH120.*.root".format(OPATH=OPATH,r_hat=args.r_hat,name=name)) # update the path here!
    
    mus,muerrs = [],[]
    for toy in toyfiles:
        fin = ROOT.TFile(toy)
        if fin.IsZombie():
            continue
        try:
            mu = uproot.open(toy)["limit"]["limit"].array()
            muerr = uproot.open(toy)["limit"]["limitErr"].array()
                
            for m,merr in zip(mu,muerr):
                mus.append(m)
                muerrs.append(merr)
        except:
            pass
        fin.Close()
    
    p_to_label = {
        "r_b" : r"$r_{b}$",
        "r_q" : r"$r_{q}$",
    
    } 
    mus = np.array(mus[3::4])
    muerrs = np.array(muerrs[3::4])
    #mus = mus[:500]
    #muerrs = muerrs[:500]
    safe = (mus-muerrs>-2) & (mus+muerrs<2)
    mus=mus[safe]
    muerrs=muerrs[safe]
    print(len(~safe))
    mus -= args.r_hat
    #mus /= muerrs
    mus /= mus.std()
    mean,stdev = mus.mean(),mus.std()
    fig,ax = plt.subplots()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    #n, bins, patches = ax.hist(mus,bins=25,range=(np.floor(mus.min()*2)/2,np.ceil(mus.max()*2)/2),facecolor='orange',linewidth=1.2,edgecolor='black')
    n, bins, patches = ax.hist(mus,bins=25,range=(np.floor(mus.min()*2)/2,np.ceil(mus.max()*2)/2),facecolor='orange',linewidth=1.2,edgecolor='black')

    x = [0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)]
    y = n.tolist()
    popt, pcov = optimize.curve_fit(gauss, x, y)
    x_fit = py.linspace(x[0], x[-1], 100)
    y_fit = gauss(x_fit, *popt)


    hep.cms.label(data=False, year=args.year, lumi=args.lumi, fontsize=20)
    plt.plot(x_fit, y_fit, lw=4, color="r")

    print(popt)
    ax.text(0.05, 0.95, (r'$\mu$={:.2f}'+'\n'+r'$\sigma$={:.2f}').format(popt[1],np.abs(popt[2])), transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
    ax.text(0.80, 0.95, (r'toys: {0}').format(mus.shape[0]), transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
    #plt.title(r"Injected $\mu={injected_mu}$, initial fit: freely floating $\mu$ VPT, toys fit dEtabb".format(injected_mu=injected_mu))
    plt.xlabel(r"(r - $\hat{r}$) / $\sigma$")#{p_to_label[p]}")
    #plt.xlabel(r"measured {param}".format(param=p))
    #plt.ylim([0,60])

    if args.r_q:
       postfix="rq"
    elif args.r:
       postfix="r"
    elif args.r_b:
       postfix="rb"
    fig.savefig("{OPATH}/Bias_{p}.pdf".format(OPATH=OPATH,p=postfix))
    fig.savefig("{OPATH}/Bias_{p}.png".format(OPATH=OPATH,p=postfix))

    
if args.debug:
    for cmd in commands:
        print(cmd)
    import sys
    sys.exit()
processes = []
for cmd in commands:
    #processes.append(os.system(cmd))
    processes.append(subprocess.Popen(cmd, shell=True).wait())

os.chdir(base_dir)
