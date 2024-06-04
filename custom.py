#!/usr/bin/env python
from __future__ import print_function
import os, argparse, json, string, re
import numpy as np
import time
# import ROOT
# ROOT.gROOT.SetBatch(True)
# ROOT.gStyle.SetOptStat(0)

np.random.seed(int(time.time()))
def pseudorand_str(size):
    ALPHABET = np.array(list(string.ascii_lowercase + string.digits))
    return "".join(np.random.choice(ALPHABET, size=size))

def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def exec_bash(command='echo "hello world"', debug=False):
    if debug:
        print("XXXXXXXXXXXXXXXXXXXXXXXX in dir: " + os.getcwd())
        print(command)
    else:
        os.system(command)
    return """%s\n""" % command


def FTest(seed=1, base=False, gen=False, fits=False, args=None, mc=True):
    debug = args.debug
    if mc:
        #overall_conf = f" --setParameters r=0 --freezeParameters r --expectSignal=0 --redefineSignalPOIs tf{args.year}_dataResidual_pt_par0_rho_par0 "
        overall_conf = f" --expectSignal=0 --setParameters r=0 --freezeParameters r "# --redefineSignalPOIs tf{args.year}_dataResidual_pt_par0_rho_par0 "
        #overall_conf += f" --redefineSignalPOIs tf{args.year}_dataResidual_0_pt_par0_rho_par0 "
        #elif args.highbvl:
        #    overall_conf += " --freezeParameters r,'rgx{{tf{year}_dataResidual_1_*}}' --redefineSignalPOIs tf{year}_dataResidual_0_pt_par0_rho_par0".format(year=args.year)
        #elif args.lowbvl:
        #    overall_conf += " --freezeParameters r,'rgx{{tf{year}_dataResidual_0_*}}' --redefineSignalPOIs tf{year}_dataResidual_1_pt_par0_rho_par0".format(year=args.year)
    else:
        overall_conf = " " #f" --freezeParameters r --setParameters r=0 " # --redefineSignalPOIs tf2017_MC_0templ_deco0 --freezeParameters r --setParameters r=0 --toysFr "
        #overall_conf = f" --redefineSignalPOIs tf{args.year}_MC_0templ_deco0 --freezeParameters r --setParameters r=0 " # --redefineSignalPOIs tf2017_MC_0templ_deco0 --freezeParameters r --setParameters r=0 --toysFr "
        #overall_conf = f" --redefineSignalPOIs tf{args.year}_dataResidual_0_pt_par0_rho_par0 --setParameters r=0 --freezeParameters r "
        #overall_conf = f" --expectSignal=0 --freezeParameters CMS_PNet_bb_{args.year} --setParameters CMS_PNet_bb_{args.year}=1 " # --freezeParameters r --setParameters r=0 --redefineSignalPOIs tf{args.year}_dataResidual_0_pt_par0_rho_par0 "# --freezeParameters r --redefineSignalPOIs tf{args.year}_dataResidual_0_pt_par0_rho_par0 "  
        #overall_conf = f" --setParameters r=0 --freezeParameters r --redefineSignalPOIs tf{args.year}_dataResidual_pt_par0_rho_par0 --toysFrequentist "  
        #overall_conf = " --setParameters r=1 --toysFrequentist --freezeParameters r --setParameterRanges r=-10,10 "
        #overall_conf = " --setParameters r=1,z=1  --toysFrequentist --freezeParameters r,z --setParameterRanges r=-10,10:z=0.95,1.05 "
        #overall_conf = " --setParameters r=0  --toysFrequentist --freezeParameters r "
    CONDOR_str = """ --job-mode condor --sub-opts='+JobFlavour = "workday"' --task-name cfg_{}"""

    ref_dir = os.getcwd()
    base_dir = os.path.realpath(args.d)
    print("BASEDIR:", base_dir)
    print("SEED: ", seed)
    # Fetch right configs
    ws_base = os.path.realpath(args.workspace)
    ws_alt = os.path.realpath(args.altspace)
    dir_base = os.path.dirname(os.path.realpath(args.workspace))
    dir_alt = os.path.dirname(os.path.realpath(args.altspace))
    configs_base = json.load(open(os.path.join(dir_base, "config.json")))
    configs_alt = json.load(open(os.path.join(dir_alt, "config.json")))
    degs_base= "".join(str(x) for x in [configs_base['ipt'],configs_base['irho']])
    #degs_base = "".join(configs_base['degs'].split(","))
    degs_alt= "".join(str(x) for x in [configs_alt['ipt'],configs_alt['irho']])
    #degs_alt = "".join(configs_alt['degs'].split(","))
    workdir = 'ftest_{}_{}'.format(degs_base, degs_alt)
    ensure_dir(os.path.join(base_dir, workdir))
    os.chdir(os.path.join(base_dir, workdir))
    condor_total_command = ""  
    # Data fits
    if base:
        command = (
            "combineTool.py -M MultiDimFit --cminDefaultMinimizerStrategy 0 --robustFit=1   "
            " -n .BaseFit  --saveWorkspace --saveFitResult "
            #" --setParameterRanges r=-5,5 "
            " --freezeParameters r --setParameters r=0 " 
            " -d {}".format(ws_base) + overall_conf)

        if args.condor:
            command += CONDOR_str.format("basemultifit")
    
        exec_bash(command, debug)
        command = (
            "combineTool.py -M MultiDimFit --cminDefaultMinimizerStrategy 0 --robustFit=1   "
            " -n .AltFit  --saveWorkspace --saveFitResult"
            #" --setParameterRanges r=-5,5 " 
            " --freezeParameters r --setParameters r=0 " 
            " -d {}".format(ws_alt) + overall_conf)
        if args.condor:
            command += CONDOR_str.format("altmultifit")
        exec_bash(command, debug)

        # Shapes from a fit
        command = (
            "combineTool.py -M FitDiagnostics --cminDefaultMinimizerStrategy 0 --robustFit=1   "
            " -n .Base  --saveWorkspace --saveShapes " #--SaveWithUncertainties"
            " --skipSBFit "
            " -d {}".format(ws_base) + overall_conf)
        if args.condor:
            command += CONDOR_str.format("basefitdiag")
        exec_bash(command, debug)

        command = (
            "combineTool.py -M FitDiagnostics --cminDefaultMinimizerStrategy 0 --robustFit=1   "
            " -n .Alt  --saveWorkspace --saveShapes "#--saveWithUncertainties"
            " --skipSBFit "
            " -d {}".format(ws_alt) + overall_conf)
        if args.condor:
            command += CONDOR_str.format("altfitdiag")
        exec_bash(command, debug)

        # GoFs Data
        command = ("combineTool.py -M GoodnessOfFit  --algo saturated --cminDefaultMinimizerStrategy 0 "
            #" --snapshotName MultiDimFit --bypassFrequentistFit "
            " -n .Base "
            " -d {ws}".format(ws="higgsCombine.BaseFit.MultiDimFit.mH120.root")
            + " --fixedSignalStrength=0 "
            +  overall_conf
        )
        if args.condor:
            command += CONDOR_str.format("basegof")
        exec_bash(command, debug)
        command = ("combineTool.py -M GoodnessOfFit  --algo saturated --cminDefaultMinimizerStrategy 0 "
            #" --snapshotName MultiDimFit --bypassFrequentistFit "
            " -n .Alt "
            " -d {ws}".format(ws="higgsCombine.AltFit.MultiDimFit.mH120.root")
            + " --fixedSignalStrength=0 "
            +  overall_conf
        )
        if args.condor:
            command += CONDOR_str.format("altgof")

        exec_bash(command, debug)

    # Generate toys
    if gen:
        command = ("combineTool.py -M GenerateOnly  --saveToys "
                #" --snapshotName MultiDimFit --bypassFrequentistFit "
                " --toysFrequentist " 
                " -n Toys "
                " -t {t} --seed {s} "
                " --expectSignal=0 --freezeParameters r --setParameters r=0 "
                " -d {ws}".format(ws="higgsCombine.BaseFit.MultiDimFit.mH120.root", t=args.toys, s=seed) 
                +  overall_conf
        )
        #if not mc:
        #    command += f" --redefineSignalPOIs tf{args.year}_MC_0templ_deco0 --freezeParameters r --setParameters r=0 "
        if args.condor:
            command += CONDOR_str.format("toysgen_{}_{}".format(seed, pseudorand_str(4)))
        exec_bash(command, debug)

    # # GoFs Toys
    if fits:
        command = ("combineTool.py -M GoodnessOfFit  --algo saturated --cminDefaultMinimizerStrategy 0 " 
            #" --snapshotName MultiDimFit --bypassFrequentistFit " 
            #" --toysFrequentist "
            " -n .BaseToys " 
            " -t {t} --seed {s} " 
            " --toysFile higgsCombineToys.GenerateOnly.mH120.{s}.root " 
            " -d {ws}".format(ws="higgsCombine.BaseFit.MultiDimFit.mH120.root", t=args.toys, s=seed) 
            +  overall_conf
        )
        #if not args.mc:
        #        command+=" --toysFrequentist "
        if args.condor:
            command += CONDOR_str.format("fitbase_{}_{}".format(seed, pseudorand_str(4)))
        exec_bash(command, debug)

        command = ("combineTool.py -M GoodnessOfFit  --algo saturated --cminDefaultMinimizerStrategy 0 " 
            #" --snapshotName MultiDimFit --bypassFrequentistFit " 
            #" --toysFrequentist "
            " -n .AltToys " 
            " -t {t} --seed {s} " 
            " --toysFile higgsCombineToys.GenerateOnly.mH120.{s}.root " 
            " -d {ws}".format(ws="higgsCombine.AltFit.MultiDimFit.mH120.root", t=args.toys, s=seed) 
            +  overall_conf
        )
        #if not args.mc:
        #        command+=" --toysFrequentist "
        if args.condor:
            command += CONDOR_str.format("fitalt_{}_{}".format(seed, pseudorand_str(4)))
        exec_bash(command, debug)

    os.chdir(ref_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument("--debug",
                        type=str2bool,
                        default='False',
                        choices={True, False},
                        help="")

    parser.add_argument('--workspace', '-w', default='model_combined.root')
    parser.add_argument('--altspace', '-a', default='model_combined.root')
    parser.add_argument('-d', default=None, help='Workdir to store outputs, otherwise will run in -w parent directory')
    parser.add_argument('--toys', '-t', default="10")
    parser.add_argument('--seed', '-s', default="1")
    parser.add_argument('--year',type=str,default="2017")
    parser.add_argument('--base', action="store_true")
    parser.add_argument('--gen', action="store_true")
    parser.add_argument('--fits', action="store_true")
    parser.add_argument('--all', action="store_true")
    parser.add_argument('--highbvl', action="store_true")
    parser.add_argument('--lowbvl', action="store_true")

    parser_mc = parser.add_mutually_exclusive_group(required=True)
    parser_mc.add_argument('--data', action='store_false', dest='mc')
    parser_mc.add_argument('--mc', action='store_true', dest='mc')

    parser.add_argument('--condor', action="store_true")
    args = parser.parse_args()

    if args.all:
        args.base = True
        args.gen = True
        args.fits = True

    if args.d is None:
        args.d = "/".join(args.workspace.split("/")[:-1])

    if re.match("-?[0-9]$", str(args.seed)):
        FTest(seed=args.seed, base=args.base, gen=args.gen, fits=args.fits, args=args, mc=args.mc) 
    else:
        if re.match("[0-9]+:[0-9]+:[0-9]+", args.seed) is not None:
            start, stop, step = args.seed.split(":")
            # Run base separetly
            FTest(args.seed, base=args.base, gen=args.gen, fits=False, args=args, mc=args.mc) 
            # Run generation (seed can be parsed)
            FTest(args.seed, base=args.base, gen=args.gen, fits=False, args=args, mc=args.mc) 
            # Run toys (must be parsed separately)
            if not args.fits:
                import sys
                sys.exit()
            for seed in range(int(start), int(stop) + int(step), int(step)):
                FTest(seed, base=False, gen=False, fits=args.fits, args=args, mc=args.mc) 

        else:
            raise ValueError("Seed not understood")
        
