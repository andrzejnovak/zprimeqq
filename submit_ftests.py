# from distutils import command
import os
import subprocess
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default="FTests/mconly16")
    parser.add_argument('--opath', required=True,type=str,)
    parser.add_argument('--outplots', default="Ftests/plots")
    parser.add_argument('--root_file', required=False, type=str, help="Path to ROOT files containing templates")
    parser.add_argument('--root_file_mu', required=False, type=str, help="Path to ROOT files containing templates")

    parser_mc = parser.add_mutually_exclusive_group(required=True)
    parser_mc.add_argument('--data', action='store_false', dest='mc')
    parser_mc.add_argument('--mc', action='store_true', dest='mc')
    parser.add_argument('--highbvl',action='store_true', dest='highbvl')
    parser.add_argument('--lowbvl',action='store_true', dest='lowbvl')
    parser.add_argument('--tworeg',action='store_true', dest='tworeg')
    parser.add_argument('--year', type=str, choices=["2016", "2016APV", "2017", "2018"], required=True, help="Year to display.")
    parser.add_argument('--is_blinded', action='store_true', help='')
    parser.add_argument('--param', type=str, choices=['bern', 'cheby', 'exp'], default='bern')

    parser.add_argument('--tagger',type=str, choices=['pnmd2prong',],required=True)
    parser.add_argument('--make', action='store_true', help='')
    parser.add_argument('--build', action='store_true', help='')

    parser.add_argument('--run', action="store_true")
    parser.add_argument('--toys', '-t', default="10")
    parser.add_argument('--seed', '-s', default="1")
    parser.add_argument('--base', action="store_true")
    parser.add_argument('--gen', action="store_true")
    parser.add_argument('--fits', action="store_true")
    parser.add_argument('--all', action="store_true")
    parser.add_argument('--condor', action="store_true")

    parser.add_argument('--plot', action="store_true")
    parser.add_argument('-p', action="store_true", help="parallel")
    parser.add_argument('--debug', action="store_true", help="Just print")

    args = parser.parse_args()
    os.system(f"mkdir -p {args.opath}")
    start = time.time()
    commands = []

    rng_pt = 4
    rng_rho = 4

    if args.param == 'cheby':
        basis = ' --basis Bernstein,Chebyshev'
    elif args.param == 'exp':
        basis = ' --basis Bernstein,Bernstein --transform '
    else:
        basis = " "
    if args.tworeg and not (args.highbvl or args.lowbvl):
        raise RuntimeError("Need args.highbvl or args.lowbvl for f-test")

    templates = {
        "2016APV": "/eos/project/c/contrast/public/cl/www/zprime/bamboo/24Apr24-2016APV-SR/results/TEMPLATES.root",
        "2016"   : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/24Apr24-2016-SR/results/TEMPLATES.root",
        "2017"   : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/24Apr24-2017-SR/results/TEMPLATES.root",
        "2018"   : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/24Apr24-2018-SR/results/TEMPLATES.root",
        #"2017" : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/4Apr24-SR-fulldataset/results/TEMPLATES.root",
    }
    if args.is_blinded:
        templates = {
            "2016APV": "/eos/project/c/contrast/public/cl/www/zprime/bamboo/17Apr24-SR-2016APV-10pct-3/results/TEMPLATES_blind.root",
            "2016"   : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/17Apr24-SR-2016-10pct-3/results/TEMPLATES_blind.root",
            "2017"   : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/24Apr24-2017-SR/results/TEMPLATES.root",
            "2018"   : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/17Apr24-SR-2018-10pct-9/results/TEMPLATES_blind.root",
            #"2017" : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/4Apr24-SR-fulldataset/results/TEMPLATES.root",
        }
    else: 
        templates = {
            "2016APV": "/eos/project/c/contrast/public/cl/www/zprime/bamboo/7May24-2016APV-SR/results/TEMPLATES.root",
            "2016"   : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/7May24-2016-SR/results/TEMPLATES.root",
            "2017"   : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/7May24-2017-SR/results/TEMPLATES.root",
            "2018"   : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/24Apr24-2018-SR/results/TEMPLATES.root",
            #"2017" : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/4Apr24-SR-fulldataset/results/TEMPLATES.root",
        }
    templates_mu = {
        "2016APV" : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/19Apr24-2016APV-CR1/results/TEMPLATES_30May24.root",
        "2016"    : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/19Apr24-2016-CR1/results/TEMPLATES_30May24.root",
        "2017"    : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/19Apr24-2017-CR1/results/TEMPLATES_30May24.root",
        "2018"    : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/19Apr24-2018-CR1/results/TEMPLATES_30May24.root",
    }
 
    if args.make:
        # Made workspaces
        for pt in range(0, rng_pt + 1):
            for rho in range(0, rng_rho + 1):
                ###I don't need just MC templates?
                if args.mc:
                    cmd = (
                        f"python3 rhalphalib_zprime.py --pseudo --year {args.year} --root_file {templates[args.year]} --o {args.opath}{pt}{rho} --ipt {pt} --irho {rho} --tagger {args.tagger} --qcd_ftest --root_file_mu {templates_mu[args.year]} "
                        + (" --highbvl" if args.highbvl else "") 
                        + (" --lowbvl" if args.lowbvl else "")
                        + (" --tworeg" if args.tworeg else "") 
                    ) 
                else:
                    cmd = (
                        f"python3 rhalphalib_zprime.py --year {args.year} --root_file {templates[args.year]} --o {args.opath}{pt}{rho} --do_systematics --tagger {args.tagger} --MCTF --ftest --root_file_mu {templates_mu[args.year]} --constrainedttbar " ## Trying to exclude muonCR in favor of constrained nuisances from a pre-fit to muonCR
                        #"--mutemplates temps/templatesmuCR_preapproval{yearshort}_CC.root  --muonCR "
                        + (" --is_blinded " if args.is_blinded else "")
                        + (" --highbvl" if args.highbvl else "") 
                        + (" --lowbvl" if args.lowbvl else "")
                        + (" --tworeg" if args.tworeg else "") 
                        
                    )
                    if args.year == "2016APV":
                        cmd += " --irhoMC 1,3 --iptMC 0,2"
                    elif args.year == "2016":
                        cmd += " --irhoMC 1,3 --iptMC 0,2"
                    elif args.year == "2017":
                        cmd += " --irhoMC 1,4 --iptMC 0,2 "
                    elif args.year == "2018":
                        cmd += " --irhoMC 3,4 --iptMC 2,2 "
                    else:
                        raise ValueError(f"Dont have MC poly for year {args.year}!")
                    if args.highbvl:
                        cmd += f" --irho {rho},XXX --ipt {pt},XXX "
                    elif args.lowbvl:
                        cmd += f" --irho XXX,{rho} --ipt XXX,{pt} "
                commands.append(cmd)

    if args.build:
        for pt in range(0, rng_pt + 1):
            for rho in range(0, rng_rho + 1):
                _refdir = os.path.realpath(os.getcwd())
                #00/pnmd2prong_0p05/ipt0_irho0/m150/m150_model
                if args.lowbvl:
                    ptrhostring = f"iptXXX,{pt}_irhoXXX,{rho}"
                elif args.highbvl:
                    ptrhostring = f"ipt{pt},XXX_irho{rho},XXX"
                path = "{}/{}{}/{}/{}/m150/m150_model".format(args.opath, str(pt), str(rho), args.tagger, ptrhostring)
                print("XXXXXXXXXXXXXXXX")
                print("Working in:")
                print("    ", path)
                print("XXXXXXXXXXXXXXXX")
                os.chdir(path)

                os.system("bash build.sh")
                os.chdir(_refdir)


    if args.all:
        args.base = True
        args.gen = True
        args.fits = True

    
    if args.run:
        run_cfg = (" --base " if args.base else "") + (" --gen " if args.gen else "") + (" --fits " if args.fits else "")
        if args.condor:
            run_cfg += " --condor "
        toy_cfg = " -t {} -s {} ".format(args.toys, args.seed)
        for i in range(0,rng_pt):
            for j in range(0,rng_pt):
                base_cmd = "python3 custom.py  --debug False -d {} {} --year {} ".format(args.opath, "--mc" if args.mc else "--data", args.year)
                if args.highbvl: base_cmd += " --highbvl "
                if args.lowbvl: base_cmd += " --lowbvl "

                if args.lowbvl:
                    ptrhostring = f"iptXXX,PT_irhoXXX,RHO"
                elif args.highbvl:
                    ptrhostring = f"iptPT,XXX_irhoRHO,XXX"
                #ws_base = " -w {base}/{i}{j}/{tagger}/ipt{i}_irho{j}/m150/m150_model/model_combined.root "
                #ws_alt = " -a {base}/{i}{j}/{tagger}/ipt{i}_irho{j}/m150/m150_model/model_combined.root "
                ws_base = " -w {base}/{i}{j}/{tagger}/{ptrhostring}/m150/m150_model/model_combined.root "
                ws_alt = " -a {base}/{i}{j}/{tagger}/{ptrhostring}/m150/m150_model/model_combined.root "
                print(ws_base,ws_alt)
                cmd = base_cmd + ws_base.format(base=args.opath,tagger=args.tagger, i=i, j=j, ptrhostring=ptrhostring.replace("PT",str(i)).replace("RHO",str(j))) + ws_alt.format(base=args.opath,tagger=args.tagger,  i=i + 1, j=j, ptrhostring=ptrhostring.replace("PT",str(i+1)).replace("RHO",str(j))) + toy_cfg + run_cfg
                commands.append(cmd)
                cmd = base_cmd + ws_base.format(base=args.opath,tagger=args.tagger,  i=i, j=j, ptrhostring=ptrhostring.replace("PT",str(i)).replace("RHO",str(j))) + ws_alt.format(base=args.opath,tagger=args.tagger,  i=i, j=j + 1, ptrhostring=ptrhostring.replace("PT",str(i)).replace("RHO",str(j+1))) + toy_cfg + run_cfg
                commands.append(cmd)
                cmd = base_cmd + ws_base.format(base=args.opath,tagger=args.tagger,  i=i, j=j, ptrhostring=ptrhostring.replace("PT",str(i)).replace("RHO",str(j))) + ws_alt.format(base=args.opath,tagger=args.tagger,  i=i + 1, j=j + 1, ptrhostring=ptrhostring.replace("PT",str(i+1)).replace("RHO",str(j+1))) + toy_cfg + run_cfg
                commands.append(cmd)


    if args.plot:
        base_cmd = "python3 new_plot_ftests.py -o {} --year {} {} {} {} ".format(args.outplots, args.year, "--mc" if args.mc else "", "--lowbvl" if args.lowbvl else "", "--highbvl" if args.highbvl else "")
        if args.mc:
            base_cmd += " --qplots "
        for i in range(rng_pt+1):
            for j in range(rng_pt+1):
                if args.lowbvl:
                    ptrhostring = f"XXX,PT1XXX,RHO1_XXX,PT2XXX,RHO2"
                elif args.highbvl:
                    ptrhostring = f"PT1,XXXRHO1,XXX_PT2,XXXRHO2,XXX"
                    #ftest_3,XXX2,XXX_4,XXX2,XXX
                #ftest_XXX,0XXX,1_XXX,1XXX,1
                cmd = base_cmd + " --degs {},{} ".format(i, j) + " --degsalt {},{} ".format(i, j + 1)
                cmd += " -d {}/ftest_{}".format(args.opath, ptrhostring.replace("PT1",str(i)).replace("RHO1",str(j)).replace("PT2",str(i)).replace("RHO2",str(j+1)))
                commands.append(cmd)
                cmd = base_cmd + " --degs {},{} ".format(i, j) + " --degsalt {},{} ".format(i + 1, j)
                cmd += " -d {}/ftest_{}".format(args.opath, ptrhostring.replace("PT1",str(i)).replace("RHO1",str(j)).replace("PT2",str(i+1)).replace("RHO2",str(j)))
                commands.append(cmd)
                cmd = base_cmd + " --degs {},{} ".format(i, j) + " --degsalt {},{} ".format(i + 1, j + 1)
                cmd += " -d {}/ftest_{}".format(args.opath, ptrhostring.replace("PT1",str(i)).replace("RHO1",str(j)).replace("PT2",str(i+1)).replace("RHO2",str(j+1)))
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

# python submit_ftests.py -d FTests/mconly16 -o Ftests/Fouts



