import argparse
import os, logging, numpy 

from rich.logging import RichHandler
from rich.prompt import Confirm
from rich.pretty import pprint
import glob
from rich.pretty import pprint
import click

parser = argparse.ArgumentParser(description="Combine cards.")
parser.add_argument("--eras", type=str, nargs="+", default=["2016APV","2016","2017","2018"])
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--opath", type=str, action="store")
parser.add_argument("--mass", type=str, action="store",default="50")
args = parser.parse_args()

if os.path.exists(f"{args.opath}/m{args.mass}/m{args.mass}_model"):
    q_overwrite = Confirm.ask(f"Path: '{args.opath}' already exists. Overwrite?")
    if not q_overwrite:
        pprint(f"Remove with \nrm -rf {args.opath}")
        sys.exit()
    else:
        os.system(f"rm -rf {args.opath}")
os.makedirs(f"{args.opath}/m{args.mass}/m{args.mass}_model",exist_ok=True)
log_level = logging.WARNING
if args.verbose:
    log_level = logging.INFO
if args.debug:
    log_level = logging.DEBUG
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.basicConfig(
    level=log_level,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, tracebacks_suppress=[click])],
)
log = logging.getLogger("rich")
opath=os.path.abspath(args.opath)
paths = {
    #"2016APV" : "results/limits/2016APV_unblinded_sigtemplfix_v1/pnmd2prong/ipt2,0_irho2,0/",
    #"2016" : "results/limits/2016_unblinded_sigtemplfix_v1/pnmd2prong/ipt2,2_irho2,2/",
    #"2017" : "results/limits/2017_unblinded_sigtemplfix_v1/pnmd2prong/ipt2,1_irho3,0/",
    #"2018" : "results/limits/2018_unblinded_sigtemplfix_v1/pnmd2prong/ipt1,0_irho1,0/",
    #"2016APV" : "results/limits/2016APV_22Jun24_unblinded_v0/pnmd2prong/ipt2,0_irho2,0/",
    #"2016" : "results/limits/2016_22Jun24_unblinded_v0/pnmd2prong/ipt2,2_irho2,2/",
    #"2017" : "results/limits/2017_22Jun24_unblinded_v0/pnmd2prong/ipt2,1_irho3,0/",
    #"2018" : "results/limits/2018_22Jun24_unblinded_v0/pnmd2prong/ipt1,0_irho1,0/",
    "2016APV" : "results/limits/2016APV_unblinded_3Jul24_sferr0p5/pnmd2prong/ipt2,0_irho2,0",
    "2016" : "results/limits/2016_unblinded_3Jul24_sferr0p5/pnmd2prong/ipt2,2_irho2,2",
    "2017" : "results/limits/2017_unblinded_3Jul24_sferr0p5/pnmd2prong/ipt2,1_irho3,0",
    "2018" : "results/limits/2018_unblinded_3Jul24_sferr0p5/pnmd2prong/ipt1,0_irho1,0",
}

pprint(paths)
def combine():
    master_str = "combineCards.py "
    for era in args.eras:
        path_to_mass = f"{os.getcwd()}/{paths[era]}/m{args.mass}/m{args.mass}_model/"
        for f in glob.glob(path_to_mass+"*.txt"):
            if "combination" in f: continue
            new_filename = f.replace('.txt', '_combination.txt')
            os.system(f"sed 's|m{args.mass}_model.root|{path_to_mass}/m{args.mass}_model.root|g' {f} > {new_filename}")
        master_str += " ptbin0passTbvlfailL_{year}={path_to_mass}ptbin0passTbvlfailL_combination.txt ptbin0passTbvlpassL_{year}={path_to_mass}ptbin0passTbvlpassL_combination.txt ptbin0failT_{year}={path_to_mass}ptbin0failT_combination.txt ptbin1passTbvlfailL_{year}={path_to_mass}ptbin1passTbvlfailL_combination.txt ptbin1passTbvlpassL_{year}={path_to_mass}ptbin1passTbvlpassL_combination.txt ptbin1failT_{year}={path_to_mass}ptbin1failT_combination.txt ptbin2passTbvlfailL_{year}={path_to_mass}ptbin2passTbvlfailL_combination.txt ptbin2passTbvlpassL_{year}={path_to_mass}ptbin2passTbvlpassL_combination.txt ptbin2failT_{year}={path_to_mass}ptbin2failT_combination.txt ptbin3passTbvlfailL_{year}={path_to_mass}ptbin3passTbvlfailL_combination.txt ptbin3passTbvlpassL_{year}={path_to_mass}ptbin3passTbvlpassL_combination.txt ptbin3failT_{year}={path_to_mass}ptbin3failT_combination.txt ptbin4passTbvlfailL_{year}={path_to_mass}ptbin4passTbvlfailL_combination.txt ptbin4passTbvlpassL_{year}={path_to_mass}ptbin4passTbvlpassL_combination.txt ptbin4failT_{year}={path_to_mass}ptbin4failT_combination.txt muonCRpassTbvlpassL_{year}={path_to_mass}muonCRpassTbvlpassL_combination.txt muonCRpassTbvlfailL_{year}={path_to_mass}muonCRpassTbvlfailL_combination.txt muonCRfailT_{year}={path_to_mass}muonCRfailT_combination.txt > {opath}/m{mass}/m{mass}_model/model_combined.txt".format(year=era, path_to_mass=path_to_mass,opath=opath,mass=args.mass)
    master_str += """\ntext2workspace.py -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel  --PO verbose --PO 'map=.*/*m{sigmass}:r_q[1,-5,5]'  --PO 'map=.*/*b{sigmass}:r_b[1,-5,5]' {opath}/m{sigmass}/m{sigmass}_model/model_combined.txt""".format(sigmass=args.mass,opath=opath)
    master_str += """\ntext2workspace.py -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel  --PO verbose --PO 'map=.*/*{sigmass}:r[1,-5,5]' {opath}/m{sigmass}/m{sigmass}_model/model_combined.txt -o {opath}/m{sigmass}/m{sigmass}_model/inclusive_workspace.root""".format(sigmass=args.mass,opath=opath)
    with open(f"{opath}/m{args.mass}/m{args.mass}_model/build.sh","w") as f:
        f.write(master_str)
    
combine()
