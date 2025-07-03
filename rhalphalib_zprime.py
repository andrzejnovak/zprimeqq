import sys
import os
import rhalphalib as rl
import numpy as np
import scipy.stats
import pickle
import ROOT
import json
import pandas as pd
import argparse
import uproot
from array import array
from rhalphalib import AffineMorphTemplate, MorphHistW2
import logging
from rich.logging import RichHandler
from rich.prompt import Confirm
from rich.pretty import pprint
import click
from common import sys_name_updown, lumi_dict, lumi_correlated_dict_unc, lumi_1718_dict_unc, lumi_dict_unc
import time
from copy import deepcopy
start_time=time.time()

from rich.traceback import install
install(show_locals=False)

rl.util.install_roofit_helpers()
# rl.ParametericSample.PreferRooParametricHist = False
np.random.seed(1)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    
parser = argparse.ArgumentParser(description="Rhalphalib setup.")
parser.add_argument(
    "--opath", action="store", type=str, required=True, help="Path to store output."
)
parser.add_argument(
    "--ipt", action="store", type=str, required=True, help="TF pt order (high,low)."
)
parser.add_argument(
    "--irho", action="store", type=str, required=True, help="TF rho order (high,low)."
)
parser.add_argument(
    "--iptMC", action="store", type=str, required=False, help="MCTF pt order (high,low)."
)
parser.add_argument(
    "--irhoMC", action="store", type=str, required=False, help="MCTF rho order (high,low)."
)
parser.add_argument(
    "--collapse", action="store_true", help="Collapse muon CR."
)
parser.add_argument(
    "--tagger",
    action="store",
    type=str,
    required=True,
    help="Tagger name to cut, for example pnmd2prong_ddt.",
)
parser.add_argument(
    "--pickle",
    action="store",
    type=str,
    required=False,
    help="Path to pickle holding templates.",
)
parser.add_argument(
    "--sigmass",
    action="store",
    type=str,
    required=False,
    default="150",
    help="mass point like 150.",
)
# parser.add_argument("--root_path", action='store', type=str, required=True, help="Path to ROOT holding templates.")
#parser.add_argument(
#    "--root_file",
#    action="store",
#    type=str,
#    required=True,
#    help="Path to ROOT holding templates.",
#)
#parser.add_argument(
#    "--root_file_mu",
#    action="store",
#    type=str,
#    required=False,
#    help="Path to ROOT holding mu templates.",
#)
parser.add_argument(
    "--h_sensitivity", action="store_true", help="Just to run sensitivty check for H with toy 150 invfb data."
)
parser.add_argument(
    "--make_prefit_plot", action="store_true", help="Just to run prefit plot."
)
parser.add_argument(
    "--all_signals", action="store_true", help="Run on all signal templates."
)
parser.add_argument(
    "--muonCR", action="store_true", default=False, help="Add muonCR and top constraint."
)
parser.add_argument(
    "--scale_qcd",
    action="store_true",
    help="Scale QCD MC so its poisson matches true uncs.",
)
parser.add_argument(
    "--qcd_ftest", action="store_true", default=False, help="Run QCD ftest."
)
parser.add_argument(
    "--highbvl", action="store_true", default=False, help="Consider only highbvl."
)
parser.add_argument(
    "--lowbvl", action="store_true", default=False, help="Consider only highbvl."
)
parser.add_argument(
    "--ftest", action="store_true", default=False, help="Run ftest.")
parser.add_argument(
    "--pseudo", action="store_true", default=False, help="Run pseudo data."
)
parser.add_argument("--MCTF", action="store_true", help="Prefit the TF params to MC.")
parser.add_argument(
    "--do_systematics", action="store_true", help="Include systematics."
)
parser.add_argument(
    "--decorr_scale_pt", action="store_true", help="Decorrelate scale by pt bin."
)
parser.add_argument(
    "--decorr_scale_cat", action="store_true", help="Decorrelate scale highbvl and lowbvl."
)
parser.add_argument(
    "--decorr_scale_cat_pt", action="store_true", help="Decorrelate scale highbvl and lowbvl per pt bin."
)
parser.add_argument(
    "--mask_outlier", action="store_true", help="Mask outlying data point."
)
parser.add_argument(
    "--shift_sf_err", action="store", default=1., type=float, help="shift SF uncertainty."
)

# do_systematics = parser.add_mutually_exclusive_group(required=True)
    # pseudo.add_argument("--data", action="store_false", dest="pseudo")
    # pseudo.add_argument("--MC", action="store_true", dest="pseudo")
    # pseudo.add_argument("--toys", action="store_true", dest="toys")
    # parser.add_argument(
    #     "--clipx",
    #     type=str2bool,
    #     default="True",
    #     choices={True, False},
    #     help="Clip x-axis to range of data",
    # )

parser.add_argument("--is_blinded", action="store_true", help="Run on 10pct dataset.")
parser.add_argument("--throwPoisson", action="store_true", help="Throw poisson.")
parser.add_argument("--constrainedttbar", action="store_true", help="Treat top nuisances as constrained parameters (default is unconstrained).")
parser.add_argument(
    "--four_pt_bins", action="store_true", help="Sum the last two pt bins."
)
parser.add_argument("--tworeg", action="store_true", help="Two regs.")
parser.add_argument(
    "--year",
    action="store",
    type=str,
    help="Year to run on : one of 2016APV, 2016, 2017, 2018.",
)
parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
parser.add_argument("--debug", "-vv", action="store_true", help="Debug logging")

args = parser.parse_args()
# Arg processing
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

topSF = {
    "2016APV" : {
        "highbvl" : {
            "tqqeffSF_val" : 1.04,
            "tqqeffSF_unc" : 0.10,
            "tqqnormSF_val" : 1.02, #1.1062e+00,
            "tqqnormSF_unc" : 0.086, #9.12e-02,
         },

        "lowbvl" : {
            "tqqeffSF_val" : 0.898,
            "tqqeffSF_unc" : 0.088,
            "tqqnormSF_val" : 1.01,
            "tqqnormSF_unc" : 0.084,
         }

    },
    "2016" : {
        "highbvl" : {
            "tqqeffSF_val" : 0.93349,
            "tqqeffSF_unc" : 0.112,
            "tqqnormSF_val" : 1.1, #1.1062e+00,
            "tqqnormSF_unc" : 0.091, #9.12e-02,
         },

        "lowbvl" : {
            "tqqeffSF_val" : 0.54,
            "tqqeffSF_unc" : 0.078,
            "tqqnormSF_val" : 1.08,
            "tqqnormSF_unc" : 0.089,
         }
    },
    "2017" : {
        "highbvl" : {
            "tqqeffSF_val" : 0.956,
            "tqqeffSF_unc" : 0.083,
            "tqqnormSF_val" : 0.964, #1.1062e+00,
            "tqqnormSF_unc" : 0.085, #9.12e-02,
         },
        "lowbvl" : {
            "tqqeffSF_val" : 0.756,
            "tqqeffSF_unc" : 0.057,
            "tqqnormSF_val" : 0.93,
            "tqqnormSF_unc" : 0.082,
         }
    },
    "2018" : { 
        "highbvl" : {
            "tqqeffSF_val" : 1.16,
            "tqqeffSF_unc" : 0.074,
            "tqqnormSF_val" : 0.957,
            "tqqnormSF_unc" : 0.074,
         },
        "lowbvl" : {
            "tqqeffSF_val" : 0.779,
            "tqqeffSF_unc" : 0.049,
            "tqqnormSF_val" : 0.91,
            "tqqnormSF_unc" : 0.07,
         }
    },
    

}
SF = {

    "2016APV": {
        "BB_SF": 1,
        "BB_SF_ERR": 0.3,
        "W_SF": 1,
        "W_SF_ERR": 0.3,
        ### W CR >200 GeV
        #"V_SF": 0.799,
        #"V_SF_ERR": 0.060,
        #"SHIFT_SF": -0.350,
        #"SHIFT_SF_ERR": 0.475,
        #"SHIFT_SF_ERR": 1.475,
        #"SMEAR_SF": 1.099,
        #"SMEAR_SF_ERR":0.057,
        ###IDK what these are?
        #'V_SF': 0.635,
        #'V_SF_ERR': 0.116,
        #'SHIFT_SF': -0.599,
        #'SHIFT_SF_ERR': 0.664 +1.,
        #'SMEAR_SF': 1.228,
        #'SMEAR_SF_ERR': 0.37875
        ### W CR >400 GeV
        #'V_SF': 0.961,
        #'V_SF_ERR': 0.137,
        'V_SF' : 1,
        'V_SF_ERR' : 0.1,
        'SHIFT_SF' : 1.206,
        'SHIFT_SF_ERR' : 0.75 + args.shift_sf_err,
        'SMEAR_SF' : 1.115,
        'SMEAR_SF_ERR' : 0.122,
    },
    "2016": {
        "BB_SF": 1,
        "BB_SF_ERR": 0.3,
        "W_SF": 1,
        "W_SF_ERR": 0.3,
        ### W CR >200 GeV
        #"V_SF": 0.735,
        #"V_SF_ERR": 0.061,
        #"SHIFT_SF": -0.560,
        #"SHIFT_SF_ERR": 0.384,
        #"SHIFT_SF_ERR": 1.384,
        #"SMEAR_SF": 1.117,
        #"SMEAR_SF_ERR":0.046,
        ### W CR >400 GeV
        #"V_SF": 0.635,
        #"V_SF_ERR": 0.116,
        'V_SF' : 1,
        'V_SF_ERR' : 0.28,
        "SHIFT_SF": -0.559,
        "SHIFT_SF_ERR": 0.664 + args.shift_sf_err,
        "SMEAR_SF": 1.228,
        "SMEAR_SF_ERR": 0.329,
    },
    "2017": {
        "BB_SF": 1,
        "BB_SF_ERR": 0.3,
        "W_SF": 1,
        "W_SF_ERR": 0.3,
        #"V_SF": 0.827,
        #"V_SF_ERR": 0.042,
        #"SHIFT_SF": 0.417,
        #"SHIFT_SF_ERR": 0.395,
        #"SMEAR_SF": 1.011,
        #"SMEAR_SF_ERR":0.024,
        #"V_SF": 0.802,
        #"V_SF_ERR": 0.043,
        #"SHIFT_SF": 0.620,
        #"SHIFT_SF_ERR": 0.272,
        #"SHIFT_SF_ERR": 1.272,
        #"SMEAR_SF": 1.22,
        #"SMEAR_SF_ERR":0.112,
        ####7May
        #"V_SF" : 0.784,
        #"V_SF_ERR" : 0.04,
        #"SHIFT_SF": 0.681,
        #"SHIFT_SF_ERR": 1.42,
        #"SMEAR_SF": 1.126,
        #"SMEAR_SF_ERR": 0.11,
        ####10May
        #"V_SF" : 0.916,
        #"V_SF_ERR" : 0.091,
        'V_SF' : 1,
        'V_SF_ERR' : 0.1,
        #'SHIFT_SF': 0.286,
        #'SHIFT_SF_ERR': 0.491,
        #'SMEAR_SF': 0.640,
        #'SMEAR_SF_ERR': 0.614,
        'SHIFT_SF': 0.181,
        #'SHIFT_SF_ERR' : 0.423,
        'SHIFT_SF_ERR' : 0.423+args.shift_sf_err,
        'SMEAR_SF': 0.997,
        'SMEAR_SF_ERR': 0.053,


    },
    "2018": {
        "BB_SF": 1.,
        "BB_SF_ERR": 0.3, 
        "W_SF": 1.,
        "W_SF_ERR": 0.3,
        ####WTAG CR WITH INCLUSIVE PT
        #"V_SF": 0.770,
        #"V_SF_ERR": 0.034,
        #"SHIFT_SF": -0.504,
        #"SHIFT_SF_ERR": 0.117,
        #"SHIFT_SF_ERR": 1.117,
        #"SMEAR_SF": 1.034,
        #"SMEAR_SF_ERR":0.026,
        ###WTAG CR WITH PT>400 (10May24)
        #'V_SF': 0.827,
        #'V_SF_ERR': 0.072,
        ###NEW STRATEGY FOR 2PRONG EFF, USE 1 +- 1sig where sig is bound from low pt CR
        'V_SF' : 1,
        'V_SF_ERR' : 0.17,
        'SHIFT_SF': 0.184,
        'SHIFT_SF_ERR': 0.441 + args.shift_sf_err,
        'SMEAR_SF': 1.17725,
        'SMEAR_SF_ERR': 0.0435,
    },
}
tagger = args.tagger

era_dict = {
    "2016APV" : "2016preVFP",
    "2016" : "2016postVFP",
    "2017" : "2017",
    "2018" : "2018",
}
def badtemp_ma(hvalues, eps=0.0000001, mask=None):
    # Need minimum size & more than 1 non-zero bins
    tot = np.sum(hvalues[mask])
    count_nonzeros = np.sum(hvalues[mask] > 0)
    if (tot < eps) or (count_nonzeros < 5):
        return True
    else:
        return False

def smass(sName):
    if "hbb" in sName:
        _mass = 125.0
    elif sName in [
        "wqq",
        "tt",
        "st",
        "vv",
    ]:
        _mass = 80.0
    elif sName in ["zqq", "zcc", "zbb"]:
        _mass = 90.0
    elif sName in ["m"+str(s) for s in range(40,400,5)]:
        _mass = float(sName.split("m")[1])
    elif sName in ["zpqq"+str(s) for s in range(40,400,5)]:
        _mass = float(sName.split("zpqq")[1])
    elif sName in ["zpbb"+str(s) for s in range(40,400,5)]:
        _mass = float(sName.split("zpbb")[1])
    elif sName in ["p"+str(s) for s in range(40,400,5)]:
        _mass = float(sName.split("p")[1])
    elif sName in ["pbb"+str(s) for s in range(40,400,5)]:
        _mass = float(sName.split("pbb")[1])
    elif sName in ["b"+str(s) for s in range(40,400,5)]:
        _mass = float(sName.split("b")[1])
    else:
        raise ValueError("DAFUQ is {}".format(sName))
    return _mass


def flipSF(SF, SF_unc, yield_pass, yield_fail):
    """
    Return (SF, SF_unc) for a pass/fail scale factor.
    """
    sf = 1 - (yield_pass * (SF - 1) / yield_fail)
    sfup = 1 + (SF_unc * yield_pass / yield_fail) / sf
  
    #sfup = 1. - (SF_unc * yield_pass/yield_fail)/sf
    sfdown = 1/sfup
    logging.debug(f"sf={sf},sfup={sfup},sfdown={sfdown},yield_pass={yield_pass},yield_fail={yield_fail},SF={SF}")
    return sf, sfup, sfdown

with open("xsec.json") as f:
    xsec_dict = json.load(f)

short_to_long = {
    "wqq": "wqq",
    "wlnu": "wlnu",
    "tt": "tt",
    "st": "tt",
    "zqq": "zqq",
    "zbb": "zbb",
    "hbb": "hbb",    
    "dy": "dy",
    "vv": "vv",
    "m50": "zpqq50",
    "m55": "zpqq55",
    "m60": "zpqq60",
    "m65": "zpqq65",
    "m70": "zpqq70",
    "m75": "zpqq75",
    "m80": "zpqq80",
    "m85": "zpqq85",
    "m90": "zpqq90",
    "m95": "zpqq95",
    "m100": "zpqq100",
    "m105": "zpqq105",
    "m110": "zpqq110",
    "m115": "zpqq115",
    "m115": "zpqq115",
    "m120": "zpqq120",
    "m125": "zpqq125",
    "m130": "zpqq130",
    "m135": "zpqq135",
    "m140": "zpqq140",
    "m145": "zpqq145",
    "m150": "zpqq150",
    "m155": "zpqq155",
    "m160": "zpqq160",
    "m165": "zpqq165",
    "m170": "zpqq170",
    "m175": "zpqq175",
    "m180": "zpqq180",
    "m185": "zpqq185",
    "m190": "zpqq190",
    "m195": "zpqq195",
    "m200": "zpqq200",
    "m205": "zpqq205",
    "m210": "zpqq210",
    "m215": "zpqq215",
    "m220": "zpqq220",
    "m225": "zpqq225",
    "m230": "zpqq230",
    "m235": "zpqq235",
    "m240": "zpqq240",
    "m245": "zpqq245",
    "m250": "zpqq250",
    "m255": "zpqq255",
    "m260": "zpqq260",
    "m265": "zpqq265",
    "m270": "zpqq270",
    "m275": "zpqq275",
    "m280": "zpqq280",
    "m285": "zpqq285",
    "m290": "zpqq290",
    "m295": "zpqq295",
    "m300": "zpqq300",
    "b50": "zpbb50",
    "b55": "zpbb55",
    "b60": "zpbb60",
    "b65": "zpbb65",
    "b70": "zpbb70",
    "b75": "zpbb75",
    "b80": "zpbb80",
    "b85": "zpbb85",
    "b90": "zpbb90",
    "b95": "zpbb95",
    "b100": "zpbb100",
    "b105": "zpbb105",
    "b110": "zpbb110",
    "b115": "zpbb115",
    "b115": "zpbb115",
    "b120": "zpbb120",
    "b125": "zpbb125",
    "b130": "zpbb130",
    "b135": "zpbb135",
    "b140": "zpbb140",
    "b145": "zpbb145",
    "b150": "zpbb150",
    "b155": "zpbb155",
    "b160": "zpbb160",
    "b165": "zpbb165",
    "b170": "zpbb170",
    "b175": "zpbb175",
    "b180": "zpbb180",
    "b185": "zpbb185",
    "b190": "zpbb190",
    "b195": "zpbb195",
    "b200": "zpbb200",
    "b205": "zpbb205",
    "b210": "zpbb210",
    "b215": "zpbb215",
    "b220": "zpbb220",
    "b225": "zpbb225",
    "b230": "zpbb230",
    "b235": "zpbb235",
    "b240": "zpbb240",
    "b245": "zpbb245",
    "b250": "zpbb250",
    "b255": "zpbb255",
    "b260": "zpbb260",
    "b265": "zpbb265",
    "b270": "zpbb270",
    "b275": "zpbb275",
    "b280": "zpbb280",
    "b285": "zpbb285",
    "b290": "zpbb290",
    "b295": "zpbb295",
    "b300": "zpbb300",
    "p50": "pbb50",
    "p55": "pbb55",
    "p60": "pbb60",
    "p65": "pbb65",
    "p70": "pbb70",
    "p75": "pbb75",
    "p80": "pbb80",
    "p85": "pbb85",
    "p90": "pbb90",
    "p95": "pbb95",
    "p100": "pbb100",
    "p105": "pbb105",
    "p110": "pbb110",
    "p115": "pbb115",
    "p115": "pbb115",
    "p120": "pbb120",
    "p125": "pbb125",
    "p130": "pbb130",
    "p135": "pbb135",
    "p140": "pbb140",
    "p145": "pbb145",
    "p150": "pbb150",
    "p155": "pbb155",
    "p160": "pbb160",
    "p165": "pbb165",
    "p170": "pbb170",
    "p175": "pbb175",
    "p180": "pbb180",
    "p185": "pbb185",
    "p190": "pbb190",
    "p195": "pbb195",
    "p200": "pbb200",
    "p205": "pbb205",
    "p210": "pbb210",
    "p215": "pbb215",
    "p220": "pbb220",
    "p225": "pbb225",
    "p230": "pbb230",
    "p235": "pbb235",
    "p240": "pbb240",
    "p245": "pbb245",
    "p250": "pbb250",
    "p255": "pbb255",
    "p260": "pbb260",
    "p265": "pbb265",
    "p270": "pbb270",
    "p275": "pbb275",
    "p280": "pbb280",
    "p285": "pbb285",
    "p290": "pbb290",
    "p295": "pbb295",
    "p300": "pbb300",
}

sys_types = {
    "JES": "lnN",
    "JER": "lnN",
    "UES": "lnN",
    "jet_trigger": "lnN",
    "btagEffStat": "lnN",
    "btagWeight": "lnN",
    "pileup_weight": "lnN",
    "Z_d2kappa_EW": "lnN",
    "Z_d3kappa_EW": "lnN",
    "W_d2kappa_EW": "lnN",
    "W_d3kappa_EW": "lnN",
    "d1kappa_EW": "lnN",
    "d1K_NLO": "lnN",
    "d2K_NLO": "lnN",
    "d3K_NLO": "lnN",
    "L1Prefiring": "lnN",
    "scalevar_7pt": "lnN",
    "scalevar_3pt": "lnN",
    "muotrig": "lnN",
    "muoiso": "lnN",
    "muoid": "lnN",
    "HEMissue": "lnN",
    "btagSF_heavy_correlated" : "lnN",
    "btagSF_light_correlated" : "lnN",
    f"btagSF_heavy_{era_dict[args.year]}" : "lnN",
    f"btagSF_light_{era_dict[args.year]}" : "lnN",

}


sample_maps = {
    "QCD": [
        "QCD_HT500to700",
        "QCD_HT700to1000",
        "QCD_HT1000to1500",
        "QCD_HT1500to2000",
        "QCD_HT2000toInf",
    ],
    "wqq": ["WJetsToQQ_HT-600to800", "WJetsToQQ_HT-800toInf"],
    "zqq": ["ZJetsToQQ_HT-600to800", "ZJetsToQQ_HT-800toInf"],
    "tt": ["TTTo2L2Nu", "TTToHadronic", "TTToSemiLeptonic"],
    "dy": [
        "DYJetsToLL_Pt-200To400",
        "DYJetsToLL_Pt-400To600",
        "DYJetsToLL_Pt-600To800",
        "DYJetsToLL_Pt-800To1200",
        "DYJetsToLL_Pt-1200To2500",
    ],
    "st": [
        "ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
        "ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
    ],
    "hbb": ["GluGluHToBB"],
    "wlnu": [
        "WJetsToLNu_HT200to400",
        "WJetsToLNu_HT400to600",
        "WJetsToLNu_HT600to800",
        "WJetsToLNu_HT800to1200",
        "WJetsToLNu_HT1200to2500",
        "WJetsToLNu_HT2500toInf",
    ],
    "JetHT_2017": [
        "JetHT_Run2017B",
        "JetHT_Run2017C",
        "JetHT_Run2017D",
        "JetHT_Run2017E",
        "JetHT_Run2017F",
    ],
    "VectorZPrimeToQQ_M50": ["VectorZPrimeToQQ_M50"],
    "VectorZPrimeToQQ_M75": ["VectorZPrimeToQQ_M75"],
    "VectorZPrimeToQQ_M85": ["VectorZPrimeToQQ_M85"],
    "VectorZPrimeToQQ_M100": ["VectorZPrimeToQQ_M100"],
    "VectorZPrimeToQQ_M125": ["VectorZPrimeToQQ_M125"],
    "VectorZPrimeToQQ_M150": ["VectorZPrimeToQQ_M150"],
    "VectorZPrimeToQQ_M200": ["VectorZPrimeToQQ_M200"],
    "VectorZPrimeToQQ_M250": ["VectorZPrimeToQQ_M250"],
    # "VectorZPrimeToQQ_M300" : ["VectorZPrimeToQQ_M300"],
}

all_signals = [
        "m50",
        "m75",
        "m100",
        "m125",
        "m150",
        "m200",
        "m250",
]

signals = ["m" + args.sigmass]

    
poly_order = (args.ipt, args.irho)


def expo_sample(norm, scale, obs):
    cdf = scipy.stats.expon.cdf(scale=scale, x=obs.binning) * norm
    return (np.diff(cdf), obs.binning, obs.name)


def gaus_sample(norm, loc, scale, obs):
    cdf = scipy.stats.norm.cdf(loc=loc, scale=scale, x=obs.binning) * norm
    return (np.diff(cdf), obs.binning, obs.name)


def plot_mctf(tf_MCtempl, msdbins, name):
    """
    Plot the MC pass / fail TF as function of (pt,rho) and (pt,msd)
    """
    import matplotlib.pyplot as plt

    # arrays for plotting pt vs msd
    pts = np.linspace(525, 1200, 10)
    ptpts, msdpts = np.meshgrid(
        pts[:-1] + 0.5 * np.diff(pts),
        msdbins[:-1] + 0.5 * np.diff(msdbins),
        indexing="ij",
    )
    ptpts_scaled = (ptpts - 525.0) / (1200.0 - 525.0)
    rhopts = 2 * np.log(msdpts / ptpts)

    rhopts_scaled = (rhopts - (-5.5)) / ((-2.0) - (-5.5))
    validbins = (rhopts_scaled >= 0) & (rhopts_scaled <= 1)

    ptpts = ptpts[validbins]
    msdpts = msdpts[validbins]
    ptpts_scaled = ptpts_scaled[validbins]
    rhopts_scaled = rhopts_scaled[validbins]

    tf_MCtempl_vals = tf_MCtempl(ptpts_scaled, rhopts_scaled, nominal=True)
    # pprint(tf_MCtempl_vals)
    df = pd.DataFrame([])
    df["msd"] = msdpts.reshape(-1)
    df["pt"] = ptpts.reshape(-1)
    df["MCTF"] = tf_MCtempl_vals.reshape(-1)
    # pprint(df['MCTF'])
    fig, ax = plt.subplots()
    h = ax.hist2d(x=df["msd"], y=df["pt"], weights=df["MCTF"], bins=(msdbins, pts))
    plt.xlabel("$m_{sd}$ [GeV]")
    plt.ylabel("$p_{T}$ [GeV]")
    cb = fig.colorbar(h[3], ax=ax)
    cb.set_label("Ratio")
    fig.savefig(f"{opath}/MCTF_msdpt_" + name + ".png")
    fig.savefig(f"{opath}/MCTF_msdpt_" + name + ".pdf")
    plt.clf()

    # arrays for plotting pt vs rho
    rhos = np.linspace(-5.5, -2.0, 300)
    ptpts, rhopts = np.meshgrid(
        pts[:-1] + 0.5 * np.diff(pts), rhos[:-1] + 0.5 * np.diff(rhos), indexing="ij"
    )
    ptpts_scaled = (ptpts - 525.0) / (1500.0 - 525.0)
    rhopts_scaled = (rhopts - (-5.5)) / ((-2.0) - (-5.5))
    validbins = (rhopts_scaled >= 0) & (rhopts_scaled <= 1)

    ptpts = ptpts[validbins]
    rhopts = rhopts[validbins]
    ptpts_scaled = ptpts_scaled[validbins]
    rhopts_scaled = rhopts_scaled[validbins]

    tf_MCtempl_vals = tf_MCtempl(ptpts_scaled, rhopts_scaled, nominal=True)

    df = pd.DataFrame([])
    df["rho"] = rhopts.reshape(-1)
    df["pt"] = ptpts.reshape(-1)
    df["MCTF"] = tf_MCtempl_vals.reshape(-1)

    fig, ax = plt.subplots()
    h = ax.hist2d(x=df["rho"], y=df["pt"], weights=df["MCTF"], bins=(rhos, pts))
    plt.xlabel("rho")
    plt.ylabel("$p_{T}$ [GeV]")
    cb = fig.colorbar(h[3], ax=ax)
    cb.set_label("Ratio")
    fig.savefig(
        f"{opath}/MCTF_rhopt_" + name + ".png",
    )
    fig.savefig(
        f"{opath}/MCTF_rhopt_" + name + ".pdf",
    )

    return

#isinterp = signals[0] in ["m"+str(x) for x in range(40,350,5)] and signals[0] not in all_signals
#if isinterp:
#    log.debug(f"m {signals[0]} is interpolated signal")


hist_files = {
    #"2016APV" : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/22Jun24-2016APV/results/TEMPLATES__0624.root",
    #"2016" : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/22Jun24-2016/results/TEMPLATES__0624.root",
    #"2017" : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/22Jun24-2017/results/TEMPLATES__0624.root",
    #"2018" : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/22Jun24-2018/results/TEMPLATES__0624.root",
    "2016APV" : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/el9/9Jul24-2016APV-SR-2/results/TEMPLATES_v0_0711.root",
    "2016" : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/el9/9Jul24-2016-SR/results/TEMPLATES_v0_0711.root",
    "2017" : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/el9/9Jul24-2017-SR/results/TEMPLATES_v0_0711.root",
    "2018" : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/el9/9Jul24-2018-SR/results/TEMPLATES_v0_0711.root",
}

hist_signal_files = {
    #"2016APV" : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/22Jun24-2016APV/results/TEMPLATES__0624_interpolated_.root",
    #"2016" : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/22Jun24-2016/results/TEMPLATES__0624_interpolated_.root",
    #"2017" : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/22Jun24-2017/results/TEMPLATES__0624_interpolated_.root",
    "2016APV" : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/el9/9Jul24-2016APV-SR-2/results/TEMPLATES_v0_0711_interpolated_v0.root",
    "2016" : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/el9/9Jul24-2016-SR/results/TEMPLATES_v0_0711_interpolated_v0.root",
    "2017" : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/el9/9Jul24-2017-SR/results/TEMPLATES_v0_0711_interpolated_v0.root",
    "2018" : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/el9/9Jul24-2018-SR/results/TEMPLATES_v0_0711_interpolated_v0.root",
}

hist_mucr_files = {
    "2016APV" : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/19Apr24-2016APV-CR1/results/TEMPLATES_30May24.root",
    "2016" : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/19Apr24-2016-CR1/results/TEMPLATES_30May24.root",
    "2017" : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/19Apr24-2017-CR1/results/TEMPLATES_30May24.root",
    "2018" : "/eos/project/c/contrast/public/cl/www/zprime/bamboo/19Apr24-2018-CR1/results/TEMPLATES_30May24.root",
}
root_file_signals = uproot.open(hist_signal_files[args.year]) #uproot.open(args.root_file.replace(".root","_interpolated.root"))
root_file_mu = uproot.open(hist_mucr_files[args.year])
root_file = uproot.open(hist_files[args.year])
def get_templ(
    region,
    sample,
    ptbin,
    tagger,
    syst=None,
    muon=False,
    nowarn=False,
    year="2017",
    scaledown=False,
    #scaleup=False,
    fourptbins=False,
    observable=None,
):
    if muon:
        hist_str = f"CR1_ptbin5_{sample}_{tagger}_{region}"
    else:
        hist_str = f"SR_{sample}_ptbin{ptbin}_{tagger}_{region}"
    if syst is not None:
        hist_str = hist_str + "__" + syst
    if muon:
        hist = root_file_mu[hist_str]
    elif "zpqq" in sample or "zpbb" in sample or "pbb" in sample:
        hist = root_file_signals[hist_str]
    else:
        hist = root_file[hist_str] 
    hist_values = hist.values()
    hist_variances = hist.variances()
    hist_edges = hist.axis().edges()
    if scaledown:
        hist_values *= 1e-3

    if fourptbins and ptbin == 3:
        with uproot.open(args.root_file) as f:
            # pprint(f.keys())
            hist = f[hist_str.replace("ptbin3", "ptbin4")]
        hist_values += hist.values()
        hist_variances += hist.variances()

    if np.any(~np.isfinite(hist_values)):
        _invalid = ~np.isfinite(hist_values)
        hist_values[ _invalid ] = 0.
        hist_variances[ _invalid ] = 0.
        log.warning(f"Some invalid values in region={region}, sample={sample}, ptbin={ptbin}, syst={syst}, muon={muon} template; setting those to zero")
    if np.any(hist_values < 0.0):
        _invalid = hist_values < 0.
        hist_values[ _invalid ] = 0.
        hist_variances[ _invalid ] = 0.
        log.warning(f"Some negative values in region={region}, sample={sample}, ptbin={ptbin}, syst={syst}, muon={muon} template; setting those to zero")
    if observable is not None:
        if observable.nbins != len(hist_values): #hotfix for muon nbins issue differing in pass/fail
            log.warning(f"Observable {observable.name} has nbins = {observable.nbins}, whereas provided array has {len(hist_values)}")
            hist_values = hist_values[:observable.nbins]
            hist_edges = hist_edges[:observable.nbins+1]
            hist_variances = hist_variances[:observable.nbins] 
    
    if muon:
        hist_key = "msd_muon"
    else:
        hist_key = "msd"
  
    log.debug(f"sample={sample} region={region} ptbin={ptbin} hist_values={hist_values}")
    return (hist_values, hist_edges, hist_key, hist_variances)

def one_bin(template):
    try:
        h_vals, h_edges, h_key, h_variances = template
        return (np.array([np.sum(h_vals)]), np.array([0., 1.]), "onebin", np.array([np.sum(h_variances)]))
    except:
        h_vals, h_edges, h_key = template
        return (np.array([np.sum(h_vals)]), np.array([0., 1.]), "onebin")


def th1_to_numpy(path, label="msd"):
    with uproot.open(path) as file:
        th1d = file[label]
        _hist, _ = th1d.to_numpy()
    return _hist


def shape_to_num(
    region, sName, ptbin, syst_down_up, mask, muon=False, bound=0.5, inflate=False, scaledown=False, 
):
    # pprint(sName)

    _nom = get_templ(region, sName, ptbin, tagger, muon=muon,scaledown=scaledown,)#scaleup=scaleup,)
    # _nom = th1_to_numpy(path)
    #print(sName,syst_down_up,"A")
    # if template is very small don't add unc
    if _nom[0] is None:
        return None
    if muon:
        mask = np.ones_like(_nom[0])
    #print(sName,syst_down_up,"B")
    _nom_rate = np.sum(_nom[0] * mask)
    #if "zpqq" in sName or "zpbb" in sName:
    #    _nom_rate /= 2.066
    if _nom_rate < 0.1:
        return 1.0
    # ignore one sided for now
    _one_side = (
        None  # get_templ(f, region, sName, ptbin, syst=syst, muon=muon, nowarn=True)
    )

    #print(sName,syst_down_up,"C")
    syst_name_up = syst_down_up[1]
    syst_name_down = syst_down_up[0]
    if "2016" in syst_name_up or "2017" in syst_name_up or "2018" in syst_name_up:
        syst_name_up = syst_name_up.replace(args.year,era_dict[args.year])
        syst_name_down = syst_name_down.replace(args.year,era_dict[args.year])

    _up = get_templ(
        region, sName, ptbin, tagger, syst=syst_name_up, muon=muon, nowarn=True, scaledown=scaledown, #scaleup=scaleup,
    )

    # _up = th1_to_numpy(path)

    _down = get_templ(
        region, sName, ptbin, tagger, syst=syst_name_down, muon=muon, nowarn=True, scaledown=scaledown,#scaleup=scaleup,
    )
    # _down = th1_to_numpy(path)
    #print(sName,syst_down_up,"D")
    if _up is None and _down is None and _one_side is None:
        return None
    else:
        if _one_side is not None:
            _up_rate = np.sum(_one_side[0] * mask)
            _diff = np.abs(_up_rate - _nom_rate)
            magnitude = _diff / _nom_rate
        elif _down[0] is not None and _up[0] is not None:
            _up_rate = np.sum(_up[0] * mask)
            _down_rate = np.sum(_down[0] * mask)
            # pprint("_up_rate",_up_rate)
            # pprint("_down_rate",_down_rate)
            _diff = np.abs(_up_rate - _nom_rate) + np.abs(_down_rate - _nom_rate)
            magnitude = _diff / (2.0 * _nom_rate)
            log.debug(f"sample={sName}, syst={syst_down_up[0]} downrate={_down_rate}, syst={syst_down_up[1]} uprate={_up_rate}, magnitude={magnitude}, nomrate={_nom_rate}")
        else:
            raise NotImplementedError
    #print(sName,syst_down_up,"E")
    if bound is not None:
        magnitude = min(magnitude, bound)
    # inflate uncs while debugging what went wrong
    if inflate:
        log.warning("Inflating uncs!")
        magnitude *= 10
    # pprint(magnitude)
    return 1.0 + magnitude


def test_rhalphabet(tmpdir, sig, throwPoisson=False):
    sys_shape_dict = {}
    sys_shape_dict["JES"] = rl.NuisanceParameter(
        "CMS_scale_j_{}".format(args.year), sys_types["JES"]
    )
    sys_shape_dict["JER"] = rl.NuisanceParameter(
        "CMS_res_j_{}".format(args.year), sys_types["JER"]
    )
    # don't have UES for now
    sys_shape_dict["UES"] = rl.NuisanceParameter(
        "CMS_ues_j_{}".format(args.year), sys_types["UES"]
    )
    sys_shape_dict["jet_trigger"] = rl.NuisanceParameter(
        "CMS_trigger_{}".format(args.year), sys_types["jet_trigger"]
    )
    sys_shape_dict["L1Prefiring"] = rl.NuisanceParameter(
        "CMS_L1prefire_{}".format(args.year), sys_types["L1Prefiring"]
    )

    sys_shape_dict["pileup_weight"] = rl.NuisanceParameter(
        "CMS_PU_{}".format(args.year), sys_types["pileup_weight"]
    )
    # don't have HEM for now
    sys_shape_dict["HEMissue"] = rl.NuisanceParameter(
        "CMS_HEM_{}".format(args.year), sys_types["HEMissue"]
    )
    # don't have mu for now
    sys_shape_dict["muotrig"] = rl.NuisanceParameter(
        "CMS_mu_trigger_{}".format(args.year), sys_types["muotrig"]
    )
    sys_shape_dict["muoiso"] = rl.NuisanceParameter(
        "CMS_mu_isoweight_{}".format(args.year), sys_types["muoiso"]
    )
    sys_shape_dict["muoid"] = rl.NuisanceParameter(
        "CMS_mu_idweight_{}".format(args.year), sys_types["muoid"]
    )
    sys_shape_dict["btagSF_heavy_correlated"] = rl.NuisanceParameter(
        "CMS_btagSF_heavy_correlated", sys_types["btagSF_heavy_correlated"]
    )
    sys_shape_dict["btagSF_light_correlated"] = rl.NuisanceParameter(
        "CMS_btagSF_light_correlated", sys_types["btagSF_light_correlated"]
    )
    sys_shape_dict[f"btagSF_heavy_{era_dict[args.year]}"] = rl.NuisanceParameter(
        "CMS_btagSF_heavy_{}".format(era_dict[args.year]), sys_types[f"btagSF_heavy_{era_dict[args.year]}"]
    )
    sys_shape_dict[f"btagSF_light_{era_dict[args.year]}"] = rl.NuisanceParameter(
        "CMS_btagSF_light_{}".format(era_dict[args.year]), sys_types[f"btagSF_light_{era_dict[args.year]}"]
    )

    for EW_syst in ['d1kappa_EW', 'W_d2kappa_EW', 'W_d3kappa_EW','Z_d2kappa_EW', 'Z_d3kappa_EW', 'd1K_NLO', 'd2K_NLO', 'd3K_NLO']:
        sys_shape_dict[EW_syst] = rl.NuisanceParameter('CMS_{}'.format(EW_syst), sys_types[EW_syst])

    # sys_shape_dict['scalevar_7pt'] = rl.NuisanceParameter('CMS_th_scale7pt', sys_types['scalevar_7pt'])
    # sys_shape_dict['scalevar_3pt'] = rl.NuisanceParameter('CMS_th_scale3pt', sys_types['scalevar_3pt'])

    sys_eleveto = rl.NuisanceParameter("CMS_e_veto_{}".format(args.year), "lnN")
    sys_muveto = rl.NuisanceParameter("CMS_m_veto_{}".format(args.year), "lnN")
    sys_tauveto = rl.NuisanceParameter("CMS_tau_veto_{}".format(args.year), "lnN")

    sys_veff = rl.NuisanceParameter("CMS_PNet_2prong_{}".format(args.year), "lnN")
    sys_bbeff = rl.NuisanceParameter("CMS_PNet_bb_{}".format(args.year), "lnN")
    sys_Weff = rl.NuisanceParameter("CMS_PNet_W_{}".format(args.year), "lnN")

    sys_lumi = rl.NuisanceParameter("CMS_lumi_{}".format(args.year), "lnN")
    sys_lumi_correlated = rl.NuisanceParameter("CMS_lumi_13TeV_correlated", "lnN")
    sys_lumi_1718 = rl.NuisanceParameter("CMS_lumi_13TeV_1718", "lnN")

    if args.decorr_scale_cat:
        sys_shape_dict[f"CMS_scale_{args.year}_highbvl"] = rl.NuisanceParameter(
                f"CMS_scale_{args.year}_highbvl", 'shape'
        )
        sys_shape_dict[f"CMS_scale_{args.year}_lowbvl"] = rl.NuisanceParameter(
                f"CMS_scale_{args.year}_lowbvl", 'shape'
        )
        sys_shape_dict[f"CMS_scale_{args.year}_fail"] = rl.NuisanceParameter(
                f"CMS_scale_{args.year}_fail", 'shape'
        )

    elif args.decorr_scale_cat_pt:
        for iptbin in range(0,5):
            sys_shape_dict[f"CMS_scale_{args.year}_highbvl_ptbin{iptbin}"] = rl.NuisanceParameter(
                f"CMS_scale_{args.year}_highbvl_ptbin{iptbin}", 'shape'
            )
            sys_shape_dict[f"CMS_scale_{args.year}_lowbvl_ptbin{iptbin}"] = rl.NuisanceParameter(
                f"CMS_scale_{args.year}_lowbvl_ptbin{iptbin}", 'shape'
            )
    elif args.decorr_scale_pt:
        for iptbin in range(0,5):
            sys_shape_dict[f"CMS_scale_{args.year}_ptbin{iptbin}"] = rl.NuisanceParameter(
                f"CMS_scale_{args.year}_ptbin{iptbin}", 'shape'
            )
    else:
        sys_scale = rl.NuisanceParameter('CMS_scale_{}'.format(args.year), 'shape')
    sys_smear = rl.NuisanceParameter('CMS_smear_{}'.format(args.year), 'shape')
    #print("Systs being considered =", sys_shape_dict.keys())
    tqqeffHLSF = rl.IndependentParameter("tqqeffHLSF_{year}".format(year=args.year), 1.0, 0, 10)
    tqqeffSF_highbvl = rl.IndependentParameter("tqqeffSF_highbvl_{year}".format(year=args.year), 1.0, 0, 10)
    tqqnormSF_highbvl = rl.IndependentParameter("tqqnormSF_highbvl_{year}".format(year=args.year), 1.0, 0, 10)
    tqqeffSF_lowbvl = rl.IndependentParameter("tqqeffSF_lowbvl_{year}".format(year=args.year), 1.0, 0, 10)
    tqqnormSF_lowbvl = rl.IndependentParameter("tqqnormSF_lowbvl_{year}".format(year=args.year), 1.0, 0, 10)

    if args.constrainedttbar:
        tqqeffSF = rl.NuisanceParameter('tqqeffSF_{year}'.format(year=args.year), 'lnN')
        tqqnormSF = rl.NuisanceParameter('tqqnormSF_{year}'.format(year=args.year), 'lnN')
    else:
        tqqeffSF = rl.IndependentParameter("tqqeffSF_{year}".format(year=args.year), 1.0, 0, 10)
        tqqnormSF = rl.IndependentParameter("tqqnormSF_{year}".format(year=args.year), 1.0, 0, 10)

    # with open(args.pickle, "rb") as f:
    #    df = pickle.load(f)
    if "2017" in args.year: 
        ptmin=525.0
        ptbins = np.array([525, 575, 625, 700, 800, 1200])
    else:
        ptmin=500.
        ptbins = np.array([500,550,600,700,800,1200])
    if args.four_pt_bins:
        ptbins = np.array([525, 575, 625, 700, 1200])
    npt = len(ptbins) - 1
    msdbins = np.linspace(40, 350, 63)
    msd = rl.Observable("msd", msdbins)
    msdbins_muon = np.linspace(40, 240 , 41)
    msd_muon = rl.Observable("msd_muon", msdbins_muon)

    # here we derive these all at once with 2D array
    ptpts, msdpts = np.meshgrid(
        ptbins[:-1] + 0.3 * np.diff(ptbins),
        msdbins[:-1] + 0.5 * np.diff(msdbins),
        indexing="ij",
    )
    rhopts = 2 * np.log(msdpts / ptpts)
    ptscaled = (ptpts - ptmin) / (1200.0 - ptmin)
    rhoscaled = (rhopts - (-5.5)) / ((-2.0) - (-5.5))
    validbins = (rhoscaled >= 0.0) & (rhoscaled <= 1.0)
    rhoscaled[~validbins] = 1  # we will mask these out later

    ####################################################
    # MCTF Setups
    if args.tworeg:
        # pass,fail,pass_highbvl,pass_lowbvl
        qcdmodels = [rl.Model("qcdmodel0"), rl.Model("qcdmodel1")]
        qcd_counts_pass = [0.0, 0.0]
        qcd_counts_fail = [0.0, 0.0]
        pass_regs = ["pass_T_bvl_pass_L", "pass_T_bvl_fail_L"]
        fail_regs = ["fail_T", "fail_T"]
 
        if args.qcd_ftest or args.ftest:
            if args.highbvl:
                qcdmodels, qcd_counts_pass, qcd_counts_fail, pass_regs, fail_regs = qcdmodels[0:1], qcd_counts_pass[0:1], qcd_counts_fail[0:1], pass_regs[0:1], fail_regs[0:1]
            elif args.lowbvl:
                qcdmodels, qcd_counts_pass, qcd_counts_fail, pass_regs, fail_regs = qcdmodels[1:2], qcd_counts_pass[1:2], qcd_counts_fail[1:2], pass_regs[1:2], fail_regs[1:2]
            else:
                raise RuntimeError ("You must supply either highbvl or lowbvl to args.")
    else:
        qcdmodels = [rl.Model("qcdmodel")]
        qcd_counts_pass = [0.0]
        qcd_counts_fail = [0.0]
        pass_regs = ["pass"]
        fail_regs = ["fail"]

    qcdeffs = [] 
    #from Collections import defaultdict
    #scale_qcd = defaultdict(list)
    it = 0 
    for qcdmodel, qcdpass, qcdfail, rpass, rfail in zip(
        qcdmodels, qcd_counts_pass, qcd_counts_fail, pass_regs, fail_regs
    ):
        for ptbin in range(npt):
            failCh = rl.Channel("ptbin%d%s" % (ptbin, "fail"))
            passCh = rl.Channel("ptbin%d%s" % (ptbin, "pass"))
            failTempl = get_templ(
                rfail, "QCD", ptbin, tagger, fourptbins=args.four_pt_bins
            )
            passTempl = get_templ(
                rpass, "QCD", ptbin, tagger, fourptbins=args.four_pt_bins
            )
            failCh.setObservation(failTempl, read_sumw2=True)
            passCh.setObservation(passTempl, read_sumw2=True)
            qcdfail += failCh.getObservation()[0].sum()
            qcdpass += passCh.getObservation()[0].sum()
            if args.MCTF:
                qcdmodel.addChannel(failCh)
                qcdmodel.addChannel(passCh)
            if args.make_prefit_plot:
                values, _, _, _ = get_templ(rpass, f"JetHT_2017", ptbin, tagger, fourptbins=args.four_pt_bins)
                #scale_qcd[f"{pass}"].append(values.sum()/qcdpass)
        qcdeffs.append(qcdpass / qcdfail)
        it += 1

    if args.MCTF:
        l_iptMC = args.iptMC.split(",")
        l_irhoMC = args.irhoMC.split(",")
        tf_MCtempls = []
        tf_MCtempl_params = []
        for i, qcdmodel in enumerate(qcdmodels):
            #degsMC = tuple([int(s) for s in [args.iptMC, args.irhoMC]])
            degpt, degrho = l_iptMC[i], l_irhoMC[i]
            if args.ftest:  #in ftest we treat one region at a time, the degree args are (high,low) for both pt and rho
                if args.highbvl: degpt, degrho = l_iptMC[0], l_irhoMC[0]
                elif args.lowbvl: degpt, degrho = l_iptMC[1], l_irhoMC[1]
            degsMC = tuple([int(s) for s in [degpt, degrho]])
            _initsMC = np.ones(tuple(n + 1 for n in degsMC))
            log.debug(f"Initializing MCTF `{qcdmodel.name}` with n_pt={degpt} and n_rho={degrho}")
            log.debug(_initsMC)
            tf_MCtempl = rl.BasisPoly(
                f"tf{args.year}_MC_{i}templ",
                degsMC,
                ["pt", "rho"],
                basis="Bernstein",
                init_params=_initsMC,
                limits=(-50, 50),
                coefficient_transform=None,
            )
            tf_MCtempls.append(tf_MCtempl)
            tf_MCtempl_params.append(qcdeffs[i] * tf_MCtempl(ptscaled, rhoscaled))
            for ptbin in range(npt):
                failCh = qcdmodel[f"ptbin{ptbin}fail"]
                passCh = qcdmodel[f"ptbin{ptbin}pass"]
                failObs = failCh.getObservation()[0]
                qcdparams = np.array(
                    [
                        rl.IndependentParameter(f"qcdparam{i}_ptbin{ptbin}_msdbin{mbin}".format(i=i,ptbin=ptbin,mbin=mbin,), 0)
                        for mbin in range(msd.nbins)
                    ]
                )
                sigmascale = 10.0
                scaledparams = (
                    failObs
                    * (1 + sigmascale / np.maximum(1.0, np.sqrt(failObs))) ** qcdparams
                )
                fail_qcd = rl.ParametericSample(
                    f"ptbin{ptbin}fail_qcd", rl.Sample.BACKGROUND, msd, scaledparams
                )
                failCh.addSample(fail_qcd)
                pass_qcd = rl.TransferFactorSample(
                    f"ptbin{ptbin}pass_qcd",
                    rl.Sample.BACKGROUND,
                    tf_MCtempl_params[i][ptbin, :],
                    fail_qcd,
                )
                passCh.addSample(pass_qcd)
                failCh.mask = validbins[ptbin]
                passCh.mask = validbins[ptbin]
        
        # Run fits
        MCTFs = []
        log.info("Fitting MCTFs")
        for i, (qcdmodel, tf_MCtempl, tf_MCtempl_param) in enumerate(zip(qcdmodels, tf_MCtempls, tf_MCtempl_params)):  
            qcdfit_ws = ROOT.RooWorkspace("qcdfit_ws")
            simpdf, obs = qcdmodel.renderRoofit(qcdfit_ws)
            qcdfit = simpdf.fitTo(
                obs,
                ROOT.RooFit.Extended(True),
                ROOT.RooFit.Offset(True),  # Gets rid of negative value warnings
                ROOT.RooFit.SumW2Error(True),
                ROOT.RooFit.Strategy(2),
                ROOT.RooFit.Save(),
                ROOT.RooFit.Minimizer("Minuit2", "migrad"),
                ROOT.RooFit.Verbose(0),
                ROOT.RooFit.PrintLevel(-1),
            )
            qcdfit_ws.add(qcdfit)
            MCTFs.append(qcdfit)
            
            # Set parameters to fitted values
            allparams = dict(zip(qcdfit.nameArray(), qcdfit.valueArray()))
            pvalues = []
            for i, p in enumerate(tf_MCtempl.parameters.reshape(-1)):
                p.value = allparams[p.name]
            pvalues += [p.value]
            
        # Show res
        for model, fit_result, tf_MCtempl in zip(qcdmodels, MCTFs, tf_MCtempls):
            if not (fit_result.status() == 0 or qcdfit.status() == 1):
                raise RuntimeError("Could not fit qcd")
            log.info(f"MCTF - {model.name} fit status: {fit_result.status()}")
            _values = [par.value for par in tf_MCtempl.parameters.flatten()]
            _names = [par.name for par in tf_MCtempl.parameters.flatten()]
            for name, value in zip(_names, _values):
                log.debug(f"{name} = {value:.3f}")
                
            if "pytest" not in sys.modules:  #idk
                qcdfit_ws.writeToFile(os.path.join(str(tmpdir), f"testModel_{model.name}.root"))


        # plot_mctf(tf_MCtempl, msdbins, "all")
        # Decorrelate
        all_tf_MCtempl_params_final = []
        for model, fit_result, tf_MCtempl in zip(qcdmodels, MCTFs, tf_MCtempls):
            param_names = [p.name for p in tf_MCtempl.parameters.reshape(-1)]
            decoVector = rl.DecorrelatedNuisanceVector.fromRooFitResult(
                tf_MCtempl.name + "_deco", fit_result, param_names
            )
            tf_MCtempl.parameters = decoVector.correlated_params.reshape(
                tf_MCtempl.parameters.shape
            )
            tf_MCtempl_params_final = tf_MCtempl(ptscaled, rhoscaled)
            all_tf_MCtempl_params_final.append(tf_MCtempl_params_final)
    
    #####################################
    # Data TF
    all_tf_params = []
    #nTFs = 2 if args.tworeg else 1
    nTFs = len(pass_regs)
    for i in range(nTFs):
        l_ipt = args.ipt.split(",")
        l_irho = args.irho.split(",")
        degpt, degrho = l_ipt[i], l_irho[i]
        if args.ftest:  #in ftest we treat one region at a time, the degree args are (high,low) for both pt and rho
            if args.highbvl: degpt, degrho = l_ipt[0], l_irho[0]
            elif args.lowbvl: degpt, degrho = l_ipt[1], l_irho[1]
        degs = tuple([int(s) for s in [degpt, degrho]])
        _inits = np.ones(tuple(n + 1 for n in degs))
        log.debug(f"Initializing TF {i} (data) with n_pt={degpt} and n_rho={degrho}")
        log.debug(_inits)
        tf_dataResidual = rl.BasisPoly(
            f"tf{args.year}_dataResidual_{i}",
            degs,
            ["pt", "rho"],
            basis="Bernstein",
            init_params=_inits,
            limits=(-10, 10),
            coefficient_transform=None,
        )
        # tf_dataResidual = rl.BernsteinPoly("tf_dataResidual", poly_order, ["pt", "rho"], init_params=_inits, limits=(0, 10))
        tf_dataResidual_params = tf_dataResidual(ptscaled, rhoscaled)

        if args.MCTF:
            tf_params = qcdeffs[i] * all_tf_MCtempl_params_final[i] * tf_dataResidual_params
            if args.make_prefit_plot:
                tf_params *= 1/np.mean(scale_qcd[i])
        else:
            tf_params = qcdeffs[i] * tf_dataResidual_params
            if args.make_prefit_plot:
                tf_params *= 1/np.mean(scale_qcd[i])
        all_tf_params.append(tf_params)
        
    ##############################
    # Build main model
    log.info("Building main model")   
    model = rl.Model(f"{sig}_model")
    siggy = sig
    bsiggy = sig.replace("m", "b")
    psiggy = sig.replace("m", "p")
    if not (args.ftest or args.qcd_ftest):
        log.info(f"Signals: {siggy}, {bsiggy}, {psiggy}")
        model.t2w_config = ("-P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel  --PO verbose "
                f"--PO 'map=.*/*{siggy}:r_q[1,-5,5]'  --PO 'map=.*/*{bsiggy}:r_b[1,-5,5]' --PO 'map=.*/*{psiggy}:r_p[1,-5,5]'"
                        )
    #else:
    #    log.info(f"One signal")
    #    model.t2w_config = ("-P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel  --PO verbose "
    #                    f"--PO 'map=.*/*{args.sigmass}:r[1,-5,5]'"
    #                    )

    scale_pass = []
    
    for ptbin in range(npt):
        # for region in ["pass", "fail"]:
        all_regs = pass_regs + fail_regs[:1] #only look at one (common) fail region 
        #for region in list(set(pass_regs+fail_regs)):
        for region in sorted(all_regs, key=lambda x: (not x.startswith("pass"), x)):
            model_reg_name = region.replace("_", "")         
            ch = rl.Channel(f"ptbin{ptbin}{model_reg_name}")
            log.debug(f"Initializing model region: ptbin{ptbin}{model_reg_name}")   
            model.addChannel(ch)

            # Fetch templates
            templates = {
                "wqq": get_templ(
                    region, "wqq", ptbin, tagger, fourptbins=args.four_pt_bins
                ),
                "zqq": get_templ(
                    region,
                    "zqq",
                    ptbin,
                    tagger,
                    fourptbins=args.four_pt_bins,
                    scaledown=True if args.pseudo else False,
                ),
                "zbb": get_templ(
                    region,
                    "zbb",
                    ptbin,
                    tagger,
                    fourptbins=args.four_pt_bins,
                    scaledown=True if args.pseudo else False,
                ),
                "tt": get_templ(
                    region, "tt", ptbin, tagger, fourptbins=args.four_pt_bins
                ),
                "vv": get_templ(
                    region, "vv", ptbin, tagger, fourptbins=args.four_pt_bins
                ),
                "wlnu": get_templ(
                    region, "wlnu", ptbin, tagger, fourptbins=args.four_pt_bins
                ),
                "dy": get_templ(
                    region, "dy", ptbin, tagger, fourptbins=args.four_pt_bins
                ),
                "st": get_templ(
                    region, "st", ptbin, tagger, fourptbins=args.four_pt_bins
                ),
                "hbb": get_templ(
                    region, "hbb", ptbin, tagger, fourptbins=args.four_pt_bins
                ),
                "qcd": get_templ(
                    region, "QCD", ptbin, tagger, fourptbins=args.four_pt_bins
                ),
            }

            templates[siggy] = get_templ(
                    region,
                    short_to_long[siggy],
                    ptbin,
                    tagger,
                    fourptbins=args.four_pt_bins,
                    scaledown=True if args.ftest else False,
                    #scaleup=True,
                )
            #if not (args.ftest or args.qcd_ftest):
            templates[bsiggy] = get_templ(
                    region,
                    short_to_long[bsiggy],
                    ptbin,
                    tagger,
                    fourptbins=args.four_pt_bins,
                    #scaleup=True,
            )
            templates[psiggy] = get_templ(
                    region,
                    short_to_long[psiggy],
                    ptbin,
                    tagger,
                    fourptbins=args.four_pt_bins,
                    #scaleup=False,
            )
            mask = validbins[ptbin].copy()
            
            if args.qcd_ftest:
                include_samples = [siggy]  # qcd here?
            elif args.ftest or args.h_sensitivity:
                include_samples = ["wqq", "zqq", "zbb", "tt", "wlnu", "dy", "st", "hbb", "vv", siggy]
            else:
                include_samples = ["wqq", "zqq", "zbb", "tt", "wlnu", "dy", "st", "hbb", "vv", siggy, bsiggy, psiggy]
            
            for sName in include_samples:
                # some mock expectations
                templ = templates[sName]
                logging.info(f"Adding sample={sName} in ptbin={ptbin}, region={region}.")
                if badtemp_ma(templ[0]):
                    logging.info(f"Sample has bad template: {templ[0]}. Filling zeros.")
                    templ = (np.zeros_like(templ[0]),templ[1],templ[2],templ[3])
                    logging.info(f"After filling: {templ[0]}.")
                    ##continue
                #if args.qcd_(ftest:
                #    stype = rl.Sample.SIGNAL if sName == "zqq" else rl.Sample.BACKGROUND
                #    # templ[0] = templ[0]*1e-4 #Scale down signal?
                #if args.ftest: #qcd_ftest and ftest here for now
                #    stype = rl.Sample.SIGNAL if sName in ["zqq"] else rl.Sample.BACKGROUND
                #else:
                stype = rl.Sample.SIGNAL if sName in [siggy, bsiggy, psiggy] else rl.Sample.BACKGROUND
                sample = rl.TemplateSample(ch.name + "_" + sName, stype, templ, force_positive=True)

                def smorph(templ, sName):
                    if templ is None:
                        return None
                    if sName not in ["qcd", "dy", "wlnu", "tt", "st"]:
                        return MorphHistW2(templ).get(
                            shift=SF[args.year]['SHIFT_SF']/smass('wqq') * smass(sName),
                            smear=SF[args.year]['SMEAR_SF']
                        )
                    else:
                        return templ
                templ = smorph(templ, sName)
                ##https://github.com/nsmith-/rhalphalib/blob/master/rhalphalib/template_morph.py#L45-L58 how do i do this on ROOT templates?
                if args.do_systematics:
                    if args.ftest and sName == siggy:
                        sample.setParamEffect(sys_lumi, 1.1)
                        ch.addSample(sample)

                        continue

                    sample.setParamEffect(sys_lumi, lumi_dict_unc[args.year])
                    sample.setParamEffect(
                        sys_lumi_correlated, lumi_correlated_dict_unc[args.year]
                    )
                    if "2016" not in args.year:
                        sample.setParamEffect(
                            sys_lumi_1718, lumi_1718_dict_unc[args.year]
                        )
                    sample.setParamEffect(sys_eleveto, 1.005)
                    sample.setParamEffect(sys_muveto, 1.005)
                    sample.setParamEffect(sys_tauveto, 1.05)
                    lnN = True # False if sName in [siggy, bsiggy] else True #Use shape unc for signal MC stats
                    sample.autoMCStats(lnN=lnN,sample_name=f"{sName}_{args.year}")

                    sys_names = [
                        "JES",
                        "JER",
                        "jet_trigger",
                        "pileup_weight",
                        'Z_d2kappa_EW', 'Z_d3kappa_EW', 
                        'W_d2kappa_EW', 'W_d3kappa_EW', 
                        'd1kappa_EW', 'd1K_NLO', 'd2K_NLO', 'd3K_NLO',
                        'btagSF_heavy_correlated','btagSF_light_correlated',
                        f'btagSF_heavy_{era_dict[args.year]}', f'btagSF_light_{era_dict[args.year]}', 
                        #'scalevar_7pt', 'scalevar_3pt',
                        'UES',#'btagEffStat', 'btagWeight',
                    ]
                    if "2016" in args.year or "2017" in args.year: 
                        sys_names.append("L1Prefiring")
                    if "2018" in args.year:
                        sys_names.append("HEMissue") 
                    if stype == rl.Sample.SIGNAL : #and not args.ftest:
                        sName = short_to_long[sName]
                    for sys_name in sys_names:
                        if (
                            ("NLO" in sys_name) or ("EW" in sys_name)
                        ) and not sName in ["zqq", "wqq", "zbb"]:
                            continue
                        if ("Z_d" in sys_name) and sName not in ["zqq","dy", "zbb"]:
                            continue
                        if ("W_d" in sys_name) and sName not in ["wqq","wlnu"]:
                            continue
                        logging.debug(f"  Adding systematic: '{sys_name}'")
                        name_up = sys_name_updown[sys_name.replace(era_dict[args.year],"year")][1]
                        name_down = sys_name_updown[sys_name.replace(era_dict[args.year],"year")][0]
                        if "year" in name_up:
                            name_up = name_up.replace("year",args.year)
                            name_down = name_down.replace("year",args.year)
                        if sys_shape_dict[sys_name].combinePrior == "lnN":
                            _sys_ef = shape_to_num(
                                region,
                                sName,
                                ptbin,
                                [name_down,name_up],#sys_name_updown[sys_name],
                                mask,
                                bound=None if "scalevar" not in sys_name else 0.25,
                                inflate=False,
                                scaledown=True if args.ftest and siggy == sName else False,
                                #scaleup=False #### CHECK THIS!!! True if (sName in "zpqq" or sName in "zpbb") else False,
                            )
                            
                            if _sys_ef is None:
                                log.warning(f"Effect of {sys_name} is None for {sName} in ptbin {ptbin}")
                                continue
                            
                            sample.setParamEffect(sys_shape_dict[sys_name], _sys_ef)
                    mtempl = AffineMorphTemplate(templ)
                    _extra_scaling = 4.
                    if sName not in ['qcd', 'dy', 'wlnu',"tt","st",] : #and "pass" in region:
                        log.debug(f"Adding SF shift/smear nuisance for sample {sName} with extra scaling {_extra_scaling}.")
                        realshift = SF[args.year]['SHIFT_SF_ERR']/smass('wqq') * smass(sName) * _extra_scaling
                        _up = mtempl.get(shift=realshift)
                        _down = mtempl.get(shift=-realshift)
                        if badtemp_ma(templ[0]) or badtemp_ma(_up[0]) or badtemp_ma(_down[0]):
                            log.info("Filling sample {} in ptbin {} in channel {} with zero, scale systematic would be empty. nominal template={}, up={}, down={}".format(sName,ptbin,ch.name,np.round(templ[0],5),np.round(_up[0],5), np.round(_down[0],5)))
                            sample.mask = np.ones_like(_down[0])                
                        #    continue
                        if args.decorr_scale_cat:
                            if "pass_T_bvl_pass_L" in region:
                                log.debug(f"Setting nuisance parameter CMS_scale_{args.year}_highbvl on sample {sName}.") 
                                sample.setParamEffect(sys_shape_dict[f"CMS_scale_{args.year}_highbvl"], deepcopy(_up), deepcopy(_down), scale=1/_extra_scaling)
                            elif "pass_T_bvl_fail_L" in region:
                                log.debug(f"Setting nuisance parameter CMS_scale_{args.year}_lowbvl on sample {sName}.") 
                                sample.setParamEffect(sys_shape_dict[f"CMS_scale_{args.year}_lowbvl"], deepcopy(_up), deepcopy(_down), scale=1/_extra_scaling)
                            elif "fail_T" in region:
                                log.debug(f"Setting nuisance parameter CMS_scale_{args.year}_fail on sample {sName}.") 
                                sample.setParamEffect(sys_shape_dict[f"CMS_scale_{args.year}_fail"], deepcopy(_up), deepcopy(_down), scale=1/_extra_scaling)
                        elif args.decorr_scale_cat_pt:
                            if "pass_T_bvl_pass_L" in region:
                                log.debug(f"Setting nuisance parameter CMS_scale_{args.year}_highbvl on sample {sName} in ptbin {ptbin}.") 
                                sample.setParamEffect(sys_shape_dict[f"CMS_scale_{args.year}_highbvl_ptbin{ptbin}"], deepcopy(_up), deepcopy(_down), scale=1/_extra_scaling)
                            elif "pass_T_bvl_fail_L" in region:
                                log.debug(f"Setting nuisance parameter CMS_scale_{args.year}_lowbvl on sample {sName} in ptbin {ptbin}.") 
                                sample.setParamEffect(sys_shape_dict[f"CMS_scale_{args.year}_lowbvl_ptbin{ptbin}"], deepcopy(_up), deepcopy(_down), scale=1/_extra_scaling)
                        elif args.decorr_scale_pt: 
                            log.debug(f"Setting nuisance parameter CMS_scale_{args.year}_ptbin{ptbin} on sample {sName}.") 
                            sample.setParamEffect(sys_shape_dict[f"CMS_scale_{args.year}_ptbin{ptbin}"], deepcopy(_up), deepcopy(_down), scale=1/_extra_scaling)
                        else:
                            sample.setParamEffect(sys_scale, deepcopy(_up), deepcopy(_down), scale=1/_extra_scaling)
                        _extra_scaling = 0.4/SF[args.year]['SMEAR_SF_ERR'] ## Because the smear uncertainties vary so much by era, instead of a fixed extra_scaling, use an extra_scaling that pushes things to 40% uncertainty . This should keep the interpolation from touching zero.
                        _up = mtempl.get(smear=1 + SF[args.year]['SMEAR_SF_ERR'] * _extra_scaling) ###When the smear unc is large,  extra scaling that is too large can cause negative values
                        _down = mtempl.get(smear=1 - SF[args.year]['SMEAR_SF_ERR'] * _extra_scaling) ###When the smear unc is large, extra scaling that is too large can cause negative values
                        if badtemp_ma(_up[0]) or badtemp_ma(_down[0]):
                            log.info("Filling sample {} in ptbin {} in channel {} with zero, smear systematic would be empty. nominal template={}, up={}, down={}".format(sName,ptbin,ch.name,np.round(templ[0],5),np.round(_up[0],5), np.round(_down[0],5)))
                            sample.mask = np.ones_like(_down[0])                
                            log.info("After masking: nominal template={}".format(sample.show(),5))
                            #continue
                        sample.setParamEffect(sys_smear, _up, _down, scale=1 /_extra_scaling) 

                else:
                    sample.setParamEffect(sys_lumi, 1.1)

                if "zpqq" in sName or "zpbb" in sName:
                    log.warning(f"Scaling sample {sName} by 2.066 for xsec")
                    sample.scale(2.066)
                ch.addSample(sample)

            if not args.pseudo:
                data_obs = get_templ(
                    region, f"JetHT_{era_dict[args.year]}", ptbin, tagger, fourptbins=args.four_pt_bins
                )
                if throwPoisson:
                    yields = np.random.poisson(yields)
            else:
                yields = []
                if args.qcd_ftest:
                    include_samples = ["QCD"]
                #elif args.h_sensitivity:
                else:
                    include_samples = ["wqq", "zqq", "zbb", "tt", "wlnu", "dy", "st", "hbb","QCD", "vv"]
                for sName in include_samples:
                    _sample = get_templ(
                        region, sName, ptbin, tagger, fourptbins=args.four_pt_bins
                    )
                    _sample_yield = _sample[0]
                    if args.scale_qcd:
                        if "pass" in region:
                            dummyqcd = rl.TemplateSample(
                                "dummyqcd", rl.Sample.BACKGROUND, _sample, force_positive=True
                            )
                            nomrate = dummyqcd._nominal
                            downrate = np.sum(
                                np.nan_to_num(
                                    dummyqcd._nominal - np.sqrt(dummyqcd._sumw2), 0.0
                                )
                            )
                            uprate = np.sum(
                                np.nan_to_num(
                                    dummyqcd._nominal + np.sqrt(dummyqcd._sumw2), 0.0
                                )
                            )
                            diff = np.sum(
                                np.abs(uprate - nomrate) + np.abs(downrate - nomrate)
                            )
                            mean = diff / (2.0 * np.sum(nomrate))
                            # sqrt(nom*N) = mean -> N = mean**2/nom
                            scale = mean**2 / np.sum(nomrate)
                            if args.lowbvl:
                                scale = 6.5  # 1./np.sqrt(scale)
                            elif args.highbvl:
                                scale = 5.5  # 1./np.sqrt(scale)
                            scale_pass.append(scale)
                            #pprint(
                            #    "qcdscale needed to match mcstat uncs: using poisson:",scale,
                            #)
                            # _sample_yield = _sample_yield.copy()*1./scale
                        else:
                            scale = scale_pass[ptbin]
                        yields.append(_sample_yield * scale)
                    else:
                        yields.append(_sample_yield)
                yields = np.sum(np.array(yields), axis=0)
                #pprint("ptbin/region", ptbin, region, yields)
                if throwPoisson:
                    yields = np.random.poisson(yields)

                data_obs = (yields, msd.binning, msd.name)

            ch.setObservation(data_obs[0:3])

            # drop bins outside rho validity
            # validbins[ptbin][0:2] = False
            mask = validbins[ptbin]
            if args.year=="2016APV" and ptbin==3 and "bvl_pass_L" in region and args.mask_outlier:
                mask[9] = False
            ch.mask = mask
        
    # Systematics    
    # ["wqq", "zqq", "zbb", "tt", "wlnu", "dy", "st", "hbb", siggy, bsiggy]
    if args.do_systematics and not args.ftest:
        # Do 2-region SFs
        indep = rl.IndependentParameter("failscale_{year}".format(year=args.year), 1., 0., 100.1)
        indepW = rl.IndependentParameter("failscaleW_{year}".format(year=args.year), 1.,0., 100.1)

        for ptbin in range(npt):
            log.debug(f"Making BB and QQ SFs for {ptbin}. Channels are {model.channels}.")
            ch_pass = model[f"ptbin{ptbin}passTbvlpassL"]
            ch_fail = model[f"ptbin{ptbin}passTbvlfailL"]

            ### W MISTAGGING ###  
            qq_samples = ["zqq","wqq","vv",siggy]
            for sName in qq_samples:  
                sample_pass = ch_pass[sName]
                sample_fail = ch_fail[sName]
                ### PHIL WAY, I DONT KNOW IF THIS WORKS
                '''
                for sf, unc in zip([SF[args.year]['W_SF']], [SF[args.year]['W_SF_ERR']]):                      
                    template_pass_pass = get_templ(
                            "pass_T_bvl_pass_L", short_to_long[sName], ptbin, tagger, fourptbins=args.four_pt_bins
                    )
                    template_pass_fail = get_templ(
                            "pass_T_bvl_fail_L", short_to_long[sName], ptbin, tagger, fourptbins=args.four_pt_bins
                    )
                    yield_pass = template_pass_pass[0].sum()
                    yield_fail = template_pass_fail[0].sum()
                    sfunc = 1. + unc / sf
                    if yield_pass > 0.:
                        sample_pass.scale(sf)                
                        sample_pass.setParamEffect(sys_Weff, sfunc, 1/sfunc)
                        logging.debug(f"  Nuisance: '{sys_Weff.name}', sample: '{sName}', region: 'passhighbvl', ptbin: {ptbin}, sf: {sf:.3f}, sfunc_nominal: {unc:.3f}, card_unc: {sfunc:.3f}/{sfunc:.3f}")
                    if yield_fail > 0.:
                        # Scale fail
                        sff = 1 + (1 - sf) * yield_pass / yield_fail
                        sample_fail.scale(sff)   
                        ratio =  yield_pass / yield_fail
                        rmod = (yield_pass * sf) / (yield_fail * sff)
                        sample_fail.setParamEffect(indepW, (1 - sys_Weff * unc * rmod))
                        logging.debug(f"  Nuisance: '{sys_Weff.name}', sample: '{sName}', region: 'passlowbvl', ptbin: {ptbin}, sf: {sf:.3f}, sfunc_nominal: {unc:.3f}, card_unc: {sfunc:.3f}/{sfunc:.3f}")
                '''
                ### ANDRZEJ WEY, I THINK THIS WORKS
                for sf, unc in zip([SF[args.year]['W_SF']], [SF[args.year]['W_SF_ERR']]):                      
                    template_pass_pass = get_templ(
                            "pass_T_bvl_pass_L", short_to_long[sName], ptbin, tagger, fourptbins=args.four_pt_bins
                    )
                    template_pass_fail = get_templ(
                            "pass_T_bvl_fail_L", short_to_long[sName], ptbin, tagger, fourptbins=args.four_pt_bins
                    )
                    yield_pass = template_pass_pass[0].sum()
                    yield_fail = template_pass_fail[0].sum()
                    sfunc = 1. + unc / sf
                    if yield_fail > 0.:
                        sample_fail.scale(sf)                
                        sample_fail.setParamEffect(sys_Weff, sfunc, 1/sfunc)
                        logging.debug(f"  Nuisance: '{sys_Weff.name}', sample: '{sName}', region: 'passlowbvl', ptbin: {ptbin}, sf: {sf:.3f}, sfunc_nominal: {unc:.3f}, card_unc: {sfunc:.3f}/{sfunc:.3f}")
                    if yield_pass > 0.:
                        # Scale fail
                        sff = 1 + (1 - sf) * yield_fail / yield_pass
                        sample_pass.scale(sff)   
                        ratio =  yield_fail / yield_pass
                        rmod = (yield_fail * sf) / (yield_pass * sff)
                        sample_pass.setParamEffect(indepW, (1 - sys_Weff * unc * rmod))
                        logging.debug(f"  Nuisance: '{sys_Weff.name}', sample: '{sName}', region: 'passhighbvl', ptbin: {ptbin}, sf: {sf:.3f}, sfunc_nominal: {unc:.3f}, card_unc: {sfunc:.3f}/{sfunc:.3f}")

            ### BB MISTAGGING ###
            bb_samples = ["zbb","hbb"]
            if not args.ftest: bb_samples += [bsiggy,psiggy]
            for sName in bb_samples:
                for sf, unc in zip([SF[args.year]['BB_SF']], [SF[args.year]['BB_SF_ERR']]):                      
                    sample_pass = ch_pass[f"{sName}"]
                    sample_fail = ch_fail[f"{sName}"]
   
                    template_pass_pass = get_templ(
                            "pass_T_bvl_pass_L", short_to_long[sName], ptbin, tagger, fourptbins=args.four_pt_bins
                    )
                    template_pass_fail = get_templ(
                            "pass_T_bvl_fail_L", short_to_long[sName], ptbin, tagger, fourptbins=args.four_pt_bins
                    )
                    yield_pass = template_pass_pass[0].sum()
                    yield_fail = template_pass_fail[0].sum()
            
                    sfunc = 1. + unc / sf
                    if yield_pass > 0.:
                        sample_pass.scale(sf)                
                        sample_pass.setParamEffect(sys_bbeff, sfunc, 1/sfunc)
                        logging.debug(f"  Nuisance: '{sys_bbeff.name}', sample: '{sName}', region: 'passhighbvl', ptbin: {ptbin}, sf: {sf:.3f}, sfunc_nominal: {unc:.3f}, card_unc: {sfunc:.3f}/{sfunc:.3f}")
                    if yield_fail > 0.:
                        # Scale fail
                        sff = 1 + (1 - sf) * yield_pass / yield_fail
                        sample_fail.scale(sff)   
                        ratio =  yield_pass / yield_fail
                        rmod = (yield_pass * sf) / (yield_fail * sff)
                        sample_fail.setParamEffect(indep, (1 - sys_bbeff * unc * rmod))
                        logging.debug(f"  Nuisance: '{sys_veff.name}', sample: '{sName}', region: 'passlowbvl', ptbin: {ptbin}, sf: {sf:.3f}, sfunc_nominal: {unc:.3f}, card_unc: {sfunc:.3f}/{sfunc:.3f}")
                
    if args.do_systematics and not args.ftest:        
        # Do 3-region SFs
        for ptbin in range(npt):
            log.debug(f"Making two-pronged SFs for ptbin{ptbin}. Channels are {model.channels}.")
            ch_fail = model[f"ptbin{ptbin}failT"]
            ch_pass_pass = model[f"ptbin{ptbin}passTbvlpassL"]
            ch_pass_fail = model[f"ptbin{ptbin}passTbvlfailL"]
            qq_samples = ["wqq", "zqq", "zbb", "hbb", "vv", siggy, bsiggy, psiggy]
            #if not args.ftest: qq_samples += [bsiggy] 
            for sName in qq_samples:  # consider tt/st
                log.debug(f"Working on {sName}")
                log.debug(f"ch_pass_pass samples are {ch_pass_pass.samples}")
                log.debug(f"ch_pass_fail samples are {ch_pass_fail.samples}")
                sample_fail = ch_fail[f"{sName}"]
                sample_pass_pass = ch_pass_pass[f"{sName}"]
                sample_pass_fail = ch_pass_fail[f"{sName}"]
                template_pass_pass = get_templ(
                        "pass_T_bvl_pass_L", short_to_long[sName], ptbin, tagger, fourptbins=args.four_pt_bins
                )
                template_pass_fail = get_templ(
                        "pass_T_bvl_fail_L", short_to_long[sName], ptbin, tagger, fourptbins=args.four_pt_bins
                )
                template_fail = get_templ(
                        "fail_T", short_to_long[sName], ptbin, tagger, fourptbins=args.four_pt_bins
                )
                yield_pass_pass = template_pass_pass[0].sum()
                yield_pass_fail = template_pass_fail[0].sum()
                yield_fail = template_fail[0].sum()
                for sf, unc in zip([SF[args.year]['V_SF']], [SF[args.year]['V_SF_ERR']]):                      
                    sfunc = 1. + unc / sf
                    # Scale both pass regions      
                    log.debug(f"Scaling sample {sName} by {sf} +- {unc} for 2prong.")
                    if yield_pass_pass > 0.:      
                        sample_pass_pass.scale(sf)                
                        sample_pass_pass.setParamEffect(sys_veff, sfunc)
                        logging.debug(f"  Nuisance: '{sys_veff.name}', sample: '{sName}', region: 'passhighbvl', ptbin: {ptbin}, sf: {sf:.3f}, sfunc_nominal: {unc:.3f}, card_unc: {sfunc:.3f}/{1/sfunc:.3f}")
                    if yield_pass_fail > 0.:
                        sample_pass_fail.scale(sf)                
                        sample_pass_fail.setParamEffect(sys_veff, sfunc)
                        logging.debug(f"  Nuisance: '{sys_veff.name}', sample: '{sName}', region: 'passlowbvl', ptbin: {ptbin}, sf: {sf:.3f}, sfunc_nominal: {unc:.3f}, card_unc: {sfunc:.3f}/{sfunc:.3f}")
                    # Scale fail
                    if yield_fail > 0.:
                        sf_flipped, sfup, sfdn = flipSF(sf, unc, yield_pass_pass+yield_pass_fail, yield_fail)
                        sample_fail.scale(sf_flipped)                
                        sample_fail.setParamEffect(sys_veff, sfup, sfdn)
                        logging.debug(f"  Nuisance: '{sys_veff.name}', sample: '{sName}', region: 'fail', ptbin: {ptbin}, sf: {sf_flipped:.3f}, sfunc_nominal: {unc:.3f}, card_unc: {sfup:.3f}/{sfdn:.3f}")
    elif args.do_systematics and args.ftest:
        for ptbin in range(npt):
            passkey = f"ptbin{ptbin}pass"
            if args.highbvl:
                passkey = f"ptbin{ptbin}passTbvlpassL"
                histkey = "pass_T_bvl_pass_L"
            elif args.lowbvl:
                passkey = f"ptbin{ptbin}passTbvlfailL"
                histkey = "pass_T_bvl_fail_L"
            failkey = f"ptbin{ptbin}failT"
            ch_fail = model[failkey]
            ch_pass = model[passkey]
            qq_samples = ["wqq", "zqq", "zbb", "hbb","vv"] #added siggy here? 
            for sName in qq_samples:  
                sample_fail = ch_fail[f"{sName}"]
                sample_pass = ch_pass[f"{sName}"]
                template_pass = get_templ(
                        histkey, short_to_long[sName], ptbin, tagger, fourptbins=args.four_pt_bins
                )
                template_fail = get_templ(
                        "fail_T", short_to_long[sName], ptbin, tagger, fourptbins=args.four_pt_bins
                )
                yield_pass = template_pass[0].sum()
                yield_fail = template_fail[0].sum()
                for sf, unc in zip([SF[args.year]['V_SF']], [SF[args.year]['V_SF_ERR']]):                      
                    sfunc = 1. + unc / sf
                    # Scale both pass regions           
                    if yield_pass > 0.: 
                        sample_pass.scale(sf)                
                        sample_pass.setParamEffect(sys_veff, sfunc)
                        logging.debug(f"  Nuisance: '{sys_veff.name}', sample: '{sName}', region: 'pass', ptbin: {ptbin}, sf: {sf:.3f}, sfunc_nominal: {unc:.3f}, card_unc: {sfunc:.3f}/{1/sfunc:.3f}")
                    # Scale fail
                    if yield_fail > 0.:

                        sf_flipped, sfup, sfdn = flipSF(sf, unc, yield_pass, yield_fail)
                        sample_fail.scale(sf_flipped)                
                        sample_fail.setParamEffect(sys_veff, sfup, sfdn)
                        logging.debug(f"  Nuisance: '{sys_veff.name}', sample: '{sName}', region: 'fail', ptbin: {ptbin}, sf: {sf_flipped:.3f}, sfunc_nominal: {unc:.3f}, card_unc: {sfup:.3f}/{1/sfdn:.3f}")

    # Build QCD TF templates
    log.info("Building QCD TF templates")    
    if args.tworeg and not (args.qcd_ftest or args.ftest):
        for ptbin in range(npt):
            failCh = model[f"ptbin{ptbin}failT"]
            passChpass = model[f"ptbin{ptbin}passTbvlpassL"]
            passChfail = model[f"ptbin{ptbin}passTbvlfailL"]

            # Fail region
            qcdparams = np.array(
                [
                    rl.IndependentParameter("qcdparam_{year}_ptbin{ptbin}_msdbin{i}".format(year=args.year,ptbin=ptbin, i=i,), 0)
                    for i in range(msd.nbins)
                ]
            )
            initial_qcd = failCh.getObservation().astype(
                float
            )  # was integer, and numpy complained about subtracting float from it
            log.debug(f"Initial_qcd: {initial_qcd}")
            for sample in failCh:
                initial_qcd -= sample.getExpectation(nominal=True)
            if args.pseudo:
                initial_qcd[initial_qcd < 0] = 0.0
            if np.any(initial_qcd < 0.0):
                raise ValueError("initial_qcd negative for some bins..", initial_qcd)
            sigmascale = 10  # to scale the deviation from initial
            scaledparams = (
                initial_qcd
                * (1 + sigmascale / np.maximum(1.0, np.sqrt(initial_qcd))) ** qcdparams
            )
            fail_qcd = rl.ParametericSample(
                f"ptbin{ptbin}failT_{args.year}_qcd", rl.Sample.BACKGROUND, msd, scaledparams
            )
            failCh.addSample(fail_qcd)

            # Transfer factor templates
            passpass_qcd = rl.TransferFactorSample(
                f"ptbin{ptbin}passTbvlpassL_{args.year}_qcd",
                rl.Sample.BACKGROUND,
                all_tf_params[0][ptbin, :],
                fail_qcd,
            )
            passChpass.addSample(passpass_qcd)

            passfail_qcd = rl.TransferFactorSample(
                f"ptbin{ptbin}passTbvlfailL_{args.year}_qcd",
                rl.Sample.BACKGROUND,
                all_tf_params[1][ptbin, :],
                fail_qcd,
            )
            passChfail.addSample(passfail_qcd)
    else:
        for ptbin in range(npt):
            passkey = f"ptbin{ptbin}pass"
            if args.highbvl:
                passkey = f"ptbin{ptbin}passTbvlpassL"
            elif args.lowbvl:
                passkey = f"ptbin{ptbin}passTbvlfailL"
            failkey = f"ptbin{ptbin}failT"
            failCh = model[failkey]
            passCh = model[passkey]

            qcdparams = np.array(
                [
                    rl.IndependentParameter("qcdparam_ptbin{ptbin}_msdbin{i}_{year}".format(ptbin=ptbin, i=i, year=args.year), 0)
                    for i in range(msd.nbins)
                ]
            )
            initial_qcd = failCh.getObservation().astype(
                float
            )  # was integer, and numpy complained about subtracting float from it
            #pprint(initial_qcd)
            #log.debug("Initial_qcd", initial_qcd)
            for sample in failCh:
                initial_qcd -= sample.getExpectation(nominal=True)
            if args.pseudo:
                initial_qcd[initial_qcd < 0] = 0.0
            if np.any(initial_qcd < 0.0):
                raise ValueError("initial_qcd negative for some bins..", initial_qcd)
            sigmascale = 10  # to scale the deviation from initial
            scaledparams = (
                initial_qcd
                * (1 + sigmascale / np.maximum(1.0, np.sqrt(initial_qcd))) ** qcdparams
            )
            fail_qcd = rl.ParametericSample(
                f"ptbin{ptbin}failT_{args.year}_qcd", rl.Sample.BACKGROUND, msd, scaledparams
            )
            failCh.addSample(fail_qcd)
            pass_qcd = rl.TransferFactorSample(
                #f"ptbin{ptbin}pass_{args.year}_qcd",
                f"{passkey}_{args.year}_qcd",
                rl.Sample.BACKGROUND,
                tf_params[ptbin, :],
                fail_qcd,
            )
            passCh.addSample(pass_qcd)

    if args.constrainedttbar: 
        log.debug("Assuming muonCR has been fit independently and we can run ftests by applying constrained nuisances.")
        for ptbin in range(npt):
            passkey = f"ptbin{ptbin}pass"
            if args.highbvl:
                passkey = f"ptbin{ptbin}passTbvlpassL"
                sfkey="highbvl"
            elif args.lowbvl:
                passkey = f"ptbin{ptbin}passTbvlfailL"
                sfkey="lowbvl"
            failCh = model[f'ptbin{ptbin}failT']
            passCh = model[passkey]
            tqqpass = passCh['tt']
            tqqfail = failCh['tt']
            stqqpass = passCh['st']
            stqqfail = failCh['st']
            sumPass = tqqpass.getExpectation(nominal=True).sum()
            sumFail = tqqfail.getExpectation(nominal=True).sum()
            sumPass += stqqpass.getExpectation(nominal=True).sum()
            sumFail += stqqfail.getExpectation(nominal=True).sum()
            tqqPF =  sumPass / sumFail
            normSF, normunc = topSF[args.year][sfkey]["tqqnormSF_val"], topSF[args.year][sfkey]["tqqnormSF_unc"]
            effSF, effunc = topSF[args.year][sfkey]["tqqeffSF_val"], topSF[args.year][sfkey]["tqqeffSF_unc"]

            normunc = 1. + normunc / normSF
            effunc = 1. + effunc / effSF
 
            effsf, effup, effdown = flipSF(effSF, effunc, sumPass, sumFail)

            tqqpass.scale(normSF)
            tqqfail.scale(normSF)
            tqqpass.scale(effSF)
            tqqfail.scale(effsf)
            tqqpass.setParamEffect(tqqeffSF, effunc)
            tqqfail.setParamEffect(tqqeffSF, effup, effdown)
            tqqpass.setParamEffect(tqqnormSF, normunc)
            tqqfail.setParamEffect(tqqnormSF, normunc)

            stqqpass.scale(normSF)
            stqqfail.scale(normSF)
            stqqpass.scale(effSF)
            stqqfail.scale(effsf)
            stqqpass.setParamEffect(tqqeffSF, effunc)
            stqqfail.setParamEffect(tqqeffSF, effup, effdown)
            stqqpass.setParamEffect(tqqnormSF, normunc)
            stqqfail.setParamEffect(tqqnormSF, normunc)

    if args.muonCR:
        log.debug("Adding constraints to tt and st in SR") 
        ##Two regions AND not running ftest (for ftest we assume only 1 pass either highbvl or lowbvl and 1 fail)
        if args.tworeg and not args.ftest:
            for ptbin in range(npt):

                failCh = model[f"ptbin{ptbin}{fail_regs[0].replace('_','')}"]
                passCh_highbvl = model[f"ptbin{ptbin}{pass_regs[0].replace('_','')}"]
                passCh_lowbvl  = model[f"ptbin{ptbin}{pass_regs[1].replace('_','')}"]
                tqqpass_highbvl = passCh_highbvl["tt"]
                tqqpass_lowbvl  = passCh_lowbvl["tt"]
                tqqfail         = failCh["tt"]
                stqqpass_highbvl= passCh_highbvl["st"]
                stqqpass_lowbvl = passCh_lowbvl["st"]
                stqqfail        = failCh["st"]

                sumPass = tqqpass_highbvl.getExpectation(nominal=True).sum() +\
                          tqqpass_lowbvl.getExpectation(nominal=True).sum() +\
                          stqqpass_highbvl.getExpectation(nominal=True).sum() +\
                          stqqpass_lowbvl.getExpectation(nominal=True).sum()

                sumFail = tqqfail.getExpectation(nominal=True).sum() +\
                          stqqfail.getExpectation(nominal=True).sum()

                tqqPF = sumPass / sumFail 

                sumHighbvl = tqqpass_highbvl.getExpectation(nominal=True).sum() +\
                             stqqpass_highbvl.getExpectation(nominal=True).sum()
                sumLowbvl =  tqqpass_lowbvl.getExpectation(nominal=True).sum() +\
                             stqqpass_lowbvl.getExpectation(nominal=True).sum()

                tqqHL = sumHighbvl / sumLowbvl

                # common eff SFs for both high and low bvl regions (tt and st)
                tqqpass_highbvl .setParamEffect(tqqeffSF, 1 * tqqeffSF)
                tqqpass_lowbvl  .setParamEffect(tqqeffSF, 1 * tqqeffSF)
                tqqfail         .setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
                stqqpass_highbvl.setParamEffect(tqqeffSF, 1 * tqqeffSF)
                stqqpass_lowbvl .setParamEffect(tqqeffSF, 1 * tqqeffSF)
                stqqfail        .setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)

                # SF between high and low bvl regions (tt and st)
                tqqpass_highbvl .setParamEffect(tqqeffHLSF, 1 * tqqeffHLSF)
                tqqpass_lowbvl  .setParamEffect(tqqeffHLSF, 1 * (1 - tqqeffHLSF) * tqqHL + 1)
                stqqpass_highbvl.setParamEffect(tqqeffHLSF, 1 * tqqeffHLSF)
                stqqpass_lowbvl .setParamEffect(tqqeffHLSF, 1 * (1 - tqqeffHLSF) * tqqHL + 1)

                # common norm SFs for both pass and fail 
                tqqpass_highbvl .setParamEffect(tqqnormSF, 1 * tqqnormSF)
                tqqpass_lowbvl  .setParamEffect(tqqnormSF, 1 * tqqnormSF)
                tqqfail         .setParamEffect(tqqnormSF, 1 * tqqnormSF)
                stqqpass_highbvl.setParamEffect(tqqnormSF, 1 * tqqnormSF)
                stqqpass_lowbvl .setParamEffect(tqqnormSF, 1 * tqqnormSF)
                stqqfail        .setParamEffect(tqqnormSF, 1 * tqqnormSF)

        else:
            passkey = f"ptbin{ptbin}pass"
            if args.highbvl:
                passkey = f"ptbin{ptbin}passTbvlpassL"
            elif args.lowbvl:
                passkey = f"ptbin{ptbin}passTbvlfailL"
            for ptbin in range(npt):
                failCh = model[f'ptbin{ptbin}failT']
                passCh = model[passkey]
                tqqpass = passCh['tt']
                tqqfail = failCh['tt']
                stqqpass = passCh['st']
                stqqfail = failCh['st']
                sumPass = tqqpass.getExpectation(nominal=True).sum()
                sumFail = tqqfail.getExpectation(nominal=True).sum()
                sumPass += stqqpass.getExpectation(nominal=True).sum()
                sumFail += stqqfail.getExpectation(nominal=True).sum()
                tqqPF =  sumPass / sumFail

                tqqpass.setParamEffect(tqqeffSF, 1 * tqqeffSF)
                tqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
                tqqpass.setParamEffect(tqqnormSF, 1 * tqqnormSF)
                tqqfail.setParamEffect(tqqnormSF, 1 * tqqnormSF)
                stqqpass.setParamEffect(tqqeffSF, 1 * tqqeffSF)
                stqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
                stqqpass.setParamEffect(tqqnormSF, 1 * tqqnormSF)
                stqqfail.setParamEffect(tqqnormSF, 1 * tqqnormSF)
 
        # Top CR
        log.debug("Adding Top CR") 
        for region in list(set(pass_regs+fail_regs)):
            model_reg_name = region.replace("_", "")
            ch = rl.Channel(f"muonCR{model_reg_name}")
            model.addChannel(ch)
    
            templates = {
                    "tt": get_templ(
                        region, "tt", 0, tagger, fourptbins=args.four_pt_bins,muon=True,observable=msd_muon
                    ),
                    "st": get_templ(
                        region, "st", 0, tagger, fourptbins=args.four_pt_bins,muon=True,observable=msd_muon
                    ),
                    "qcd": get_templ(
                        region, "QCD", 0, tagger, fourptbins=args.four_pt_bins,muon=True,observable=msd_muon
                    ),
                    "wlnu": get_templ(
                        region, "wlnu", 0, tagger, fourptbins=args.four_pt_bins,muon=True,observable=msd_muon
                    ),
                    "dy": get_templ(
                        region, "dy", 0, tagger, fourptbins=args.four_pt_bins,muon=True,observable=msd_muon
                    ),
            }
    
            include_samples = ["tt", "wlnu", "dy", "st","qcd",] 
            for sName in include_samples:
                templ = templates[sName] 
                if args.collapse:
                    templ = one_bin(templ)
                stype = rl.Sample.BACKGROUND

                sample = rl.TemplateSample(ch.name + "_" + sName, stype, templ, force_positive=True)
                if args.do_systematics:
                    sample.setParamEffect(sys_lumi, lumi_dict_unc[args.year])
                    sample.setParamEffect(sys_lumi_correlated, lumi_correlated_dict_unc[args.year])
                    if '2016' not in args.year:
                        sample.setParamEffect(sys_lumi_1718, lumi_1718_dict_unc[args.year])
                    sample.setParamEffect(sys_eleveto, 1.005)
                    sample.setParamEffect(sys_tauveto, 1.05)

                    sys_names = [
                        'JES', 'JER', 'muotrig', 'muoid','muoiso', 'pileup_weight',
                        #'UES',
                        #'L1Prefiring',
                        'btagSF_heavy_correlated','btagSF_light_correlated',
                        f'btagSF_heavy_{era_dict[args.year]}', f'btagSF_light_{era_dict[args.year]}', 
                    ]
                    if "2016" in args.year or "2017" in args.year: 
                        sys_names.append("L1Prefiring")

                    for sys_name in sys_names:
                        if (
                            ("NLO" in sys_name) or ("EW" in sys_name)
                        ) and not sName in ["zqq", "wqq"]:
                            continue
                        if ("Z_d" in sys_name) and sName not in ["zqq","dy"]:
                            continue
                        if ("W_d" in sys_name) and sName not in ["wqq","wlnu"]:
                            continue
                        name_up = sys_name_updown[sys_name.replace(era_dict[args.year],"year")][1]
                        name_down = sys_name_updown[sys_name.replace(era_dict[args.year],"year")][0]
                        if "year" in name_up:
                            name_up = name_up.replace("year",args.year)
                            name_down = name_down.replace("year",args.year)
                        if "qcd" in sName: 
                            continue #sName = "QCD"
                        if sys_shape_dict[sys_name].combinePrior == "lnN":
                            _sys_ef = shape_to_num(
                                region,
                                sName,
                                0,
                                [name_down,name_up],#sys_name_updown[sys_name],
                                None,
                                bound=None if "scalevar" not in sys_name else 0.25,
                                inflate=False,
                                muon=True, 
                            )
                            if _sys_ef is None:
                                continue
                            sample.setParamEffect(sys_shape_dict[sys_name], _sys_ef)
                else:
                    sample.setParamEffect(sys_lumi, lumi_dict_unc[args.year])
    
                ch.addSample(sample)
    
            
            data_obs = get_templ(
                    region, f"SingleMuon_{era_dict[args.year]}", 0, tagger, fourptbins=args.four_pt_bins,muon=True,observable=msd_muon,
            )
            if args.collapse:
                data_obs = one_bin(data_obs)
            ch.setObservation(data_obs[0:3])
        if args.tworeg and not args.ftest:
            tqqpass_highbvl = model[f"muonCR{pass_regs[0].replace('_','')}"]["tt"]
            tqqpass_lowbvl  = model[f"muonCR{pass_regs[1].replace('_','')}"]["tt"]
            tqqfail         = model[f"muonCR{fail_regs[0].replace('_','')}"]["tt"]
            stqqpass_highbvl = model[f"muonCR{pass_regs[0].replace('_','')}"]["st"]
            stqqpass_lowbvl  = model[f"muonCR{pass_regs[1].replace('_','')}"]["st"]
            stqqfail         = model[f"muonCR{fail_regs[0].replace('_','')}"]["st"]


            sumPass = tqqpass_highbvl.getExpectation(nominal=True).sum() +\
                      tqqpass_lowbvl.getExpectation(nominal=True).sum() +\
                      stqqpass_highbvl.getExpectation(nominal=True).sum() +\
                      stqqpass_lowbvl.getExpectation(nominal=True).sum()

            sumFail = tqqfail.getExpectation(nominal=True).sum() +\
                      stqqfail.getExpectation(nominal=True).sum()

            tqqPF = sumPass / sumFail


            sumHighbvl = tqqpass_highbvl.getExpectation(nominal=True).sum() +\
                         stqqpass_highbvl.getExpectation(nominal=True).sum()
            sumLowbvl =  tqqpass_lowbvl.getExpectation(nominal=True).sum() +\
                         stqqpass_lowbvl.getExpectation(nominal=True).sum()

            tqqHL = sumHighbvl / sumLowbvl

            # common eff SFs for both high and low bvl regions (tt and st)
            tqqpass_highbvl .setParamEffect(tqqeffSF, 1 * tqqeffSF)
            tqqpass_lowbvl  .setParamEffect(tqqeffSF, 1 * tqqeffSF)
            tqqfail         .setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
            stqqpass_highbvl.setParamEffect(tqqeffSF, 1 * tqqeffSF)
            stqqpass_lowbvl .setParamEffect(tqqeffSF, 1 * tqqeffSF)
            stqqfail        .setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)

            # SF between high and low bvl regions (tt and st)
            tqqpass_highbvl .setParamEffect(tqqeffHLSF, 1 * tqqeffHLSF)
            tqqpass_lowbvl  .setParamEffect(tqqeffHLSF, 1 * (1 - tqqeffHLSF) * tqqHL + 1)
            stqqpass_highbvl.setParamEffect(tqqeffHLSF, 1 * tqqeffHLSF)
            stqqpass_lowbvl .setParamEffect(tqqeffHLSF, 1 * (1 - tqqeffHLSF) * tqqHL + 1)

            # common norm SFs for both pass and fail 
            tqqpass_highbvl .setParamEffect(tqqnormSF, 1 * tqqnormSF)
            tqqpass_lowbvl  .setParamEffect(tqqnormSF, 1 * tqqnormSF)
            tqqfail         .setParamEffect(tqqnormSF, 1 * tqqnormSF)
            stqqpass_highbvl.setParamEffect(tqqnormSF, 1 * tqqnormSF)
            stqqpass_lowbvl .setParamEffect(tqqnormSF, 1 * tqqnormSF)
            stqqfail        .setParamEffect(tqqnormSF, 1 * tqqnormSF)


        else: 
            passkey = f"muonCRpassT"
            if args.highbvl:
                passkey = f"muonCRpassTbvlpassL"
            elif args.lowbvl:
                passkey = f"muonCRpassTbvlfailL"
            tqqpass = model[passkey]["tt"]
            tqqfail = model["muonCRfailT"]["tt"]
            stqqpass = model[passkey]["st"]
            stqqfail = model["muonCRfailT"]["st"]
            tqqPF = (tqqpass.getExpectation(nominal=True).sum() + stqqpass.getExpectation(nominal=True).sum())/ ( tqqfail.getExpectation(nominal=True).sum() + stqqfail.getExpectation(nominal=True).sum())

            tqqpass.setParamEffect(tqqeffSF, 1 * tqqeffSF)
            tqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
            tqqpass.setParamEffect(tqqnormSF, 1 * tqqnormSF)
            tqqfail.setParamEffect(tqqnormSF, 1 * tqqnormSF)
            stqqpass.setParamEffect(tqqeffSF, 1 * tqqeffSF)
            stqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
            stqqpass.setParamEffect(tqqnormSF, 1 * tqqnormSF)
            stqqfail.setParamEffect(tqqnormSF, 1 * tqqnormSF)

    with open(os.path.join(str(tmpdir), f"{sig}_model.pkl"), "wb") as fout:
        pickle.dump(model, fout)

    model.renderCombine(os.path.join(str(tmpdir), f"{sig}_model"))

    conf_dict = vars(args)
    conf_dict["nbins"] = float(np.sum(validbins))
    pprint(conf_dict)
    import json

    # Serialize data into file:
    json.dump(
        conf_dict,
        open(
            "{}/config.json".format(
                f"{tmpdir}/{sig}_model",
            ),
            "w",
        ),
        sort_keys=True,
        indent=4,
        separators=(",", ": "),
    )


if __name__ == "__main__":
    import time
    start_time = time.time()
    global opath
    startopath = f"{args.opath}/{tagger}/ipt{args.ipt}_irho{args.irho}"
    os.system(f"cp rhalphalib_zprime.py {startopath}/rhalphalib_zprime.py")
    for sig in signals:
        opath = f"{startopath}/{sig}/"
        if os.path.exists(opath):
            q_overwrite = Confirm.ask(f"Path: '{opath}' already exists. Overwrite?")
            if not q_overwrite:
                pprint(f"Remove with \nrm -rf {opath}")
                sys.exit()
            else:
                os.system(f"rm -rf {opath}")
        os.makedirs(opath)
        test_rhalphabet(opath, sig, args.throwPoisson)

    elapsed = time.time() - start_time
    pprint(f"Walltime: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")



