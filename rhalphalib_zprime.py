from __future__ import print_function, division
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
import click
from common import * 

rl.util.install_roofit_helpers()
# rl.ParametericSample.PreferRooParametricHist = False
np.random.seed(1)
parser = argparse.ArgumentParser(description="Rhalphalib setup.")
parser.add_argument(
    "--opath", action="store", type=str, required=True, help="Path to store output."
)
parser.add_argument(
    "--ipt", action="store", type=int, required=True, help="TF pt order."
)
parser.add_argument(
    "--irho", action="store", type=int, required=True, help="TF rho order."
)
parser.add_argument(
    "--iptMC", action="store", type=int, required=False, help="MCTF pt order."
)
parser.add_argument(
    "--irhoMC", action="store", type=int, required=False, help="MCTF rho order."
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
parser.add_argument(
    "--root_file",
    action="store",
    type=str,
    required=True,
    help="Path to ROOT holding templates.",
)
parser.add_argument(
    "--root_file_mu",
    action="store",
    type=str,
    required=False,
    help="Path to ROOT holding mu templates.",
)
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
parser.add_argument("--is_blinded", action="store_true", help="Run on 10pct dataset.")
parser.add_argument("--throwPoisson", action="store_true", help="Throw poisson.")
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


SF = {
    "2017": {
        "V_SF": 0.827,
        "V_SF_ERR": 0.042,
        "SHIFT_SF": 0.417,
        "SHIFT_SF_ERR": 0.395,
        "SMEAR_SF": 1.011,
        "SMEAR_SF_ERR":0.024,
    }
}
tagger = args.tagger

def smass(sName):
    if "hbb" in sName:
        _mass = 125.0
    elif sName in [
        "wqq",
        "tt",
        "st",
    ]:
        _mass = 80.0
    elif sName in ["zqq", "zcc", "zbb"]:
        _mass = 90.0
    
    else:
        raise ValueError("DAFUQ is {}".format(sName))
    return _mass


with open("xsec.json") as f:
    xsec_dict = json.load(f)

short_to_long = {
    "wqq": "WJetsToQQ",
    "zqq": "ZJetsToQQ",
    "tt": "TTbar",
    "st": "SingleTop",
    "wlnu": "WJetsToLNu",
    # "m50": "VectorZPrimeToQQ_M50",
    # "m75": "VectorZPrimeToQQ_M75",
    # "m100": "VectorZPrimeToQQ_M100",
    # "m125": "VectorZPrimeToQQ_M125",
    # "m150": "VectorZPrimeToQQ_M150",
    # "m200": "VectorZPrimeToQQ_M200",
    # "m250": "VectorZPrimeToQQ_M250",
    "m50": "zpqq50",
    "m75": "zpqq75",
    "m100": "zpqq100",
    "m125": "zpqq125",
    "m150": "zpqq150",
    "m200": "zpqq200",
    "m250": "zpqq250",
    "b50": "zpbb50",
    "b75": "zpbb75",
    "b100": "zpbb100",
    "b125": "zpbb125",
    "b150": "zpbb150",
    "b200": "zpbb200",
    "b250": "zpbb250",
    # "m300" : "VectorZPrimeToQQ_M300",
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
    "d1kappa_EW": "lnN",
    "d1K_NLO": "lnN",
    "d2K_NLO": "lnN",
    "d3K_NLO": "lnN",
    "L1Prefiring": "lnN",
    "scalevar_7pt": "lnN",
    "scalevar_3pt": "lnN",
    "mu_trigger": "lnN",
    "mu_isoweight": "lnN",
    "mu_idweight": "lnN",
    "HEM18": "lnN",
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
    "VectorZPrimeToQQ_M100": ["VectorZPrimeToQQ_M100"],
    "VectorZPrimeToQQ_M125": ["VectorZPrimeToQQ_M125"],
    "VectorZPrimeToQQ_M150": ["VectorZPrimeToQQ_M150"],
    "VectorZPrimeToQQ_M200": ["VectorZPrimeToQQ_M200"],
    "VectorZPrimeToQQ_M250": ["VectorZPrimeToQQ_M250"],
    # "VectorZPrimeToQQ_M300" : ["VectorZPrimeToQQ_M300"],
}

if args.all_signals:
    signals = [
        "m50",
        "m75",
        "m100",
        "m125",
        "m150",
        "m200",
        "m250",
    ]
else:
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
    # print(tf_MCtempl_vals)
    df = pd.DataFrame([])
    df["msd"] = msdpts.reshape(-1)
    df["pt"] = ptpts.reshape(-1)
    df["MCTF"] = tf_MCtempl_vals.reshape(-1)
    # print(df['MCTF'])
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
    fourptbins=False,
):

    if muon:
        hist_str = f"CR1_{sample}_{tagger}_{region}"
        root_file = args.root_file_mu
    else:
        hist_str = f"SR_{sample}_ptbin{ptbin}_{tagger}_{region}"
        root_file = args.root_file
    
    if syst is not None:
        hist_str = hist_str + "__" + syst
    # print(hist_str)
    
    with uproot.open(root_file) as f:
        # print(f.keys())
        hist = f[hist_str]
    hist_values = hist.values()
    hist_variances = hist.variances()
    hist_edges = hist.axis().edges()
    if scaledown:
        hist_values *= 1e-2

    if fourptbins and ptbin == 3:
        with uproot.open(args.root_file) as f:
            # print(f.keys())
            hist = f[hist_str.replace("ptbin3", "ptbin4")]
        hist_values += hist.values()
        hist_variances += hist.variances()

    if muon:
        hist_key = "msd_muon"
    else:
        hist_key = "msd"
    return (hist_values, hist_edges, hist_key, hist_variances)


def th1_to_numpy(path, label="msd"):
    with uproot.open(path) as file:
        th1d = file[label]
        _hist, _ = th1d.to_numpy()
    return _hist


def shape_to_num(
    region, sName, ptbin, syst_down_up, mask, muon=False, bound=0.5, inflate=False
):
    # print(sName)

    _nom = get_templ(region, sName, ptbin, tagger)
    # _nom = th1_to_numpy(path)

    # if template is very small don't add unc
    if _nom[0] is None:
        return None
    _nom_rate = np.sum(_nom[0] * mask)
    if _nom_rate < 0.1:
        return 1.0
    # ignore one sided for now
    _one_side = (
        None  # get_templ(f, region, sName, ptbin, syst=syst, muon=muon, nowarn=True)
    )
    _up = get_templ(
        region, sName, ptbin, tagger, syst=syst_down_up[1], muon=muon, nowarn=True
    )

    # _up = th1_to_numpy(path)

    _down = get_templ(
        region, sName, ptbin, tagger, syst=syst_down_up[0], muon=muon, nowarn=True
    )
    # _down = th1_to_numpy(path)
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
            # print("_up_rate",_up_rate)
            # print("_down_rate",_down_rate)
            _diff = np.abs(_up_rate - _nom_rate) + np.abs(_down_rate - _nom_rate)
            magnitude = _diff / (2.0 * _nom_rate)
        else:
            raise NotImplementedError
    if bound is not None:
        magnitude = min(magnitude, bound)
    # inflate uncs while debugging what went wrong
    if inflate:
        magnitude *= 10
    # print(magnitude)
    return 1.0 + magnitude


def test_rhalphabet(tmpdir, sig, throwPoisson=False):
    jec = rl.NuisanceParameter("CMS_jec", "lnN")
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
    sys_shape_dict["HEM18"] = rl.NuisanceParameter(
        "CMS_HEM_{}".format(args.year), sys_types["HEM18"]
    )
    # don't have mu for now
    sys_shape_dict["mu_trigger"] = rl.NuisanceParameter(
        "CMS_mu_trigger_{}".format(args.year), sys_types["mu_trigger"]
    )
    sys_shape_dict["mu_isoweight"] = rl.NuisanceParameter(
        "CMS_mu_isoweight_{}".format(args.year), sys_types["mu_isoweight"]
    )
    sys_shape_dict["mu_idweight"] = rl.NuisanceParameter(
        "CMS_mu_idweight_{}".format(args.year), sys_types["mu_idweight"]
    )
    for EW_syst in ['d1kappa_EW', 'Z_d2kappa_EW', 'Z_d3kappa_EW', 'd1K_NLO', 'd2K_NLO', 'd3K_NLO']:
        sys_shape_dict[EW_syst] = rl.NuisanceParameter('CMS_{}'.format(EW_syst), sys_types[EW_syst])

    # sys_shape_dict['scalevar_7pt'] = rl.NuisanceParameter('CMS_th_scale7pt', sys_types['scalevar_7pt'])
    # sys_shape_dict['scalevar_3pt'] = rl.NuisanceParameter('CMS_th_scale3pt', sys_types['scalevar_3pt'])

    sys_eleveto = rl.NuisanceParameter("CMS_e_veto_{}".format(args.year), "lnN")
    sys_muveto = rl.NuisanceParameter("CMS_m_veto_{}".format(args.year), "lnN")
    sys_tauveto = rl.NuisanceParameter("CMS_tau_veto_{}".format(args.year), "lnN")

    sys_veff = rl.NuisanceParameter("CMS_veff_{}".format(args.year), "lnN")

    sys_lumi = rl.NuisanceParameter("CMS_lumi", "lnN")
    sys_lumi_correlated = rl.NuisanceParameter("CMS_lumi_13TeV_correlated", "lnN")
    sys_lumi_1718 = rl.NuisanceParameter("CMS_lumi_13TeV_1718", "lnN")

    tqqeffSF = rl.IndependentParameter("tqqeffSF", 1.0, 0, 10)
    tqqnormSF = rl.IndependentParameter("tqqnormSF", 1.0, 0, 10)
    tqqeffSF_highbvl = rl.IndependentParameter("tqqeffSF_highbvl", 1.0, 0, 10)
    tqqnormSF_highbvl = rl.IndependentParameter("tqqnormSF_highbvl", 1.0, 0, 10)
    tqqeffSF_lowbvl = rl.IndependentParameter("tqqeffSF_lowbvl", 1.0, 0, 10)
    tqqnormSF_lowbvl = rl.IndependentParameter("tqqnormSF_lowbvl", 1.0, 0, 10)

    # with open(args.pickle, "rb") as f:
    #    df = pickle.load(f)
    ptbins = np.array([525, 575, 625, 700, 800, 1200])
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
    ptscaled = (ptpts - 525.0) / (1200.0 - 525.0)
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
        pass_regs = ["pass_highbvl", "pass_lowbvl"]
        fail_regs = ["fail", "fail"]
 
        if args.qcd_ftest:
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
        print(qcdmodel, qcdpass, qcdfail, rpass, rfail) 
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
        tf_MCtempls = []
        tf_MCtempl_params = []
        for i, qcdmodel in enumerate(qcdmodels):
            degsMC = tuple([int(s) for s in [args.iptMC, args.irhoMC]])
            _initsMC = np.ones(tuple(n + 1 for n in degsMC))
            log.debug(f"Initializing MCTF `{qcdmodel.name}` with n_pt={args.iptMC} and n_rho={args.irhoMC}")
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
                        rl.IndependentParameter(f"qcdparam{i}_ptbin{ptbin}_msdbin{mbin}", 0)
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
        degs = tuple([int(s) for s in [args.ipt, args.irho]])
        _inits = np.ones(tuple(n + 1 for n in degs))
        log.debug(f"Initializing TF {i} (data) with n_pt={args.ipt} and n_rho={args.irho}")
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
    log.info(f"Signals: {siggy}, {bsiggy}")
    model.t2w_config = ("-P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel  --PO verbose "
                        f"--PO 'map=.*/*{siggy}*:r_q[1,-5,5]'  --PO 'map=.*/*{bsiggy}*:r_b[1,-5,5]'"
                        )

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

            #if not args.ftest:
            templates[siggy] = get_templ(
                    region,
                    short_to_long[siggy],
                    ptbin,
                    tagger,
                    fourptbins=args.four_pt_bins,
                )
            templates[bsiggy] = get_templ(
                    region,
                    short_to_long[bsiggy],
                    ptbin,
                    tagger,
                    fourptbins=args.four_pt_bins,
                )
            mask = validbins[ptbin].copy()

            if args.qcd_ftest:
                include_samples = ["zqq"]  # qcd here?
            elif args.h_sensitivity:
                include_samples = ["wqq", "zqq", "zbb", "tt", "wlnu", "dy", "st", "hbb",]
            else:
                include_samples = ["wqq", "zqq", "zbb", "tt", "wlnu", "dy", "st", "hbb", siggy, bsiggy]

            for sName in include_samples:
                # some mock expectations
                templ = templates[sName]
                logging.debug(f"Adding sample={sName} in ptbin={ptbin}, region={region}.")

                if args.qcd_ftest:
                    stype = rl.Sample.SIGNAL if sName == "zqq" else rl.Sample.BACKGROUND
                    # templ[0] = templ[0]*1e-4 #Scale down signal?
                elif args.h_sensitivity:
                    stype = rl.Sample.SIGNAL if sName == "hbb" else rl.Sample.BACKGROUND
                else:
                    stype = rl.Sample.SIGNAL if sName in [siggy, bsiggy] else rl.Sample.BACKGROUND
                sample = rl.TemplateSample(ch.name + "_" + sName, stype, templ)

                def smorph(templ, sName):
                    if templ is None:
                        return None
                    if sName not in ["qcd", "dy", "wlnu", "tt", "st"]:
                        return MorphHistW2(templ).get(
                            shift=SF[year]['shift_SF']/smass('wqq') * smass(sName),
                            smear=SF[year]['smear_SF']
                        )
                    else:
                        return templ

                ##https://github.com/nsmith-/rhalphalib/blob/master/rhalphalib/template_morph.py#L45-L58 how do i do this on ROOT templates?
                if args.do_systematics:
                    sample.setParamEffect(sys_lumi, lumi_dict_unc[args.year])
                    sample.setParamEffect(
                        sys_lumi_correlated, lumi_correlated_dict_unc[args.year]
                    )
                    if args.year != "2016":
                        sample.setParamEffect(
                            sys_lumi_1718, lumi_1718_dict_unc[args.year]
                        )
                    sample.setParamEffect(sys_eleveto, 1.005)
                    sample.setParamEffect(sys_muveto, 1.005)
                    sample.setParamEffect(sys_tauveto, 1.005)

                    sample.autoMCStats(lnN=True)

                    sys_names = [
                        "JES",
                        "JER",
                        "jet_trigger",
                        "pileup_weight",
                        "L1Prefiring",
                        #'Z_d2kappa_EW', 'Z_d3kappa_EW', 
                        'd1kappa_EW', 'd1K_NLO', 'd2K_NLO', 'd3K_NLO',
                        #'scalevar_7pt', 'scalevar_3pt',
                        #'UES','btagEffStat', 'btagWeight',
                    ]
                    sys_name_updown = {
                        "JES": ["jesTotaldown", "jesTotalup"],
                        "JER": ["jerdown", "jerup"],
                        "pileup_weight": ["pudown", "puup"],
                        "jet_trigger": ["stat_dn", "stat_up"],
                        "L1Prefiring": ["L1PreFiringdown", "L1PreFiringup"],
                        "d1kappa_EW" : ["d1kappa_EW_down", "d1kappa_EW_up"],
                        "d1K_NLO" : ["d1K_NLO_down", "d1K_NLO_up"],
                        "d2K_NLO" : ["d2K_NLO_down", "d2K_NLO_up"],
                        "d3K_NLO" : ["d3K_NLO_down", "d3K_NLO_up"],
                        "Z_d2kappa_EW" : ["Z_d2kappa_EW_down","Z_d2kappa_EW_up"],
                        "Z_d3kappa_EW" : ["Z_d3kappa_EW_down","Z_d3kappa_EW_up"],
                    }
                    if stype == rl.Sample.SIGNAL and not args.ftest:
                        sName = short_to_long[sName]
                    for sys_name in sys_names:
                        if (
                            ("NLO" in sys_name) or ("EW" in sys_name)
                        ) and not sName in ["zqq", "wqq"]:
                            continue
                        if ("Z_d" in sys_name) and sName not in ["zqq","dy"]:
                            continue
                        if sys_shape_dict[sys_name].combinePrior == "lnN":
                            _sys_ef = shape_to_num(
                                region,
                                sName,
                                ptbin,
                                sys_name_updown[sys_name],
                                mask,
                                bound=None if "scalevar" not in sys_name else 0.25,
                                inflate=True,
                            )
                            if _sys_ef is None:
                                continue
                            sample.setParamEffect(sys_shape_dict[sys_name], _sys_ef)

                    if sName not in [
                        "qcd",
                        "dy",
                        "wlnu",
                        "tt",
                        "st",
                    ]:
                        sample.scale(SF[args.year]["V_SF"])
                        sample.setParamEffect(
                            sys_veff,
                            1.0 + SF[args.year]["V_SF_ERR"] / SF[args.year]["V_SF"],
                        )
                    ###SFs complicated by high-purity bb region...fully insitu using Zbb?

                else:
                    sample.setParamEffect(sys_lumi, lumi_dict_unc[args.year])

                ch.addSample(sample)

            if not args.pseudo:
                data_obs = get_templ(
                    region, f"JetHT_2017", ptbin, tagger, fourptbins=args.four_pt_bins
                )
                if throwPoisson:
                    yields = np.random.poisson(yields)
            else:
                yields = []
                if args.qcd_ftest:
                    include_samples = ["QCD"]
                elif args.h_sensitivity:
                    include_samples = ["wqq", "zqq", "zbb", "tt", "wlnu", "dy", "st", "hbb","QCD"]
                print(include_samples)
                for sName in include_samples:
                    _sample = get_templ(
                        region, sName, ptbin, tagger, fourptbins=args.four_pt_bins
                    )
                    _sample_yield = _sample[0]
                    if args.scale_qcd:
                        print(sName, region)
                        if "pass" in region:
                            dummyqcd = rl.TemplateSample(
                                "dummyqcd", rl.Sample.BACKGROUND, _sample
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
                            print(
                                "qcdscale needed to match mcstat uncs: using poisson:",
                                scale,
                            )
                            # _sample_yield = _sample_yield.copy()*1./scale
                        else:
                            scale = scale_pass[ptbin]
                        yields.append(_sample_yield * scale)
                    else:
                        yields.append(_sample_yield)
                yields = np.sum(np.array(yields), axis=0)
                print("ptbin/region", ptbin, region, yields)
                if throwPoisson:
                    yields = np.random.poisson(yields)

                data_obs = (yields, msd.binning, msd.name)

            ch.setObservation(data_obs[0:3])

            # drop bins outside rho validity
            # validbins[ptbin][0:2] = False
            mask = validbins[ptbin]
            ch.mask = mask

    ###################################
    # Build QCD TF temples
    log.info("Building QCD TF templates")    
    if args.tworeg and not args.qcd_ftest:
        for ptbin in range(npt):
            failCh = model[f"ptbin{ptbin}fail"]
            passChpass = model[f"ptbin{ptbin}passhighbvl"]
            passChfail = model[f"ptbin{ptbin}passlowbvl"]

            # Fail region
            qcdparams = np.array(
                [
                    rl.IndependentParameter("qcdparam_ptbin%d_msdbin%d" % (ptbin, i), 0)
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
                f"ptbin{ptbin}fail_{args.year}_qcd", rl.Sample.BACKGROUND, msd, scaledparams
            )
            failCh.addSample(fail_qcd)

            # Transfer factor templates
            passpass_qcd = rl.TransferFactorSample(
                f"ptbin{ptbin}passhighbvl_{args.year}_qcd",
                rl.Sample.BACKGROUND,
                all_tf_params[0][ptbin, :],
                fail_qcd,
            )
            passChpass.addSample(passpass_qcd)

            passfail_qcd = rl.TransferFactorSample(
                f"ptbin{ptbin}passlowbvl_{args.year}_qcd",
                rl.Sample.BACKGROUND,
                all_tf_params[1][ptbin, :],
                fail_qcd,
            )
            passChfail.addSample(passfail_qcd)
    else:
        for ptbin in range(npt):
            passkey = f"ptbin{ptbin}pass"
            if args.highbvl:
                passkey = f"ptbin{ptbin}passhighbvl"
            elif args.lowbvl:
                passkey = f"ptbin{ptbin}passlowbvl"
            failkey = f"ptbin{ptbin}fail"
            failCh = model[failkey]
            passCh = model[passkey]

            qcdparams = np.array(
                [
                    rl.IndependentParameter("qcdparam_ptbin%d_msdbin%d" % (ptbin, i), 0)
                    for i in range(msd.nbins)
                ]
            )
            initial_qcd = failCh.getObservation().astype(
                float
            )  # was integer, and numpy complained about subtracting float from it
            print(initial_qcd)
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
                f"ptbin{ptbin}fail_{args.year}_qcd", rl.Sample.BACKGROUND, msd, scaledparams
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

    if args.muonCR:
        if args.tworeg:
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
                sumPass_highbvl = tqqpass_highbvl.getExpectation(nominal=True).sum() + stqqpass_highbvl.getExpectation(nominal=True).sum()
                sumPass_lowbvl = tqqpass_lowbvl.getExpectation(nominal=True).sum() + stqqpass_lowbvl.getExpectation(nominal=True).sum()
                sumFail = tqqfail.getExpectation(nominal=True).sum() + stqqfail.getExpectation(nominal=True).sum()
                tqqPF_highbvl = sumPass_highbvl / sumFail
                tqqPF_lowbvl  = sumPass_lowbvl / sumFail
                tqqpass_highbvl.setParamEffect(tqqeffSF_highbvl, 1 * tqqeffSF_highbvl)
                tqqpass_lowbvl .setParamEffect(tqqeffSF_lowbvl, 1 * tqqeffSF_lowbvl)
                tqqfail        .setParamEffect(tqqeffSF_highbvl, (1 - tqqeffSF_highbvl) * tqqPF_highbvl + 1)
                tqqfail        .setParamEffect(tqqeffSF_lowbvl, (1 - tqqeffSF_lowbvl) * tqqPF_lowbvl + 1)
                tqqpass_highbvl.setParamEffect(tqqnormSF_highbvl, 1 * tqqnormSF_highbvl)
                tqqpass_lowbvl .setParamEffect(tqqnormSF_lowbvl, 1 * tqqnormSF_lowbvl)
                tqqfail        .setParamEffect(tqqnormSF_highbvl, 1 * tqqnormSF_highbvl) 
                tqqfail        .setParamEffect(tqqnormSF_lowbvl, 1 * tqqnormSF_lowbvl)
 

                stqqpass_highbvl.setParamEffect(tqqeffSF_highbvl, 1 * tqqeffSF_highbvl)
                stqqpass_lowbvl .setParamEffect(tqqeffSF_lowbvl, 1 * tqqeffSF_lowbvl)
                stqqfail        .setParamEffect(tqqeffSF_highbvl, (1 - tqqeffSF_highbvl) * tqqPF_highbvl + 1)
                stqqfail        .setParamEffect(tqqeffSF_lowbvl, (1 - tqqeffSF_lowbvl) * tqqPF_lowbvl + 1)
                stqqpass_highbvl.setParamEffect(tqqnormSF_highbvl, 1 * tqqnormSF_highbvl)
                stqqpass_lowbvl .setParamEffect(tqqnormSF_lowbvl, 1 * tqqnormSF_lowbvl)
                stqqfail        .setParamEffect(tqqnormSF_highbvl, 1 * tqqnormSF_highbvl)
                stqqfail        .setParamEffect(tqqnormSF_lowbvl, 1 * tqqnormSF_lowbvl)
        else:
            for ptbin in range(npt):
                failCh = model['ptbin{}fail{}'.format(ptbin, year)]
                passCh = model['ptbin{}pass{}'.format(ptbin, year)]
                tqqpass = passCh['tqq']
                tqqfail = failCh['tqq']
                stqqpass = passCh['stqq']
                stqqfail = failCh['stqq']
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
                        region, "tt", 0, tagger, fourptbins=args.four_pt_bins,muon=True,
                    ),
                    "st": get_templ(
                        region, "st", 0, tagger, fourptbins=args.four_pt_bins,muon=True,
                    ),
                    "qcd": get_templ(
                        region, "QCD", 0, tagger, fourptbins=args.four_pt_bins,muon=True,
                    ),
                    "wlnu": get_templ(
                        region, "wlnu", 0, tagger, fourptbins=args.four_pt_bins,muon=True,
                    ),
                    "dy": get_templ(
                        region, "dy", 0, tagger, fourptbins=args.four_pt_bins,muon=True,
                    ),
                    "wqq": get_templ(
                        region, "wqq", 0, tagger, fourptbins=args.four_pt_bins,muon=True,
                    ),
                    "zqq": get_templ(
                        region, "zqq", 0, tagger, fourptbins=args.four_pt_bins,muon=True,
                    ),
            }
    
            include_samples = ["wqq", "zqq", "tt", "wlnu", "dy", "st", "qcd"]
            for sName in include_samples:
                templ = templates[sName]
                stype = rl.Sample.BACKGROUND
                sample = rl.TemplateSample(ch.name + "_" + sName, stype, templ)
                if args.do_systematics:
                    sample.setParamEffect(sys_lumi, lumi_dict[year])
                    sample.setParamEffect(sys_lumi_correlated, lumi_correlated_dict[year])
                    if year != '2016':
                        sample.setParamEffect(sys_lumi_1718, lumi_1718_dict[year])
                    sample.setParamEffect(sys_eleveto, 1.005)
                    sample.setParamEffect(sys_tauveto, 1.005)

                    if sName in ["qcd"]:
                        continue

                    sys_names = [
                        'JES', 'JER', 'UES', 'mu_trigger', 'mu_isoweight', 'mu_idweight', #'btagEffStat', 'btagWeight', 'pileup_weight',
                        #'Z_d2kappa_EW', 'Z_d3kappa_EW',  
                        'd1kappa_EW', 'd1K_NLO', 'd2K_NLO', 'd3K_NLO',
                        'L1Prefiring',
                    ]

                    for sys_name in sys_names:
                        if (
                            ("NLO" in sys_name) or ("EW" in sys_name)
                        ) and not sName in ["zqq", "wqq"]:
                            continue
                        if ("Z_d" in sys_name) and sName not in ["zqq","dy"]:
                            continue
                        if sys_shape_dict[sys_name].combinePrior == "lnN":
                            _sys_ef = shape_to_num(
                                region,
                                sName,
                                ptbin,
                                sys_name_updown[sys_name],
                                mask,
                                bound=None if "scalevar" not in sys_name else 0.25,
                                inflate=True,
                            )
                            if _sys_ef is None:
                                continue
                            sample.setParamEffect(sys_shape_dict[sys_name], _sys_ef)
                else:
                    sample.setParamEffect(sys_lumi, lumi_dict_unc[args.year])
    
                ch.addSample(sample)
    
            
            data_obs = get_templ(
                    region, f"SingleMuon_{args.year}", 0, tagger, fourptbins=args.four_pt_bins,muon=True,
            )
            ch.setObservation(data_obs[0:3])
        if args.tworeg:
            
            tqqpass_highbvl = model[f"muonCR{pass_regs[0].replace('_','')}"]["tt"]
            tqqpass_lowbvl  = model[f"muonCR{pass_regs[1].replace('_','')}"]["tt"]
            tqqfail         = model[f"muonCR{fail_regs[0].replace('_','')}"]["tt"]
            stqqpass_highbvl = model[f"muonCR{pass_regs[0].replace('_','')}"]["st"]
            stqqpass_lowbvl  = model[f"muonCR{pass_regs[1].replace('_','')}"]["st"]
            stqqfail         = model[f"muonCR{fail_regs[0].replace('_','')}"]["st"]

            tqqPF_highbvl   = (tqqpass_highbvl.getExpectation(nominal=True).sum()  + tqqpass_highbvl.getExpectation(nominal=True).sum()) / (tqqfail.getExpectation(nominal=True).sum() + stqqfail.getExpectation(nominal=True).sum() ) 
            tqqPF_lowbvl    = (tqqpass_lowbvl.getExpectation(nominal=True).sum() + stqqpass_lowbvl.getExpectation(nominal=True).sum()) / (tqqfail.getExpectation(nominal=True).sum() + stqqfail.getExpectation(nominal=True).sum())

            tqqpass_highbvl .setParamEffect(tqqeffSF_highbvl, 1 * tqqeffSF_highbvl)
            tqqpass_lowbvl  .setParamEffect(tqqeffSF_lowbvl, 1 * tqqeffSF_lowbvl)
            tqqfail         .setParamEffect(tqqeffSF_highbvl, (1 - tqqeffSF_highbvl) * tqqPF_highbvl + 1)
            tqqfail         .setParamEffect(tqqeffSF_lowbvl,  (1 - tqqeffSF_lowbvl) * tqqPF_lowbvl + 1)
            tqqpass_highbvl .setParamEffect(tqqnormSF_highbvl, 1 * tqqnormSF_highbvl)
            tqqpass_lowbvl  .setParamEffect(tqqnormSF_lowbvl, 1 * tqqnormSF_lowbvl)
            tqqfail         .setParamEffect(tqqnormSF_highbvl, 1 * tqqnormSF_highbvl)
            tqqfail         .setParamEffect(tqqnormSF_lowbvl, 1 * tqqnormSF_lowbvl)
 
            stqqpass_highbvl .setParamEffect(tqqeffSF_highbvl, 1 * tqqeffSF_highbvl)
            stqqpass_lowbvl  .setParamEffect(tqqeffSF_lowbvl, 1 * tqqeffSF_lowbvl)
            stqqfail         .setParamEffect(tqqeffSF_highbvl, (1 - tqqeffSF_highbvl) * tqqPF_highbvl + 1)
            stqqfail         .setParamEffect(tqqeffSF_lowbvl,  (1 - tqqeffSF_lowbvl) * tqqPF_lowbvl + 1)
            stqqpass_highbvl .setParamEffect(tqqnormSF_highbvl, 1 * tqqnormSF_highbvl)
            stqqpass_lowbvl  .setParamEffect(tqqnormSF_lowbvl, 1 * tqqnormSF_lowbvl)
            stqqfail         .setParamEffect(tqqnormSF_highbvl, 1 * tqqnormSF_highbvl)
            stqqfail         .setParamEffect(tqqnormSF_lowbvl, 1 * tqqnormSF_lowbvl)

        else: 
            tqqpass = model["muonCRpass"]["tt"]
            tqqfail = model["muonCRfail"]["tt"]
            stqqpass = model["muonCRpass"]["st"]
            stqqfail = model["muonCRfail"]["st"]
            tqqPF = (tqqpass.getExpectation(nominal=True).sum() + stqqpass.getExpectation(nominal=True).sum())/ ( tqqfail.getExpectation(nominal=True).sum() + tqqfail.getExpectation(nominal=True).sum())
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
    print(conf_dict)
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
    from rich.prompt import Prompt, Confirm

    global opath
    startopath = f"{args.opath}/{tagger}/ipt{args.ipt}_irho{args.irho}"
    os.system(f"cp rhalphalib_zprime.py {startopath}/rhalphalib_zprime.py")
    for sig in signals:
        opath = f"{startopath}/{sig}/"
        if os.path.exists(opath):
            q_overwrite = Confirm.ask(f"Path: {opath} exists. Overwrite?")
            if not q_overwrite:
                print(f"Remove with \nrm -rf {opath}")
                sys.exit()
            else:
                os.system(f"rm -rf {opath}")
        os.makedirs(opath)
        test_rhalphabet(opath, sig, args.throwPoisson)
