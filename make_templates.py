import argparse
import ROOT
from array import array
from common import * 
import json

parser = argparse.ArgumentParser(description='Makes templates based off bamboo output.')
parser.add_argument("--root_path", action='store', type=str, default="",help="Path to ROOT holding templates for signal region.")
parser.add_argument("--root_path_mu", action='store', type=str, default="",help="Path to ROOT holding single muon templates.")
parser.add_argument("--is_blinded", action='store_true', help="Blinded dataset.")
parser.add_argument("--year", action='store', choices=["2016APV", "2016", "2017", "2018","fullrun2"], type=str, help="Year to run on")
args = parser.parse_args()


with open("xsec.json") as f:
    xsec_dict = json.load(f)




def make_templates(path,region,sample,ptbin,tagger,syst=None,muon=False,nowarn=False,year="2017"):

    def get_factor(f,subsample):
        factor = 0.
        tree = f.Get("Runs")
        genEventSumw_buffer = array('d',[0.0])
        tree.SetBranchAddress("genEventSumw",genEventSumw_buffer)
        sum_genEventSumw = 0.
        for entry in range(tree.GetEntries()):
            tree.GetEntry(entry)
            sum_genEventSumw += genEventSumw_buffer[0]

        lumi = lumi_dict[year]
        if args.is_blinded:
            lumi /= 10.
        xsec = xsec_dict[subsample]
        if sum_genEventSumw > 0.:
            factor = xsec*lumi/sum_genEventSumw
        else:
            raise RuntimeError(f"Factor for sample {subsample} <= 0")
        return factor

    master_hist = None

    if muon:
        #if "lowbvl" in region or "highbvl" in region:
        #    hist_str = f"CR1_{tagger}_{region}"
        #else: 
        #    hist_str = f"CR1__{tagger}_{region}"
        hist_str = f"CR1_{tagger}_{region}"
        master_hist_name = hist_str.replace(f"_{tagger}",f"_{sample}_{tagger}")
        file0 = ROOT.TFile(f"{path}/{sample_maps_mu[sample][0]}.root", "READ")
    else:
        hist_str = f"SR_ptbin{ptbin}_{tagger}_{region}"
        master_hist_name = hist_str.replace(f"_ptbin{ptbin}",f"_{sample}_ptbin{ptbin}")
        file0 = ROOT.TFile(f"{path}/{sample_maps[sample][0]}.root", "READ")
    if syst is not None:
        hist_str += f"__{syst}"
        master_hist_name += f"__{syst}"
    print("master_hist_name",master_hist_name)
    hist0 = file0.Get(hist_str)
    master_hist = hist0.Clone(master_hist_name)#+"_"+sample)
    master_hist.Reset()
    for subsample in sample_maps[sample]:
        file = ROOT.TFile(f"{path}/{subsample}.root")
        hist = file.Get(hist_str)
        if not("JetHT" in subsample or "SingleMuon" in subsample):
            factor = get_factor(file,subsample)
            hist.Scale(factor)
        master_hist.Add(hist)  # Add to master, uncertainties are handled automatically
        file.Close()
    
    #hist_str = hist_str.replace(f"_ptbin{ptbin}",f"_{sample}_ptbin{ptbin}")
    master_hist.SetTitle(master_hist_name+";"+master_hist_name+";;")
    output_file.cd()
    master_hist.Write()
    file0.Close()
    return 

if args.root_path:
    output_file = ROOT.TFile(args.root_path+f"/TEMPLATES{'_blind' if args.is_blinded else ''}{'_fullrun2' if 'run2' in args.year else ''}.root", "RECREATE")
    print("Making SR templates from path ",args.root_path)
    for isamp,isamplist in sample_maps.items():
        if "SingleMuon" in isamp: continue
        for tagger in ["pnmd2prong"]:
            for region in ["fail_T","pass_T_bvl_fail_L","pass_T_bvl_pass_L","pass_T_bvl_fail_T","pass_T_bvl_pass_T","pass_T_bvl_fail_VT","pass_T_bvl_pass_VT"]:
                for iptbin in range(0,5):
                    print(f"Making hists for sample {isamp}, tagger {tagger}, region {region}, iptbin {iptbin}")
                    make_templates(args.root_path,region,isamp,iptbin,tagger,syst=None,muon=False,nowarn=False,year="2017")
                    if "JetHT" in isamp: continue
                    for syst in sys_names:
                        if "muo" in syst: continue
                        if syst in ['W_d2kappa_EW', 'W_d3kappa_EW'] and not isamp in ["wqq","wlnu"]: continue
                        if syst in ['Z_d2kappa_EW', 'Z_d3kappa_EW'] and not isamp in ["zqq","dy"]: continue
                        if syst in ['d1kappa_EW','d1K_NLO','d2K_NLO','d3K_NLO'] and isamp not in ["wqq","wlnu","zqq","dy",]: continue
                        syst_name_up = sys_name_updown[syst][0]
                        syst_name_down = sys_name_updown[syst][1]
 
                        if "year" in syst_name_up:
                            syst_name_up = syst_name_up.replace('year',args.year)
                            syst_name_down = syst_name_down.replace('year',args.year)
                        make_templates(args.root_path,region,isamp,iptbin,tagger,syst=syst_name_up,muon=False,nowarn=False,year="2017")
                        make_templates(args.root_path,region,isamp,iptbin,tagger,syst=syst_name_down,muon=False,nowarn=False,year="2017")
                #break
    

if args.root_path_mu:
    output_file = ROOT.TFile(args.root_path_mu+f"/TEMPLATES{'_blind' if args.is_blinded else ''}.root", "RECREATE")
    print("Making CR templates from path ",args.root_path_mu)
    for isamp,isamplist in sample_maps_mu.items():
        for tagger in ["pnmd2prong_0p01"]:
            for region in ["fail","pass_lowbvl","pass_highbvl",]:
                make_templates(args.root_path_mu,region,isamp,0,tagger,syst=None,muon=True,nowarn=False,year="2017")
                if "SingleMuon" in isamp: continue
                for syst in sys_names:
                    if syst in ["jet_trigger","muoiso"]: continue
                    if syst in ['W_d2kappa_EW', 'W_d3kappa_EW'] and not isamp in ["wqq","wlnu"]: continue
                    if syst in ['Z_d2kappa_EW', 'Z_d3kappa_EW'] and not isamp in ["zqq","dy"]: continue
                    if syst in ['d1kappa_EW','d1K_NLO','d2K_NLO','d3K_NLO'] and isamp not in ["wqq","wlnu","zqq","dy",]: continue 
                    print(isamp,isamplist,0,tagger,sys_name_updown[syst][0],)
                    make_templates(args.root_path_mu,region,isamp,0,tagger,syst=sys_name_updown[syst][0],muon=True,nowarn=False,year="2017")
                    make_templates(args.root_path_mu,region,isamp,0,tagger,syst=sys_name_updown[syst][1],muon=True,nowarn=False,year="2017")

output_file.Close()

