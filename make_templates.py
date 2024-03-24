import argparse
import ROOT
from array import array
import json

parser = argparse.ArgumentParser(description='Makes templates based off bamboo output.')
parser.add_argument("--root_path", action='store', type=str, default="",help="Path to ROOT holding templates for signal region.")
parser.add_argument("--root_path_mu", action='store', type=str, default="",help="Path to ROOT holding single muon templates.")
parser.add_argument("--is_blinded", action='store_true', help="Blinded dataset.")
parser.add_argument("--year", action='store', type=str, help="Year to run on : one of 2016APV, 2016, 2017, 2018.")
args = parser.parse_args()

lumi_dict = {
    "2017" : 41500
}

with open("xsec.json") as f:
    xsec_dict = json.load(f)

sample_maps = {
    "QCD" : ["QCD_HT500to700","QCD_HT700to1000","QCD_HT1000to1500","QCD_HT1500to2000","QCD_HT2000toInf"],
    "wqq" : ["WJetsToQQ_HT-600to800","WJetsToQQ_HT-800toInf"],
    "zqq" : ["ZJetsToQQ_HT-600to800","ZJetsToQQ_HT-800toInf"],
    "zbb" : ["ZJetsToBB_HT-600to800","ZJetsToBB_HT-800toInf"],
    "tt"  : ["TTTo2L2Nu","TTToHadronic","TTToSemiLeptonic"],
    "dy"  : ["DYJetsToLL_HT-400To600","DYJetsToLL_HT-600To800","DYJetsToLL_HT-800To1200","DYJetsToLL_HT-1200To2500"],
    "st"  : ["ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8","ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8","ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8","ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8"],
    "hbb" : ["GluGluHToBB","VBFHToBB","ttHToBB","ggZHToBB","ZHToBB","WplusH_HToBB","WminusH_HToBB",],
    "wlnu" : ["WJetsToLNu_HT400to600","WJetsToLNu_HT600to800","WJetsToLNu_HT800to1200","WJetsToLNu_HT1200to2500","WJetsToLNu_HT2500toInf"],
    "JetHT_2017" : ["JetHT_Run2017B","JetHT_Run2017C","JetHT_Run2017D","JetHT_Run2017E","JetHT_Run2017F"],
    "zpqq50" : ["VectorZPrimeToQQ_M50"],
    "zpqq75" : ["VectorZPrimeToQQ_M75"],
    "zpqq100" : ["VectorZPrimeToQQ_M100"],
    "zpqq125" : ["VectorZPrimeToQQ_M125"],
    "zpqq150" : ["VectorZPrimeToQQ_M150"],
    "zpqq200" : ["VectorZPrimeToQQ_M200"],
    "zpqq250" : ["VectorZPrimeToQQ_M250"],
    #"VectorZPrimeToQQ_M300" : ["VectorZPrimeToQQ_M300"],
    "zpbb50" : ["VectorZPrimeToBB_M50"],
    "zpbb75" : ["VectorZPrimeToBB_M75"],
    "zpbb100" : ["VectorZPrimeToBB_M100"],
    "zpbb125" : ["VectorZPrimeToBB_M125"],
    "zpbb150" : ["VectorZPrimeToBB_M150"],
    "zpbb200" : ["VectorZPrimeToBB_M200"],
    "zpbb250" : ["VectorZPrimeToBB_M250"],
    "SingleMuon_2017" : ["SingleMuon_Run2017B","SingleMuon_Run2017C","SingleMuon_Run2017D","SingleMuon_Run2017E","SingleMuon_Run2017F"]
}

sample_maps_mu = {

    "QCD" : ["QCD_HT500to700","QCD_HT700to1000","QCD_HT1000to1500","QCD_HT1500to2000","QCD_HT2000toInf"],
    "wqq" : ["WJetsToQQ_HT-600to800","WJetsToQQ_HT-800toInf"],
    "zqq" : ["ZJetsToQQ_HT-600to800","ZJetsToQQ_HT-800toInf"],
    "tt"  : ["TTTo2L2Nu","TTToHadronic","TTToSemiLeptonic"],
    "dy"  : ["DYJetsToLL_HT-400To600","DYJetsToLL_HT-600To800","DYJetsToLL_HT-800To1200","DYJetsToLL_HT-1200To2500"],
    "st"  : ["ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8","ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8","ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8","ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8"],
    "wlnu" : ["WJetsToLNu_HT400to600","WJetsToLNu_HT600to800","WJetsToLNu_HT800to1200","WJetsToLNu_HT1200to2500","WJetsToLNu_HT2500toInf"],
    "SingleMuon_2017" : ["SingleMuon_Run2017B","SingleMuon_Run2017C","SingleMuon_Run2017D","SingleMuon_Run2017E","SingleMuon_Run2017F"]
}
sys_names = [
    'JES', 'JER', 'jet_trigger','pileup_weight','L1Prefiring',
    'W_d2kappa_EW', 'W_d3kappa_EW', 'd1kappa_EW', 'd1K_NLO', 'd2K_NLO', 'd3K_NLO',
    #'scalevar_7pt', 'scalevar_3pt',
    #'UES','btagEffStat', 'btagWeight',
    'muoid', 'muotrig','muoiso',                    
]

sys_name_updown = {
    'JES' : ["jesTotaldown","jesTotalup"], 
    'JER' : ["jerdown","jerup"], 
    'pileup_weight' : ["pudown","puup"], 
    'jet_trigger' : ["stat_dn","stat_up"], 
    'L1Prefiring' : ["L1PreFiringup","L1PreFiringdown"],
    'W_d2kappa_EW' : ["W_d2kappa_EW_down","W_d2kappa_EW_up"],
    'W_d3kappa_EW' : ["W_d3kappa_EW_down","W_d3kappa_EW_up"],
    'Z_d2kappa_EW' : ["Z_d2kappa_EW_down","Z_d2kappa_EW_up"],
    'Z_d3kappa_EW' : ["Z_d3kappa_EW_down","Z_d3kappa_EW_up"],
    'd1kappa_EW' : ["d1kappa_EW_down", "d1kappa_EW_up"],
    'd1K_NLO' : ["d1K_NLO_down","d1K_NLO_up"],  
    'd2K_NLO' : ["d2K_NLO_down","d2K_NLO_up"],  
    'd3K_NLO' : ["d3K_NLO_down","d3K_NLO_up"],  
    'muoid' : ["muoiddown","muoidup"],
    'muotrig' : ["muotrigdown","muotrigup"],
    'muoiso' : ["muoisodown","muoisoup"],
    
}



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
    print("hist_str",hist_str)
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
    output_file = ROOT.TFile(args.root_path+f"/TEMPLATES{'_blind' if args.is_blinded else ''}.root", "RECREATE")
    print("Making SR templates from path ",args.root_path)
    for isamp,isamplist in sample_maps.items():
        for tagger in ["pnmd2prong_0p05","pnmd2prong_0p01"]:
            for region in ["pass","fail","pass_lowbvl","pass_highbvl",]:
                for iptbin in range(0,5):
                    make_templates(args.root_path,region,isamp,iptbin,tagger,syst=None,muon=False,nowarn=False,year="2017")
                    if "JetHT" in isamp: continue
                    for syst in sys_names:
                        if "muo" in syst: continue
                        #print(isamp in ["zqq","dy"])
                        if syst in ['W_d2kappa_EW', 'W_d3kappa_EW'] and not isamp in ["wqq","wlnu"]: continue
                        if syst in ['Z_d2kappa_EW', 'Z_d3kappa_EW'] and not isamp in ["zqq","dy"]: continue
                        if syst in ['d1kappa_EW','d1K_NLO','d2K_NLO','d3K_NLO'] and isamp not in ["wqq","wlnu","zqq","dy",]: continue 
                        make_templates(args.root_path,region,isamp,iptbin,tagger,syst=sys_name_updown[syst][0],muon=False,nowarn=False,year="2017")
                        make_templates(args.root_path,region,isamp,iptbin,tagger,syst=sys_name_updown[syst][1],muon=False,nowarn=False,year="2017")
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

