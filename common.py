

lumi_dict = {
    "2017": 41100,
    "fullrun2" : 150000,
}

lumi_dict_unc = {
    "2016": 1.01,
    "2017": 1.02,
    "2018": 1.015,
}
lumi_correlated_dict_unc = {
    "2016": 1.006,
    "2017": 1.009,
    "2018": 1.02,
}
lumi_1718_dict_unc = {
    "2017": 1.006,
    "2018": 1.002,
}

#sys_name_updown = {
#    "JES": ["jesTotaldown", "jesTotalup"],
#    "JER": ["jerdown", "jerup"],
#    "pileup_weight": ["pudown", "puup"],
#    "jet_trigger": ["stat_dn", "stat_up"],
#    "L1Prefiring": ["L1PreFiringdown", "L1PreFiringup"],
#    "d1kappa_EW" : ["d1kappa_EW_down", "d1kappa_EW_up"],
#    "d1K_NLO" : ["d1K_NLO_down", "d1K_NLO_up"],
#    "d2K_NLO" : ["d2K_NLO_down", "d2K_NLO_up"],
#    "d3K_NLO" : ["d3K_NLO_down", "d3K_NLO_up"],
#    "Z_d2kappa_EW" : ["Z_d2kappa_EW_down","Z_d2kappa_EW_up"],
#    "Z_d3kappa_EW" : ["Z_d3kappa_EW_down","Z_d3kappa_EW_up"],
#}


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
    'jms', 'jmr',
    'W_d2kappa_EW', 'W_d3kappa_EW',
    'Z_d2kappa_EW', 'Z_d3kappa_EW',
    'd1kappa_EW', 'd1K_NLO', 'd2K_NLO', 'd3K_NLO',
    #'scalevar_7pt', 'scalevar_3pt',
    'UES','btagstat1', 'btagstat2','btagSF',
    'muoid', 'muotrig','muoiso',                    
]

sys_name_updown = {
    'JES' : ["jesTotaldown","jesTotalup"], 
    'JER' : ["jerdown","jerup"], 
    'pileup_weight' : ["pudown","puup"], 
    'jet_trigger' : ["stat_dn","stat_up"],
    'jms' : ["jmsdown","jmsup"],
    'jmr' : ["jmrdown","jmrup"], 
    'L1Prefiring' : ["L1PreFiringup","L1PreFiringdown"],
    'W_d2kappa_EW' : ["W_d2kappa_EW_down","W_d2kappa_EW_up"],
    'W_d3kappa_EW' : ["W_d3kappa_EW_down","W_d3kappa_EW_up"],
    'Z_d2kappa_EW' : ["Z_d2kappa_EW_down","Z_d2kappa_EW_up"],
    'Z_d3kappa_EW' : ["Z_d3kappa_EW_down","Z_d3kappa_EW_up"],
    'd1kappa_EW' : ["d1kappa_EW_down", "d1kappa_EW_up"],
    'd1K_NLO' : ["d1K_NLO_down","d1K_NLO_up"],  
    'd2K_NLO' : ["d2K_NLO_down","d2K_NLO_up"],  
    'd3K_NLO' : ["d3K_NLO_down","d3K_NLO_up"],  
    'mu_idweight' : ["muoiddown","muoidup"],
    'mu_trigger' : ["muotrigdown","muotrigup"],
    'mu_isoweight' : ["muoisodown","muoisoup"],
    'UES' : ["unclustEndown","unclustEnup"], 
    'btagSF_hfstats1' : [r"btagSF_hfstats1_year_ULdown",r"btagSF_hfstats1_year_ULup"],
    'btagSF_hfstats2' : [r"btagSF_hfstats2_year_ULdown",r"btagSF_hfstats2_year_ULup"],
    'btagSF'  : ["btagSF_hfdown","btagSF_hfup"],
}
