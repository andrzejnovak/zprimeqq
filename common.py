

lumi_dict = {
    "2016APV" : 19520.,
    "2016" : 16810.,
    "2016postVFP" : 16810.,
    "2016preVFP" : 19520.,
    "2017" : 41480.,
    "2018" : 59720.,
  
    "fullrun2" : 150000,
}

lumi_dict_unc = {
    "2016APV": 1.01,
    "2016": 1.01,
    "2017": 1.02,
    "2018": 1.015,
}
lumi_correlated_dict_unc = {
    "2016APV": 1.006,
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
    "vv" : ["WZ_TuneCP5_13TeV-pythia8", "ZZ_TuneCP5_13TeV-pythia8","WW_TuneCP5_13TeV-pythia8"],
    "QCD" : ["QCD_HT500to700","QCD_HT700to1000","QCD_HT1000to1500","QCD_HT1500to2000","QCD_HT2000toInf"],
    "wqq" : ["WJetsToQQ_HT-600to800","WJetsToQQ_HT-800toInf"],
    "zqq" : ["ZJetsToQQ_HT-600to800","ZJetsToQQ_HT-800toInf"],
    "zbb" : ["ZJetsToBB_HT-600to800","ZJetsToBB_HT-800toInf"],
    "tt"  : ["TTTo2L2Nu","TTToHadronic","TTToSemiLeptonic"],
    "dy"  : ["DYJetsToLL_HT-400To600","DYJetsToLL_HT-600To800","DYJetsToLL_HT-800To1200","DYJetsToLL_HT-1200To2500"],
    "st"  : ["ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8","ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8","ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8","ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8"],
    "hbb" : ["GluGluHToBB","VBFHToBB","ttHToBB","ggZHToBB","ZHToBB","WplusH_HToBB","WminusH_HToBB",],
    "wlnu" : ["WJetsToLNu_HT400to600","WJetsToLNu_HT600to800","WJetsToLNu_HT800to1200","WJetsToLNu_HT1200to2500","WJetsToLNu_HT2500toInf"],
    "JetHT_2016preVFP" : ["JetHT_Run2016B_ver2_HIPM","JetHT_Run2016C_HIPM","JetHT_Run2016D_HIPM","JetHT_Run2016E_HIPM","JetHT_Run2016F_HIPM",],
    "JetHT_2016postVFP" : ["JetHT_Run2016F","JetHT_Run2016G","JetHT_Run2016H"],
    "JetHT_2017" : ["JetHT_Run2017B","JetHT_Run2017C","JetHT_Run2017D","JetHT_Run2017E","JetHT_Run2017F"],
    "JetHT_2018" : ["JetHT_Run2018A","JetHT_Run2018B","JetHT_Run2018C","JetHT_Run2018D"],
    "zpqq50" : ["VectorZPrimeToQQ_M50"],
    "zpqq75" : ["VectorZPrimeToQQ_M75"],
    "zpqq100" : ["VectorZPrimeToQQ_M100"],
    "zpqq125" : ["VectorZPrimeToQQ_M125"],
    "zpqq150" : ["VectorZPrimeToQQ_M150"],
    "zpqq200" : ["VectorZPrimeToQQ_M200"],
    "zpqq250" : ["VectorZPrimeToQQ_M250"],
    "zpqq300" : ["VectorZPrimeToQQ_M300"],
    "zpbb50" : ["VectorZPrimeToBB_M50"],
    "zpbb75" : ["VectorZPrimeToBB_M75"],
    "zpbb100" : ["VectorZPrimeToBB_M100"],
    "zpbb125" : ["VectorZPrimeToBB_M125"],
    "zpbb150" : ["VectorZPrimeToBB_M150"],
    "zpbb200" : ["VectorZPrimeToBB_M200"],
    "zpbb250" : ["VectorZPrimeToBB_M250"],
    "zpbb300" : ["VectorZPrimeToBB_M300"],

    "SingleMuon_2016postVFP" : ["SingleMuon_Run2016F","SingleMuon_Run2016G","SingleMuon_Run2016H",],
    "SingleMuon_2017" : ["SingleMuon_Run2017B","SingleMuon_Run2017C","SingleMuon_Run2017D","SingleMuon_Run2017E","SingleMuon_Run2017F"],
    "SingleMuon_2018" : ["SingleMuon_Run2018A","SingleMuon_Run2018B","SingleMuon_Run2018C","SingleMuon_Run2018D"]
}

sample_maps_mu = {

    "QCD" : ["QCD_HT500to700","QCD_HT700to1000","QCD_HT1000to1500","QCD_HT1500to2000","QCD_HT2000toInf"],
    "wqq" : ["WJetsToQQ_HT-600to800","WJetsToQQ_HT-800toInf"],
    "zqq" : ["ZJetsToQQ_HT-600to800","ZJetsToQQ_HT-800toInf"],
    "tt"  : ["TTTo2L2Nu","TTToHadronic","TTToSemiLeptonic"],
    "dy"  : ["DYJetsToLL_HT-400To600","DYJetsToLL_HT-600To800","DYJetsToLL_HT-800To1200","DYJetsToLL_HT-1200To2500"],
    "st"  : ["ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8","ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8","ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8","ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8"],
    "wlnu" : ["WJetsToLNu_HT400to600","WJetsToLNu_HT600to800","WJetsToLNu_HT800to1200","WJetsToLNu_HT1200to2500","WJetsToLNu_HT2500toInf"],
    "SingleMuon_2016postVFP" : ["SingleMuon_Run2016F","SingleMuon_Run2016G","SingleMuon_Run2016H",],
    "SingleMuon_2017" : ["SingleMuon_Run2017B","SingleMuon_Run2017C","SingleMuon_Run2017D","SingleMuon_Run2017E","SingleMuon_Run2017F"],
    "SingleMuon_2018" : ["SingleMuon_Run2018A","SingleMuon_Run2018B","SingleMuon_Run2018C","SingleMuon_Run2018D"]
}
sys_names = [
    'JES', 'JER', 'jet_trigger','pileup_weight','L1Prefiring',
    'jms', 'jmr',
    'W_d2kappa_EW', 'W_d3kappa_EW',
    'Z_d2kappa_EW', 'Z_d3kappa_EW',
    'd1kappa_EW', 'd1K_NLO', 'd2K_NLO', 'd3K_NLO',
    #'scalevar_7pt', 'scalevar_3pt',
    'UES','btagSF_heavy_correlated','btagSF_light_correlated',
    'btagSF_heavy_year', 'btagSF_light_year',
    'muoid', 'muotrig','muoiso',                   
    'HEMissue', 
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
    'muoid' : ["muoiddown","muoidup"],
    'muotrig' : ["muotrigdown","muotrigup"],
    'muoiso' : ["muoisodown","muoisoup"],
    'UES' : ["unclustEndown","unclustEnup"], 
    'btagSF_heavy_correlated' : [r"btagSF_heavydown","btagSF_heavyup"],
    'btagSF_light_correlated' : [r"btagSF_lightdown","btagSF_lightup"],
    'btagSF_heavy_year'  : [r"btagSF_heavy_M_year_ULdown",r"btagSF_heavy_M_year_ULup"],
    'btagSF_light_year'  : [r"btagSF_light_M_year_ULdown",r"btagSF_light_M_year_ULup"],
    'HEMissue' : ["jesHEMIssuedown","jesHEMIssueup"],
}
