#!/bin/bash
STEP=$1
ALGO=$2
NJOB=$3
WDIR=$4
YEAR=$5
if  [[ "$ALGO" -eq 0 ]]; then
    ALGO=saturated
elif [[ "$ALGO" -eq 1 ]]; then
    ALGO=KS
fi
echo "Using algo:" $ALGO

if [[ "$STEP" -eq 0 ]]; then
    nohup combine -M FitDiagnostics -d $WDIR/inclusive_workspace.root --cminDefaultMinimizerStrategy 0 --out $WDIR --saveShapes --saveWithUncertainties --skipSBFit & # --robustFit=1 --robustHesse=1 --redefineSignalPOIs "tf${YEAR}_MC_0templ_deco0" --freezeParameters r --setParameters r=0  -n "" --out $WDIR --saveShapes --saveWithUncertainties
elif [[ "$STEP" -eq 1 ]]; then
    nohup combineTool.py -M GoodnessOfFit -d $WDIR/inclusive_workspace.root  --algo $ALGO  --fixedSignalStrength=0 --freezeParameters r --setParameters r=0 -n "DataGoF$ALGO" --there & #--out $WDIR #--redefineSignalPOIs "tf${YEAR}_MC_0templ_deco0" --freezeParameters r --setParameters r=0 -n "DataGoF$ALGO" --out $WDIR
elif [[ "$STEP" -eq 2 ]]; then
    combineTool.py -M GoodnessOfFit -d $WDIR/model_combined.root --algo $ALGO  -t 5 --toysFrequentist --fixedSignalStrength=0 --freezeParameters r --setParameters r=0 -n GoFs$ALGO --job-mode condor --sub-opts='+JobFlavour = "workday"' --task-name zprime$ALGO -s 1:"$NJOB":1 --there #--redefineSignalPOIs "tf${YEAR}_MC_0templ_deco0" --freezeParameters r --setParameters r=0 -n GoFs$ALGO --job-mode condor --sub-opts='+JobFlavour = "workday"' --task-name zprime$ALGO -s 1:"$NJOB":1 --there  
elif [[ "$STEP" -eq 3 ]]; then 
    hadd -f $WDIR/allgofs$ALGO.root $WDIR/higgsCombineGoFs$ALGO.GoodnessOfFit.mH120.*.root
elif [[ "$STEP" -eq 4 ]]; then 
    python3 plot_single_gof.py $WDIR/higgsCombineDataGoF$ALGO.GoodnessOfFit.mH120.root $WDIR/allgofs$ALGO.root --algo $ALGO --year $YEAR --opath $WDIR
fi
