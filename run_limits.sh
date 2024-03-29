#!/bin/bash
print_help() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  --ipath <path>       Specify the input path."
  echo "  --asimov             Enable Asimov mode."
  echo "  --limits             Enable limits calculation."
  echo "  --fitdiag            Enable fit diagnostics."
  echo "  --sigmass <value>    Specify the signal mass."
  echo "  --help               Display this help and exit."
  echo ""
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --ipath)
      IPATH="$2"
      shift # past argument
      shift # past value
      ;;
    --asimov)
      ASIMOV=true
      shift
      ;;
    --toysFrequentist)
      TOYSFREQUENTIST=true
      shift
      ;;
    --limits)
      LIMITS=true
      shift
      ;;
    --fitdiagnostics)
      FITDIAG=true
      shift
      ;;
    --sigmass)
      SIGMASS="$2"
      shift
      shift
      ;;
    --help)
      print_help
      return 0 2>/dev/null || exit 0
      ;;
    *)
      # Unknown option
      echo "Unknown option: $1"
      print_help
      return 1 2>/dev/null || exit 1
      ;;
  esac
done


cd ${IPATH}/m${SIGMASS}/m${SIGMASS}_model || exit

pwd 

if [[ ! -f "./model_combined.txt" ]]; then
   bash build.sh
fi 

if [[ $LIMITS == true ]]; then
   cmd="combine -M AsymptoticLimits -d model_combined.root  --cminDefaultMinimizerType Minuit2 --cminDefaultMinimizerStrategy 0 --cminDefaultMinimizerTolerance 0.1 --cminFallbackAlgo Minuit2,0:0.2 --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 --expectSignal 0.0001"
fi


if [[ $FITDIAG == true ]]; then
   cmd="combine -M AsymptoticLimits model_combined.txt  --cminDefaultMinimizerType Minuit2 --cminDefaultMinimizerStrategy 0 --cminDefaultMinimizerTolerance 0.1 --cminFallbackAlgo Minuit2,0:0.2 --X-rtd REMOVE_CONSTANT_ZERO_POINT=1"
fi

if [[ $ASIMOV == true ]]; then
   cmd+=" -t -1"
fi
if [[ $TOYSFREQUENTIST == true ]]; then
   cmd+=" --toysFrequentist -t 1"
fi

echo evaluating $cmd

if [[ -n $cmd ]]; then
  eval $cmd
fi

