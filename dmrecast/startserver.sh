#!/bin/bash
# Check proxy status and remake if needed
export X509_USER_PROXY=/uscms/home/dryu/x509up_u13884
echo "Using X509_USER_PROXY="
echo $X509_USER_PROXY
if ! [ -f $X509_USER_PROXY ]; then
	echo "Remaking proxy"
	voms-proxy-init -voms cms -valid 72:00 -out $X509_USER_PROXY
else
	echo "Found proxy with lifetime:"
	voms-proxy-info -timeleft -file $X509_USER_PROXY
fi
echo "Done with proxy"

#ls -lrth
echo "Start venv"
source venv/bin/activate

echo "Killing existing jupyter-lab processes"
pkill -f jupyter-lab

echo "Starting jupyter"
jupyter-lab --port $1 --no-browser
