#!/bin/bash
if [ -z "$1" ]; then
	PORT=7778;
else
	PORT="$1";
fi
host=cmslpc325.fnal.gov
#ssh -L localhost:${PORT}:localhost:${PORT} dryu@cmslpc-el8.fnal.gov "uname -a; cd DAZSLE/boostedzprime; echo 'Setting up env...'; ./shellnb; echo 'Launching notebook...'; source nbserver.sh ${PORT};"
ssh -L localhost:${PORT}:localhost:${PORT} dryu@${host} "cd DAZSLE/VQQDMRecast; source venv/bin/activate; source startserver.sh ${PORT};"
