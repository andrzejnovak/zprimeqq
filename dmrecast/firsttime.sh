#!/bin/bash
NAME=venv
LCG=/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt
source $LCG/setup.sh

# following https://aarongorka.com/blog/portable-virtualenv/, an alternative is https://github.com/pantsbuild/pex
python -m venv --copies $NAME
source $NAME/bin/activate
LOCALPATH=$NAME$(python -c 'import sys; print(f"/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")')
export PYTHONPATH=${LOCALPATH}:$PYTHONPATH
python -m pip install setuptools pip wheel --upgrade
python -m pip install coffea
sed -i '1s/#!.*python$/#!\/usr\/bin\/env python/' $NAME/bin/*
sed -i '38s/.*/VIRTUAL_ENV="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}" )")" \&\& pwd)"/' $NAME/bin/activate
sed -i "2a source ${LCG}/setup.sh" $NAME/bin/activate
sed -i "3a export PYTHONPATH=${LOCALPATH}:\$PYTHONPATH" $NAME/bin/activate
tar -zcf ${NAME}.tar.gz ${NAME}

