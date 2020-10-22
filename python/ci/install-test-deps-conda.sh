#!/bin/bash

echo
echo "[conda build]"
conda install -q conda-build conda-verify --yes


# create a copy of the environment file, replacing
# with the python version we specify.
sed -E "s/- python[^[:alpha:]]+$/- python=$PYTHON/" ./environment.yml > /tmp/environment_${PYTHON}.yml

conda env create -f /tmp/environment_${PYTHON}.yml

conda activate omnisci-connector-dev

conda list
echo
exit 0
