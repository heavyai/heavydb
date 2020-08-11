set -xe
cd build/Embedded
$PYTHON setup.py build_ext -g -f install
