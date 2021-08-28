
# CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# AKG_DIR="${AKG_DIR:-${CUR_DIR}/..}"

AKG_DIR=$HOME/repos/akg

cd ${AKG_DIR}
bash build.sh -e gpu -j 32

cd tests/operators/gpu/rundir
rm -rf *

if [ $# -eq 1 ]; then
  python ../test_all.py $1
else
  python ../test_all.py add
fi
