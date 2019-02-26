
export PATH=/local/gmoertl/DeepLepton/DeepJet_GPU/miniconda3/bin:$PATH
export DEEPJETCORE=../DeepJetCore

THISDIR=`pwd`
cd $DEEPJETCORE
source ./gpu_env.sh
cd $THISDIR
export PYTHONPATH=`pwd`/modules:$PYTHONPATH
export DEEPJET=`pwd`
