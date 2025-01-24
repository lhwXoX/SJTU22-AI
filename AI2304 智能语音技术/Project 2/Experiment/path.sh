# export KALDI_ROOT=`pwd`/../../..

# use cpu-compiled kaldi, for regular cases
#export KALDI_ROOT="/lustre/home/acct-stu/stu1718/kaldi_cpu"

# use gpu-compiled kaldi, for DNN-HMM
export KALDI_ROOT="/lustre/home/acct-stu/stu1718/kaldi"

# necessary environment variables
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=/lustre/home/acct-stu/stu1853/tools/kaldi/tools/kaldi_lm:$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
export PATH=/lustre/home/acct-stu/stu1853/tools/sox/bin:$PATH

[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
export PYTHONUNBUFFERED=1