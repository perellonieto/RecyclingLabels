#!/bin/bash
#PBS -N log_testRecyclingLabels
# request resources:
#PBS -l nodes=1:ppn=16
#PBS -l walltime=0:20:00

## Options to run job arrays
#PBS -t 0-99

# on compute node, change directory to 'submission directory':
cd $PBS_O_WORKDIR
# record some potentially useful details about the job:
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo "PBS job ID is ${PBS_JOBID}"
echo "The Array ID is: ${PBS_ARRAYID}"
echo "This jobs runs on the following machines:"
echo `cat $PBS_NODEFILE | uniq`
# count the number of processors available:
numprocs=`wc $PBS_NODEFILE | awk '{print $1}'`

# Activate the virtual environment
#. venv/bin/activate
source activate recyclingenv

### TODO Should I use the following declarations to facilitate the calls?
declare -A parameters_fs_ir=([method]='fully_supervised' [dataset]='iris'       [epochs]=5000)
declare -A parameters_fs_ir=([method]='EM'               [dataset]='iris'       [epochs]=3000)
declare -A parameters_fs_bl=([method]='fully_supervised' [dataset]='blobs'      [epochs]=8000)
declare -A parameters_fw_bl=([method]='fully_weak'       [dataset]='blobs'      [epochs]=3000)
declare -A parameters_pw_bl=([method]='partially_weak'   [dataset]='blobs'      [epochs]=3000)
declare -A parameters_os_bl=([method]='OSL'              [dataset]='blobs'      [epochs]=3000)
declare -A parameters_mp_bl=([method]='Mproper'          [dataset]='blobs'      [epochs]=3000)
declare -A parameters_em_bl=([method]='EM'               [dataset]='blobs'      [epochs]=3000)
declare -A parameters_fs_bw=([method]='fully_supervised' [dataset]='blobs_webs' [epochs]=300)
declare -A parameters_fw_bw=([method]='fully_weak'       [dataset]='blobs_webs' [epochs]=30)
declare -A parameters_pw_bw=([method]='partially_weak'   [dataset]='blobs_webs' [epochs]=30)
declare -A parameters_os_bw=([method]='OSL'              [dataset]='blobs_webs' [epochs]=30)
declare -A parameters_mp_bw=([method]='Mproper'          [dataset]='blobs_webs' [epochs]=30)
declare -A parameters_em_bw=([method]='EM'               [dataset]='blobs_webs' [epochs]=30)
declare -A parameters_fs_we=([method]='fully_supervised' [dataset]='webs'       [epochs]=300)
declare -A parameters_fw_we=([method]='fully_weak'       [dataset]='webs'       [epochs]=30)
declare -A parameters_pw_we=([method]='partially_weak'   [dataset]='webs'       [epochs]=30)
declare -A parameters_os_we=([method]='OSL'              [dataset]='webs'       [epochs]=30)
declare -A parameters_mp_we=([method]='Mproper'          [dataset]='webs'       [epochs]=30)
declare -A parameters_em_we=([method]='EM'               [dataset]='webs'       [epochs]=30)

declare -a parameters_all=(
    #parameters_fs_ir
    #parameters_fs_ir
    #parameters_fs_bl
    #parameters_fw_bl
    #parameters_pw_bl
    #parameters_os_bl
    #parameters_mp_bl
    #parameters_em_bl
    #parameters_fs_bw
    #parameters_fw_bw
    #parameters_pw_bw
    #parameters_os_bw
    #parameters_mp_bw
    #parameters_em_bw
    parameters_fs_we
    #parameters_fw_we
    #parameters_pw_we
    #parameters_os_we
    #parameters_mp_we
    #parameters_em_we
    )

PARAM_ID=0
method=${parameters_all[$PARAM_ID]}[method]
method=${!method}
dataset=${parameters_all[$PARAM_ID]}[dataset]
dataset=${!dataset}
epochs=${parameters_all[$PARAM_ID]}[epochs]
epochs=${!epochs}
echo ${method} ${dataset} ${epochs}
architecture='lr' # lr, mlp100m, mlp100sdm, mlp100ds100dm

path_results='results_2018_01_16_sgd_momentum_09'
verbose=1
#data=${method_names[$PBS_ARRAYID]}
loss="mse" # mse, bs
k_folds=5
iterations=10
processes=8
seed=0
lr=1.0          # sgd rmsprop adagrad adadelta adam adamax nadam
L1_LIST=($(seq 0.0 0.01 0.99))
L2_LIST=($(seq 0.0 0.01 0.99))
l1=0.0
l2=${L2_LIST[$PBS_ARRAYID]}
momentum=0.9    # sgd
decay=0.0       # sgd rmsprop adagrad adadelta adam adamax
rho=0.0         #     rmsprop         adadelta
epsilon=None    #     rmsprop adagrad adadelta adam adamax nadam
nesterov=False  # sgd
batch_size=100
optimizer='sgd' # sgd, rmsprop, adagrad, adadelta
#path_model="results_2017_10_27_no_pretraining/webs_fully_supervised_${architecture}"

mkdir -p ${path_results}
# FIXME this code is using more processes than specified in -p
#       I need to find the reason. Meanwhile, it is better to request for 16
#       processors in PBS not to affect other people
time python run_baseline.py -d ${dataset} -a ${architecture} -m ${method} \
                            -i ${iterations} -c ${epochs} \
                            -k ${k_folds} -s ${seed} -r ${path_results} \
                            -l ${loss} -v ${verbose} -p ${processes} \
                            --lr ${lr} --l1 ${l1} --l2 ${l2} \
                            --optimizer ${optimizer} --momentum ${momentum} \
                            --decay ${decay} --nesterov ${nesterov} \
                            --rho ${rho} --epsilon ${epsilon} \
                            --batch-size ${batch_size} \
                            --stdout --stderr
                            # -t ${path_model} \
                            # -M M.csv \
