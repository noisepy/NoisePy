#!/bin/zsh

RUNNER_TEMP=~/test_temp
LOG_LEVEL=info
FORMAT=$1
if [[ "$FORMAT" -ne "zarr" && "$FORMAT" -ne "asdf" ]]; then
       echo "Missing FORMAT argument (zarr or asdf)"
       exit 1
fi
echo "FORMAT is $FORMAT"
LOGFILE="$HOME/logs/log_${FORMAT}_$(date -j +'%Y%m%d_%H%M%S').txt"
STATIONS=ARV,BAK
START=2019-02-01T00:00:00
END=2019-02-01T01:00:00
INC_HOURS=1

# Uncomment for a bigger test
# STATIONS=ADO,ALP,ARV,AVM,BAK,BAR,BBR,BBS,BC3,BCW
# START=2019-02-01T00:00:00
# END=2019-02-05T00:00:00
# INC_HOURS=24

function sum_logs {
    SUM=$(cat $LOGFILE | grep "$1" |cut -d' ' -f 6,6 | paste -sd+ - | bc)
    echo "SUM for $FORMAT/$1 is $SUM" | tee -a $LOGFILE
}
rm -rf $RUNNER_TEMP
set -e
noisepy download --start_date "${START}" --end_date "${END}" --stations "${STATIONS}" --inc_hours "${INC_HOURS}" --raw_data_path $RUNNER_TEMP/RAW_DATA --log ${LOG_LEVEL} 2>&1 | tee -a $LOGFILE
rm -rf $RUNNER_TEMP/CCF
noisepy cross_correlate --raw_data_path $RUNNER_TEMP/RAW_DATA --ccf_path $RUNNER_TEMP/CCF --freq_norm rma --log ${LOG_LEVEL} --format ${FORMAT} 2>&1 | tee -a $LOGFILE
rm -rf $RUNNER_TEMP/STACK
mpiexec -n 3 noisepy stack --mpi --raw_data_path $RUNNER_TEMP/RAW_DATA --ccf_path $RUNNER_TEMP/CCF --stack_path $RUNNER_TEMP/STACK --stack_method all --log ${LOG_LEVEL} --format ${FORMAT} 2>&1 | tee -a $LOGFILE
rm -rf $RUNNER_TEMP/STACK
noisepy stack --raw_data_path $RUNNER_TEMP/RAW_DATA --ccf_path $RUNNER_TEMP/CCF --stack_path $RUNNER_TEMP/STACK --stack_method all --log ${LOG_LEVEL} --format ${FORMAT} 2>&1 | tee -a $LOGFILE
rm -rf $RUNNER_TEMP
noisepy all --start_date 2019-02-01T00:00:00 --end_date 2019-02-01T01:00:00 --stations ARV,BAK --inc_hours 1 --raw_data_path $RUNNER_TEMP/RAW_DATA \
       --ccf_path $RUNNER_TEMP/CCF --stack_path $RUNNER_TEMP/STACK --stack_method all --freq_norm rma --log ${LOG_LEVEL} --format ${FORMAT} 2>&1 | tee -a $LOGFILE

sum_logs "append"
sum_logs "loading CCF"
sum_logs "Append to store"
