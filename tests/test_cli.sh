RUNNER_TEMP=~/test_temp
LOG_LEVEL=debug
rm -rf $RUNNER_TEMP
set -e
noisepy download --start_date 2019-02-01T00:00:00 --end_date 2019-02-01T01:00:00 --stations ARV,BAK --inc_hours 1 --raw_data_path $RUNNER_TEMP/RAW_DATA --log ${LOG_LEVEL}
# rm -rf $RUNNER_TEMP/CCF
noisepy cross_correlate --raw_data_path $RUNNER_TEMP/RAW_DATA --ccf_path $RUNNER_TEMP/CCF --freq_norm rma --log ${LOG_LEVEL}
# rm -rf $RUNNER_TEMP/STACK
mpiexec -n 3 noisepy stack --mpi --raw_data_path $RUNNER_TEMP/RAW_DATA --ccf_path $RUNNER_TEMP/CCF --stack_path $RUNNER_TEMP/STACK --stack_method all --log ${LOG_LEVEL}
rm -rf $RUNNER_TEMP/STACK
noisepy stack --raw_data_path $RUNNER_TEMP/RAW_DATA --ccf_path $RUNNER_TEMP/CCF --stack_path $RUNNER_TEMP/STACK --stack_method all --log ${LOG_LEVEL} 2>&1 > log.txt
rm -rf $RUNNER_TEMP
noisepy all --start_date 2019-02-01T00:00:00 --end_date 2019-02-01T01:00:00 --stations ARV,BAK --inc_hours 1 --raw_data_path $RUNNER_TEMP/RAW_DATA \
       --ccf_path $RUNNER_TEMP/CCF --stack_path $RUNNER_TEMP/STACK --stack_method all --freq_norm rma --log ${LOG_LEVEL}
