RUNNER_TEMP=~/test_temp
LOG_LEVEL=debug
rm -rf $RUNNER_TEMP
noisepy download --start 2019_02_01_00_00_00 --end 2019_02_01_01_00_00 --stations ARV,BAK --inc_hours 1 --raw_data_path $RUNNER_TEMP/RAW_DATA --log ${LOG_LEVEL}
noisepy cross_correlate --raw_data_path $RUNNER_TEMP/RAW_DATA --ccf_path $RUNNER_TEMP/CCF --freq_norm rma --log ${LOG_LEVEL}
rm -rf $RUNNER_TEMP/STACK
noisepy stack --raw_data_path $RUNNER_TEMP/RAW_DATA --ccf_path $RUNNER_TEMP/CCF --stack_path $RUNNER_TEMP/STACK --method linear --log ${LOG_LEVEL}
rm -rf $RUNNER_TEMP
noisepy all --start 2019_02_01_00_00_00 --end 2019_02_01_01_00_00 --stations ARV,BAK --inc_hours 1 --raw_data_path $RUNNER_TEMP/RAW_DATA \
       --ccf_path $RUNNER_TEMP/CCF --stack_path $RUNNER_TEMP/STACK --method linear --freq_norm rma --log ${LOG_LEVEL}
