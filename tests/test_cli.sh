RUNNER_TEMP=~/test_temp
rm -rf $RUNNER_TEMP
noisepy download --start 2019_02_01_00_00_00 --end 2019_02_01_01_00_00 --stations ARV,BAK --inc_hours 1 --raw_data_path $RUNNER_TEMP/RAW_DATA
noisepy cross_correlate --raw_data_path $RUNNER_TEMP/RAW_DATA --ccf_path $RUNNER_TEMP/CCF --freq_norm rma
noisepy stack --raw_data_path $RUNNER_TEMP/RAW_DATA --ccf_path $RUNNER_TEMP/CCF --stack_path $RUNNER_TEMP/STACK --method linear
rm -rf $RUNNER_TEMP
noisepy all --start 2019_02_01_00_00_00 --end 2019_02_01_01_00_00 --stations ARV,BAK --inc_hours 1 --raw_data_path $RUNNER_TEMP/RAW_DATA \
       --ccf_path $RUNNER_TEMP/CCF --stack_path $RUNNER_TEMP/STACK --method linear --freq_norm rma
