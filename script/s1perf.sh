rm -rf ~/ccfs3tmp
LOGFILE="log$(date -j +'%Y%m%d_%H%M%S').txt"
echo Logging to $LOGFILE
noisepy cross_correlate --raw_data_path ~/s3tmp/scedc/2022/002 --ccf_path ~/ccfs3tmp --freq_norm rma --stations "SBC,RIO,DEV,HEC,RLR,SVD,RPV,BAK" --xml_path ~/s3tmp/FDSNstationXML/ 2>&1 | tee $LOGFILE
