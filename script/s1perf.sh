rm -rf ~/ccfs3tmp
mkdir -p $HOME/logs
LOGFILE="$HOME/logs/log$(date -j +'%Y%m%d_%H%M%S').txt"
echo Logging to $LOGFILE
# STATIONS="SBC,RIO,DEV,HEC,RLR,SVD,RPV,BAK,CBC,CCA,CCC,CDD,HLL,HOL,AGM,AGO"
STATIONS="SBC,RIO,DEV,HEC,RLR,SVD,RPV,BAK"
noisepy cross_correlate --raw_data_path ~/s3tmp/scedc/2022/002 --ccf_path ~/ccfs3tmp --freq_norm rma --stations "$STATIONS" --xml_path ~/s3tmp/FDSNstationXML/ 2>&1 | tee $LOGFILE
