#!/bin/zsh

FORMAT=$1
LOG_LEVEL=info

if [[ "$FORMAT" != "zarr" && "$FORMAT" != "asdf" && "$FORMAT" != "numpy" ]]; then
       echo "Missing or incorrect FORMAT argument. Needs to be zarr, numpy or asdf, not ${FORMAT}}"
       exit 1
fi
echo "FORMAT is _${FORMAT}_"
RUNNER_TEMP=~/test_temp_${FORMAT}

# RUNNER_TEMP=s3://carlosgjs-noisepy/test_new
# aws s3 rm --recursive $RUNNER_TEMP
rm -rf $RUNNER_TEMP
CCF=$RUNNER_TEMP/CCF
STACK=$RUNNER_TEMP/STACK
RAW=~/s3tmp/scedc/
XML=~/s3tmp/FDSNstationXML

mkdir $RUNNER_TEMP
LOGFILE="$HOME/logs/log_${FORMAT}_$(date -j +'%Y%m%d_%H%M%S').txt"
STATIONS=ARV,BAK
START=2022-02-02
END=2022-02-04
CHANNELS=BHE,BHN,BHZ
INC_HOURS=1
# Uncomment for a bigger test
# STATIONS=ADO,ALP,ARV,AVM,BAK,BAR,BBR,BBS,BC3,BCW
# START=2019-02-01T00:00:00
# END=2019-02-05T00:00:00
# INC_HOURS=24

set -e
rm -rf $CCF
noisepy cross_correlate  \
--raw_data_path=$RAW \
--xml_path=$XML \
--stations=$STATIONS \
--channels=$CHANNELS \
--start=$START \
--end=$END \
--ccf_path=$CCF \
--loglevel=${LOG_LEVEL} \
--format=${FORMAT} \
--logfile=$LOGFILE \
--loglevel=${LOG_LEVEL}

rm -rf $STACK
noisepy stack --ccf_path $CCF \
--stack_path=$STACK \
--stack_method=all \
--format=${FORMAT} \
--logfile=$LOGFILE \
--loglevel=${LOG_LEVEL}

du -ch $RUNNER_TEMP
tree $RUNNER_TEMP
