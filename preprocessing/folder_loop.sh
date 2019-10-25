#!/bin/bash
# This source code was developed under the DARPA Radio Frequency Machine 
# Learning Systems (RFMLS) program contract N00164-18-R-WQ80. All the code 
# released here is unclassified and the Government has unlimited rights 
# to the code.



export VOLK_GENERIC=1
export GR_DONT_LOAD_PREFS=1
export srcdir=.
export GR_CONF_CONTROLPORT_ON=False
export PATH=.:$PATH
export LD_LIBRARY_PATH=/gr-ieee802-11/build/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/gr-ieee802-11/build/swig:$PYTHONPATH

for d in $(find $1* -maxdepth 1 -mindepth 1 -type d);
do
    echo $d;
    python src/preprocessing/generate_bin_file.py --root $d
done
#for d in $(find /home/bruno/RFMLS/docker/data/test/1Cv2/wifi_eq/* -maxdepth 0 -mindepth 0 -type d);
#do
#    echo $d;
#    python ./generate_bin_file.py --root $d
#done
