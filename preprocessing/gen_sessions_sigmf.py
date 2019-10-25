#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
This source code was developed under the DARPA Radio Frequency Machine 
Learning Systems (RFMLS) program contract N00164-18-R-WQ80. All the code 
released here is unclassified and the Government has unlimited rights 
to the code.

Created on Mon Aug  6 12:44:29 2018

@author: shamnaz
@refactor: mauro

This script reads the SigMF metadata files and generates device specific mat files
for each session in a folder structure

TODO: think about multi processing (if there's no device limit, it should be trivial)

"""

# Pull a signal from a file given filename, start sample, and sample count
import numpy as np
import os
import json
import scipy.io as spio
import pickle
import glob
import sys


def PullBinarySample(filename, startSample, sampCount):
    # account for no buffer possible because signal starts at beginning of record
    if startSample < 0:
        startSample = 0

    with open(filename, "rb") as f:
        # Seek to startSample
        f.seek(startSample * 4)
        # Read in as ints
        raw = np.fromfile(f, dtype='int16', count=2 * sampCount)
        # Convert interleaved ints into complex
        try:
            #print raw.shape, startSample
            array = raw.reshape([sampCount, 2])
        except:
            return None
        cmp = array[:, 0] + array[:, 1] * 1j
        return cmp
        


def GenerateSessionFiles(glob_sigmf_meta_path, output_path, signal_names=set(), dev_limit=0, sampling_rate=0, all_info=False, fileType='mat', seen=set()):
    # exaple glob_sigmf_path='/Darpa_rfml/*-meta'

    if output_path[-1] != '/':
        output_path += '/'

    if not os.path.exists(output_path):
        print "ERROR! GenerateSessionFiles(): Output Directory not found. Aborting..."
        sys.exit()

    devcnt = 0

    filelist = glob.glob(glob_sigmf_meta_path)
    for f in filelist:
        #print 'Processing', f

        fnMeta = f[:-4] + 'meta'  # Trim last 4 and add 'meta' to get meta filename
        fnData = f[:-4] + 'data'
        count = 0

        # Buffer before and after signal
        buffer = 0

        allSignals = json.load(open(fnMeta))

        # Determine if the recording is wifi
        is_wifi = allSignals['annotations'][0]['rfml:label'][0] == 'wifi'
        is_adsb = allSignals['annotations'][0]['rfml:label'][0] == 'ADS-B'
        # find time step based on sample rate
        sr = allSignals['global']['core:sample_rate']
        # if a specific sampling rate is specified, check if the file reports the same one
        if sampling_rate != 0 and sr != sampling_rate:
            return 0   # if sampling rate doesn't match, return immediately with 0 device count

        Fc = allSignals['capture'][0]['core:frequency']
        dt = 1 / sr
        hwmany = 1

        # this is used to stop the outer for-loop when device limit is reached
        if dev_limit != 0 and devcnt > dev_limit:
            break

        # for each signal
        for signal in allSignals['annotations']:
            if ('capture_details:signal_reference_number' in signal) and ('rfml:label' in signal) and ('core:sample_start' in signal) and ('core:sample_count' in signal):

                # Read parameters for signal pulling
                #print signal
                startSamps = signal['core:sample_start']
                countSamps = signal['core:sample_count']
                sig_ref_no = signal['capture_details:signal_reference_number'].encode('ascii')
                dev_type = signal['rfml:label'][0].encode('ascii')
                if is_wifi:
                    dev_vendor = signal['rfml:label'][1].encode('ascii')
                    dev_id = signal['rfml:label'][2].encode('ascii')
                elif is_adsb:
                    dev_vendor = ''
                    dev_id = signal['rfml:label'][1].encode('ascii')
                else:
                    dev_type = 'newtype'
                    dev_vendor = ''
                    dev_id = signal['rfml:label'][1].encode('ascii')
                

                # in case a signal/example name has been specified, perform the extraction only if ref. number is in the list of the signals to extract
                if (len(signal_names) != 0) and sig_ref_no not in signal_names:
                    continue    # otherwise, go to the next signal
                seen.add(sig_ref_no)
                complexSignal = PullBinarySample(fnData, startSamps - buffer, countSamps + (2 * buffer))
                if complexSignal is None:
                    print "Ignoring " + sig_ref_no
                    continue
                # Pull channel info if wifi or set if ADSB
                vendor_dir = out_dir = output_path + dev_type
                
                if is_wifi:
                    lowFreq = float(signal['core:freq_lower_edge'])
                    upFreq = float(signal['core:freq_upper_edge'])
                    vendor_dir += "/" + dev_vendor 
                    deviceName_dir = vendor_dir + "/" + dev_type + "_" + dev_vendor + "_" + dev_id
                   
                elif is_adsb:
                    lowFreq = 1.085e9
                    upFreq = 1.095e9
                    deviceName_dir = out_dir + "/" + dev_type + "_" + dev_id
                    
                else:
                    lowFreq = float(signal['core:freq_lower_edge'])
                    upFreq = float(signal['core:freq_upper_edge'])
                    deviceName_dir = out_dir + "/" + dev_type + "_" + dev_id
                    
                # Pull Binary signal
                # create signal type folder
                if not os.path.exists(out_dir):
                    try:
                        os.makedirs(out_dir)
                    except:
                        pass
                # create vendor folder
                if not os.path.exists(vendor_dir):
                    try:
                        os.makedirs(vendor_dir)
                    except:
                        pass
                # create device folder
                if not os.path.exists(deviceName_dir):
                    devcnt += 1
                    # if we reached the device limit, stop the for-loop on this meta file
                    if dev_limit != 0:
                        if devcnt > dev_limit:
                            break
                    try:
                        os.makedirs(deviceName_dir)
                    except:
                        pass

                if not os.path.isfile(deviceName_dir + "/" + sig_ref_no + ".mat"):
                    if all_info:
                        if fileType == 'mat':
                            spio.savemat(deviceName_dir + "/" + sig_ref_no + ".mat", \
                                         {'complexSignal': complexSignal, 'annotations': signal, \
                                          'metaFile': fnMeta}, long_field_names=True)
                        elif fileType == 'pkl':
                            pickle.dump({'complexSignal': complexSignal, 'annotations': signal, 'metaFile': fnMeta}, \
                                        open(deviceName_dir + "/" + sig_ref_no + ".pkl", 'wb'))
                    else:
                        if fileType == 'mat':
                            spio.savemat(deviceName_dir + "/" + sig_ref_no + ".mat", \
                                         {'complexSignal': complexSignal, 'freq_low':lowFreq, \
                                          'freq_high':upFreq, 'fs':sr, 'central_freq':Fc})
                        elif fileType == 'pkl':
                            pickle.dump({'complexSignal': complexSignal}, \
                                        open(deviceName_dir + "/" + sig_ref_no + ".pkl", 'wb'))

    #print "# Device processed:", devcnt
    return devcnt


