'''
This source code was developed under the DARPA Radio Frequency Machine 
Learning Systems (RFMLS) program contract N00164-18-R-WQ80. All the code 
released here is unclassified and the Government has unlimited rights 
to the code.
'''

import numpy as np
import os
import scipy.io as sio
import struct
import pickle
from optparse import OptionParser
from my_rfmls_wifi_rx import my_rfmls_wifi_rx
import time
import collections

print("after import")

root = "/src/"
# device = "wifi-dev-1"
# series = "WN-476"


def generate_bin_recursive(root, padding, step):
    for dirpath, _, filenames in os.walk(root):
        for file in filenames:
            if file.endswith('_filtered.mat'):
                vect = []
                abs_file = os.path.abspath(os.path.join(dirpath, file))

                #print("#################################")
                #print("# Analyzing file: " + abs_file)
                
                name_wo_ext = os.path.splitext(abs_file)[0] + ".bin"
                my_bin_file_name = os.path.abspath(os.path.join(dirpath, name_wo_ext))
                my_bin_file = open(my_bin_file_name, "wb")

                payload_out_name = os.path.splitext(abs_file)[0] + "-equalized_freq_iq_symbols.bin"
                my_payload_out_name = os.path.abspath(os.path.join(dirpath, payload_out_name))
                
                coarse_fo_name = os.path.splitext(abs_file)[0] + "-coarse_fo.bin"
                my_coarse_fo_name = os.path.abspath(os.path.join(dirpath, coarse_fo_name))

                fine_fo_name = os.path.splitext(abs_file)[0] + "-fine_fo.bin"
                my_fine_fo_name = os.path.abspath(os.path.join(dirpath, fine_fo_name))

                no_coarse_fo_correction_name = os.path.splitext(abs_file)[0] + "-raw_temporal_iq.bin"
                my_no_coarse_fo_correction_name = os.path.abspath(os.path.join(dirpath, no_coarse_fo_correction_name))

                channel_name = os.path.splitext(abs_file)[0] + "-channel.bin"
                my_channel_name = os.path.abspath(os.path.join(dirpath, channel_name))

                betas_name = os.path.splitext(abs_file)[0] + "-betas.bin"
                my_betas_name = os.path.abspath(os.path.join(dirpath, betas_name))

                snr_name = os.path.splitext(abs_file)[0] + "-snr.bin"
                my_snr_name = os.path.abspath(os.path.join(dirpath, snr_name))

                modulation_name = os.path.splitext(abs_file)[0] + "-modulation.bin"
                my_modulation_name = os.path.abspath(os.path.join(dirpath, modulation_name))

                res_offs_name = os.path.splitext(abs_file)[0] + "-res_offs.bin"
                my_res_offs_name = os.path.abspath(os.path.join(dirpath, res_offs_name))

                unequalized_out_name = os.path.splitext(abs_file)[0] + "-unequalized_freq_iq_symbols.bin"
                my_unequalized_out_name = os.path.abspath(os.path.join(dirpath, unequalized_out_name))

                equalized_nocnof_out_name = os.path.splitext(abs_file)[0] + "-equalized_nocnof_freq_iq_symbols.bin"
                my_equalized_nocnof_out_name = os.path.abspath(os.path.join(dirpath, equalized_nocnof_out_name))

                dic = sio.loadmat(abs_file)
                my_sig = np.array(dic['f_sig'][0]) / max([abs(x) for x in dic['f_sig'][0]])
                capture_sampling_rate = dic["fs"][0][0] 
                channel_freq = dic["f_channel"][0][0]

                #print("# Capture Sampling Rate (Hz): " + str(capture_sampling_rate))
                #print("# Channel Frequency (Hz): " + str(channel_freq))
                #print("#################################")

                vect.extend([(0 + 0j) for _ in range(padding * 2)])
                if capture_sampling_rate == 200e6:                
                    vect.extend(my_sig[::10])
                else: # Sampled at 30e6
                    vect.extend([val for val in my_sig for _ in (0, 1)][::3])
                vect.extend([(0 + 0j) for _ in range(padding * 2)])
                
                # plt.plot([abs(x) for x in vect])
                # plt.show()

                # Writing my input to the bin file that is going to be read by Gnuradio flowgraph

                for elem in vect:
                    real, imag = elem.real, elem.imag
                    my_bin_file.write(struct.pack('f', real))
                    my_bin_file.write(struct.pack('f', imag))
                
                my_bin_file.close()

                tb = my_rfmls_wifi_rx(
                    channel_freq,
                    my_bin_file_name,
                    my_payload_out_name, 
                    my_coarse_fo_name, 
                    my_fine_fo_name,
                    my_no_coarse_fo_correction_name,
                    my_channel_name,
                    my_betas_name,
                    my_snr_name,
                    my_res_offs_name,
                    my_unequalized_out_name,
                    my_equalized_nocnof_out_name,
                    my_modulation_name
                )

                tb.start()
                if channel_freq > 3e9:
                # Allow the flowgraph to run for specific time 
                    time.sleep(0.4)
                else:
                    time.sleep(0.2)
                os.remove(my_bin_file_name)

                results = collections.defaultdict()
                raw_temporal_iq = collections.defaultdict()
                phy_payload_iq = collections.defaultdict()
                raw_freq_iq = collections.defaultdict()
                phy_payload_nocnof_iq = collections.defaultdict()

                data = np.fromfile(open(my_no_coarse_fo_correction_name, "rb"), '<f4') # Little endian

                if data.shape[0]: # Means that we have at least detected the packet

                    my_data = data.tolist()
                    complex_data = np.array([complex(a, b) for a, b in zip(my_data[::2], my_data[1::2]) if a != 0 and b != 0 and not np.isnan(a)])
                    raw_temporal_iq['raw_temporal_iq'] = complex_data
                    
                    my_data = np.fromfile(open(my_payload_out_name, "rb"), '<f4') # Little endian
                    complex_data = np.array([complex(a, b) for a, b in zip(my_data[::2], my_data[1::2]) if a != 0 and b != 0 and not np.isnan(a)])
                    phy_payload_iq['phy_payload_iq'] = complex_data
                    
                    my_data = np.fromfile(open(my_coarse_fo_name, "rb"), '<f4') # Little endian
                    results['coarse_fo'] = my_data                    

                    my_data = np.fromfile(open(my_fine_fo_name, "rb"), '<f4') # Little endian
                    results['fine_fo'] = my_data

                    my_data = np.fromfile(open(my_modulation_name, "rb"), '<f4') # Little endian
                    results['payload_iq_modulation'] = my_data

                    my_data = np.fromfile(open(my_channel_name, "rb"), '<f4') # Little endian
                    complex_data = np.array([complex(a, b) for a, b in zip(my_data[::2], my_data[1::2]) if not np.isnan(a)])
                    results['channel_taps'] = complex_data
                    
                    my_data = np.fromfile(open(my_betas_name, "rb"), '<f4') # Little endian
                    results['estimated_betas'] = my_data
                    
                    my_data = np.fromfile(open(my_snr_name, "rb"), '<f4') # Little endian
                    results['estimated_snr'] = my_data

                    my_data = np.fromfile(open(my_res_offs_name, "rb"), '<f4') # Little endian
                    results['estimated_res_offsets'] = my_data
                    
                    my_data = np.fromfile(open(my_unequalized_out_name, "rb"), '<f4') # Little endian
                    complex_data = np.array([complex(a, b) for a, b in zip(my_data[::2], my_data[1::2]) if a != 0 and b != 0 and not np.isnan(a)])
                    raw_freq_iq['raw_freq_iq'] = complex_data

                    my_data = np.fromfile(open(my_equalized_nocnof_out_name, "rb"), '<f4') # Little endian
                    complex_data = np.array([complex(a, b) for a, b in zip(my_data[::2], my_data[1::2]) if a != 0 and b != 0 and not np.isnan(a)])
                    phy_payload_nocnof_iq['phy_payload_no_offsets_iq'] = complex_data
                                        
                #pickle.dump(results, open(os.path.splitext(abs_file)[0] + "-results.pkl", "wb"))
                pickle.dump(phy_payload_nocnof_iq, open(os.path.splitext(abs_file)[0] + "-phy_payload_no_offsets_iq.pkl", "wb"))
                #pickle.dump(raw_freq_iq, open(os.path.splitext(abs_file)[0] + "-raw-freq-iq.pkl", "wb"))
                #pickle.dump(phy_payload_iq, open(os.path.splitext(abs_file)[0] + "-phy_payload_iq.pkl", "wb"))
                #pickle.dump(raw_temporal_iq, open(os.path.splitext(abs_file)[0] + "-raw_temporal_iq.pkl", "wb"))
                
                os.remove(my_no_coarse_fo_correction_name)
                os.remove(my_payload_out_name)   
                os.remove(my_coarse_fo_name)
                os.remove(my_fine_fo_name)
                os.remove(my_channel_name)
                os.remove(my_betas_name)
                os.remove(my_snr_name)
                os.remove(my_res_offs_name)
                os.remove(my_unequalized_out_name)                
                os.remove(my_equalized_nocnof_out_name)
                os.remove(my_modulation_name)

def argument_parser():
    parser = OptionParser()
    parser.add_option("-d", "--decimation",
                      dest="decimation", type=int, default=10,
                      help="Decimation factor")
    parser.add_option("-p", "--padding",
                      dest="padding", type=int, default=1000,
                      help="Padding before and after the packet")
    parser.add_option("-r", "--root",
                      dest="root", default=".",
                      help="Root directory for the script")
    return parser


def main(options=None):
    if options is None:
        options, _ = argument_parser().parse_args()

    # generate_bin_file(options.device, options.series)
    generate_bin_recursive(
        options.root, 
        options.padding, 
        options.decimation
    )


if __name__ == "__main__":
    main()
