# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Rfmls Wifi Rx
# Generated: Tue Nov  6 15:07:26 2018
##################################################

from gnuradio import blocks
from gnuradio import fft
from gnuradio import gr
from gnuradio.fft import window
from gnuradio.filter import firdes
import ieee802_11_swig as ieee802_11

class my_rfmls_wifi_rx(gr.top_block):

    def __init__(
        self,
        channel_freq,
        input_file_name,
        file_name_payload_symbs_out,
        file_name_coarse_fo_out,
        file_name_fine_fo_out,
        file_name_no_coarse_fo_correction_out,
        file_name_channel_out,
        file_name_betas_out,
        file_name_snr_out,
        file_name_res_offs_out,
        file_name_unequalized_out,
        file_name_equalized_nocnof_out,
        file_name_modulation_out
    ):
        gr.top_block.__init__(
            self, "Rfmls Wifi Rx"
        )

        ##################################################
        # Variables
        ##################################################
        self.window_size = window_size = 48
        self.sync_length = sync_length = 320
        self.samp_rate = samp_rate = int(20e6)
        self.chan_est = chan_est = 0

        ##################################################
        # Blocks
        ##################################################
        self.ieee802_11_sync_short_0 = ieee802_11.sync_short(0.56, 2, False, False)
        self.ieee802_11_sync_long_0 = ieee802_11.sync_long(sync_length, False, False)
        self.ieee802_11_parse_mac_0 = ieee802_11.parse_mac(False, True)
        self.ieee802_11_moving_average_xx_1 = ieee802_11.moving_average_ff(window_size + 16)
        self.ieee802_11_moving_average_xx_0 = ieee802_11.moving_average_cc(window_size)
        self.ieee802_11_frame_equalizer_0 = ieee802_11.frame_equalizer(chan_est, channel_freq, samp_rate, False, False)
        self.ieee802_11_decode_mac_0 = ieee802_11.decode_mac(False, True)
        self.fft_vxx_0 = fft.fft_vcc(64, True, (window.rectangular(64)), True, 1)
        self.fft_vxx_1 = fft.fft_vcc(64, True, (window.rectangular(64)), True, 1)
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, 64)
        self.blocks_stream_to_vector_1 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, 64)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_gr_complex*1, input_file_name, False)
        self.blocks_divide_xx_0 = blocks.divide_ff(1)
        self.blocks_delay_0_0 = blocks.delay(gr.sizeof_gr_complex*1, 16)
        self.blocks_delay_0 = blocks.delay(gr.sizeof_gr_complex*1, sync_length)
        self.blocks_delay_1 = blocks.delay(gr.sizeof_gr_complex*1, sync_length)
        self.blocks_conjugate_cc_0 = blocks.conjugate_cc()
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(1)
        self.blocks_complex_to_mag_0 = blocks.complex_to_mag(1)

        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.ieee802_11_decode_mac_0, 'out'), (self.ieee802_11_parse_mac_0, 'in'))      
        self.connect((self.blocks_complex_to_mag_0, 0), (self.blocks_divide_xx_0, 0))    
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.ieee802_11_moving_average_xx_1, 0))    
        self.connect((self.blocks_conjugate_cc_0, 0), (self.blocks_multiply_xx_0, 1)) 

        self.connect((self.ieee802_11_sync_short_0, 0), (self.blocks_delay_0, 0))  
        self.connect((self.ieee802_11_sync_short_0, 1), (self.blocks_delay_1, 0))

        self.connect((self.ieee802_11_sync_short_0, 0), (self.ieee802_11_sync_long_0, 0))   
        self.connect((self.blocks_delay_0, 0), (self.ieee802_11_sync_long_0, 1))   
        self.connect((self.blocks_delay_1, 0), (self.ieee802_11_sync_long_0, 2)) 
        
        self.connect((self.blocks_delay_0_0, 0), (self.blocks_conjugate_cc_0, 0))    
        self.connect((self.blocks_delay_0_0, 0), (self.ieee802_11_sync_short_0, 0))    
        self.connect((self.blocks_divide_xx_0, 0), (self.ieee802_11_sync_short_0, 2))    
        self.connect((self.blocks_file_source_0, 0), (self.blocks_throttle_0, 0))    
        self.connect((self.blocks_multiply_xx_0, 0), (self.ieee802_11_moving_average_xx_0, 0))    
 

        self.connect((self.blocks_throttle_0, 0), (self.blocks_complex_to_mag_squared_0, 0))    
        self.connect((self.blocks_throttle_0, 0), (self.blocks_delay_0_0, 0))    
        self.connect((self.blocks_throttle_0, 0), (self.blocks_multiply_xx_0, 0))    
  
        self.connect((self.ieee802_11_frame_equalizer_0, 0), (self.ieee802_11_decode_mac_0, 0))    
        self.connect((self.ieee802_11_moving_average_xx_0, 0), (self.blocks_complex_to_mag_0, 0))    
        self.connect((self.ieee802_11_moving_average_xx_0, 0), (self.ieee802_11_sync_short_0, 1))    
        self.connect((self.ieee802_11_moving_average_xx_1, 0), (self.blocks_divide_xx_0, 1))  

        self.connect((self.ieee802_11_sync_long_0, 0), (self.blocks_stream_to_vector_0, 0)) 
        self.connect((self.ieee802_11_sync_long_0, 1), (self.blocks_stream_to_vector_1, 0))   

        self.connect((self.blocks_stream_to_vector_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.blocks_stream_to_vector_1, 0), (self.fft_vxx_1, 0)) 

        self.connect((self.fft_vxx_0, 0), (self.ieee802_11_frame_equalizer_0, 0))
        self.connect((self.fft_vxx_1, 0), (self.ieee802_11_frame_equalizer_0, 1))    
        
        #####################################################
        # File sinks for debug
        #####################################################

        # Debugging payload after demodulation

        self.blocks_pdu_to_tagged_stream_payload = blocks.pdu_to_tagged_stream(blocks.complex_t, "packet_len")
        self.blocks_file_sink_payload = blocks.file_sink(gr.sizeof_gr_complex*1, file_name_payload_symbs_out, False)
        self.blocks_file_sink_payload.set_unbuffered(True)
        self.msg_connect((self.ieee802_11_frame_equalizer_0, 'symbols'), (self.blocks_pdu_to_tagged_stream_payload, 'pdus'))  
        self.connect((self.blocks_pdu_to_tagged_stream_payload, 0), (self.blocks_file_sink_payload, 0))  

        # Debugging symbols with/without CFO correction and CFO itself after sync_short

        self.blocks_pdu_to_tagged_stream_coarse_fo = blocks.pdu_to_tagged_stream(blocks.float_t, "packet_len")
        self.blocks_file_sink_coarse_fo = blocks.file_sink(gr.sizeof_float, file_name_coarse_fo_out, False)
        self.blocks_file_sink_coarse_fo.set_unbuffered(True)
        self.msg_connect((self.ieee802_11_sync_short_0, 'coarse_freq_offset'), (self.blocks_pdu_to_tagged_stream_coarse_fo, 'pdus'))  
        self.connect((self.blocks_pdu_to_tagged_stream_coarse_fo, 0), (self.blocks_file_sink_coarse_fo, 0))   

        self.blocks_file_sink_no_coarse_fo_correction = blocks.file_sink(gr.sizeof_gr_complex*1, file_name_no_coarse_fo_correction_out, False)
        self.blocks_file_sink_no_coarse_fo_correction.set_unbuffered(True)
        self.connect((self.ieee802_11_sync_short_0, 1), (self.blocks_file_sink_no_coarse_fo_correction, 0)) 

        # FFO

        self.blocks_pdu_to_tagged_stream_fine_fo = blocks.pdu_to_tagged_stream(blocks.float_t, "packet_len")
        self.blocks_file_sink_fine_fo = blocks.file_sink(gr.sizeof_float, file_name_fine_fo_out, False)
        self.blocks_file_sink_fine_fo.set_unbuffered(True)
        self.msg_connect((self.ieee802_11_sync_long_0, 'fine_freq_offset'), (self.blocks_pdu_to_tagged_stream_fine_fo, 'pdus'))  
        self.connect((self.blocks_pdu_to_tagged_stream_fine_fo, 0), (self.blocks_file_sink_fine_fo, 0))   


        self.blocks_pdu_to_tagged_stream_channel = blocks.pdu_to_tagged_stream(blocks.complex_t, "packet_len")
        self.blocks_file_sink_channel = blocks.file_sink(gr.sizeof_gr_complex, file_name_channel_out, False)
        self.blocks_file_sink_channel.set_unbuffered(True)
        self.msg_connect((self.ieee802_11_frame_equalizer_0, 'channel'), (self.blocks_pdu_to_tagged_stream_channel, 'pdus'))  
        self.connect((self.blocks_pdu_to_tagged_stream_channel, 0), (self.blocks_file_sink_channel, 0))   

        # Betas

        self.blocks_pdu_to_tagged_stream_betas = blocks.pdu_to_tagged_stream(blocks.float_t, "packet_len")
        self.blocks_file_sink_betas = blocks.file_sink(gr.sizeof_float, file_name_betas_out, False)
        self.blocks_file_sink_betas.set_unbuffered(True)
        self.msg_connect((self.ieee802_11_frame_equalizer_0, 'betas'), (self.blocks_pdu_to_tagged_stream_betas, 'pdus'))  
        self.connect((self.blocks_pdu_to_tagged_stream_betas, 0), (self.blocks_file_sink_betas, 0))  

        # SNR 

        self.blocks_pdu_to_tagged_stream_snr = blocks.pdu_to_tagged_stream(blocks.float_t, "packet_len")
        self.blocks_file_sink_snr = blocks.file_sink(gr.sizeof_float, file_name_snr_out, False)
        self.blocks_file_sink_snr.set_unbuffered(True)
        self.msg_connect((self.ieee802_11_frame_equalizer_0, 'snr'), (self.blocks_pdu_to_tagged_stream_snr, 'pdus'))  
        self.connect((self.blocks_pdu_to_tagged_stream_snr, 0), (self.blocks_file_sink_snr, 0))  

        # RESIDUAL OFFSETS

        self.blocks_pdu_to_tagged_stream_res_offs = blocks.pdu_to_tagged_stream(blocks.float_t, "packet_len")
        self.blocks_file_sink_res_offs = blocks.file_sink(gr.sizeof_float, file_name_res_offs_out, False)
        self.blocks_file_sink_res_offs.set_unbuffered(True)
        self.msg_connect((self.ieee802_11_frame_equalizer_0, 'res_offs'), (self.blocks_pdu_to_tagged_stream_res_offs, 'pdus'))  
        self.connect((self.blocks_pdu_to_tagged_stream_res_offs, 0), (self.blocks_file_sink_res_offs, 0))   

        # UNEQUALIZED 

        self.blocks_pdu_to_tagged_stream_unequalized = blocks.pdu_to_tagged_stream(blocks.complex_t, "packet_len")
        self.blocks_file_sink_unequalized = blocks.file_sink(gr.sizeof_gr_complex, file_name_unequalized_out, False)
        self.blocks_file_sink_unequalized.set_unbuffered(True)
        self.msg_connect((self.ieee802_11_frame_equalizer_0, 'unequalized'), (self.blocks_pdu_to_tagged_stream_unequalized, 'pdus'))  
        self.connect((self.blocks_pdu_to_tagged_stream_unequalized, 0), (self.blocks_file_sink_unequalized, 0))  

        # EQUALIZED NO COARSE NO FINE

        self.blocks_pdu_to_tagged_stream_equalized_nocnof = blocks.pdu_to_tagged_stream(blocks.complex_t, "packet_len")
        self.blocks_file_sink_equalized_nocnof = blocks.file_sink(gr.sizeof_gr_complex, file_name_equalized_nocnof_out, False)
        self.blocks_file_sink_equalized_nocnof.set_unbuffered(True)
        self.msg_connect((self.ieee802_11_frame_equalizer_0, 'equalized_no_offsets'), (self.blocks_pdu_to_tagged_stream_equalized_nocnof, 'pdus'))  
        self.connect((self.blocks_pdu_to_tagged_stream_equalized_nocnof, 0), (self.blocks_file_sink_equalized_nocnof, 0))   

        # MODULATION

        self.blocks_pdu_to_tagged_stream_modulation = blocks.pdu_to_tagged_stream(blocks.float_t, "packet_len")
        self.blocks_file_sink_modulation = blocks.file_sink(gr.sizeof_float, file_name_modulation_out, False)
        self.blocks_file_sink_modulation.set_unbuffered(True)
        self.msg_connect((self.ieee802_11_frame_equalizer_0, 'modulation'), (self.blocks_pdu_to_tagged_stream_modulation, 'pdus'))  
        self.connect((self.blocks_pdu_to_tagged_stream_modulation, 0), (self.blocks_file_sink_modulation, 0))  





