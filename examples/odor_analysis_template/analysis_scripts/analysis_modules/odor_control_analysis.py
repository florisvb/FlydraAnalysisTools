import sys
sys.path.append('/home/caveman/src/flydra_analysis_tools')
import flydra_analysis_plot as fap
import numpy as np
import matplotlib.pyplot as plt


############### written for hour on hour off experiments #######################


def read_odor_control_file(filename):
    
    f = open(filename)
    lines = f.readlines()
    
    # first extract time_start
    time_start_line = lines[0]
    time_start_string = time_start_line.split(',')[1][0:-1]
    time_start = float(time_start_string)
    
    # make list of ranges when odor is on
    new_odor_chunk = None
    odor_chunks = []
    for l, line in enumerate(lines[1:]):
        ssr_state = line.split(',')[0]
        time_string = line.split(',')[1][0:-1]
        time_switch = float(time_string) + time_start
        
        if new_odor_chunk is None:
            if ssr_state == 'on':
                new_odor_chunk = [time_switch]
        if new_odor_chunk is not None:
            if ssr_state == 'off':
                new_odor_chunk.append(time_switch)
                odor_chunks.append(new_odor_chunk)
                new_odor_chunk = None
    return odor_chunks
    
def is_time_in_odor_chunk(t, odor_chunks):

    def is_in_range(value, rang):
        if value>rang[0] and value<rang[1]:
            return True
        else:
            return False

    for chunk in odor_chunks:
        in_range = is_in_range(t, chunk)
        if in_range:
            return True
    return False
        

def calc_odor_vs_no_odor_simple(dataset, odor_timestamp_filename):
    
    # first construct odor on/off function, that takes time.time() as input.
    odor_chunks = read_odor_control_file(odor_timestamp_filename)
    
    # find trajectories that start and end with odor_stimulus on
    # flag them with trajec.odor = True
    for key, trajec in dataset.trajecs.items():
        in_odor_start = is_time_in_odor_chunk(trajec.timestamp_epoch, odor_chunks)
        in_odor_finish = is_time_in_odor_chunk(trajec.time_fly[-1] + trajec.timestamp_epoch, odor_chunks)
        if in_odor_start and in_odor_finish:
            trajec.in_odor = True
        elif not in_odor_start and not in_odor_finish:
            trajec.in_odor = False
        else:
            trajec.in_odor = None
            print 'none'
    
    
################################################################################
    
