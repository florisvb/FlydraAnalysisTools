#!/usr/bin/env python
import sys, os
from optparse import OptionParser

import flydra_analysis_tools as fat
fad = fat.flydra_analysis_dataset
dac = fat.dataset_analysis_core
tac = fat.trajectory_analysis_core


####################### analysis functions ###########################
def culling_function(raw_dataset):
    culled_dataset = dac.cull_dataset_min_frames(raw_dataset, min_length_frames=10, reset=True)
    culled_dataset = dac.cull_dataset_cartesian_volume(culled_dataset, [-.2, .2], [-1,.3], [-.1,.4], reset=True)
    culled_dataset = dac.cull_dataset_min_speed(culled_dataset, min_speed=0.05, reset=True)
    return culled_dataset


def prep_data(culled_dataset, path, config):
    # stuff like calculating angular velocity, saccades etc.
    culled_dataset.info = config.info
    fad.iterate_calc_function(culled_dataset, tac.calc_local_timestamps_from_strings) # calculate local timestamps
    return    
    
def main(path, config):
    
    # path stuff
    culled_dataset_name = os.path.join(path, config.culled_datasets_path, config.culled_dataset_name)
    raw_dataset_name = os.path.join(path, config.raw_datasets_path, config.raw_dataset_name)
    
    print 
    print 'Culling and Preparing Data'
    
    try:
        culled_dataset = fad.load(culled_dataset_name)
        print 'Loaded culled dataset'
    except:
        try:
            raw_dataset = fad.load(raw_dataset_name)
            print 'Loaded raw dataset'
        except:
            print 'Cannot find dataset, run save_h5_to_dataset.py first'
                
        culled_dataset = culling_function(raw_dataset) 
        print 'Preparing culled dataset'
        prep_data(culled_dataset, path, config)
        fad.save(culled_dataset, culled_dataset_name)
        print 'Saved culled dataset'
        
    prep_data(culled_dataset, path, config)
    
    return culled_dataset
    
if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--path", type="str", dest="path", default='',
                        help="path to empty data folder, where you have a configuration file")
    (options, args) = parser.parse_args()
    
    path = options.path    
    sys.path.append(path)
    import analysis_configuration
    config = analysis_configuration.Config()
    
    culled_dataset = main(path, config)
