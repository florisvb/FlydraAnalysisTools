#!/usr/bin/env python
import flydra_analysis_tools as fat
fad = fat.flydra_analysis_dataset
dac = fat.dataset_analysis_core
tac = fat.trajectory_analysis_core

import analysis_configuration

####################### analysis functions ###########################
def culling_function(raw_dataset):
    culled_dataset = dac.cull_dataset_min_frames(raw_dataset, min_length_frames=10, reset=True)
    culled_dataset = dac.cull_dataset_cartesian_volume(culled_dataset, [-.2, .2], [-1,.3], [-.1,.4], reset=True)
    culled_dataset = dac.cull_dataset_min_speed(culled_dataset, min_speed=0.05, reset=True)
    return culled_dataset


def prep_data(culled_dataset):
    # stuff like calculating angular velocity, saccades etc.
    fad.iterate_calc_function(culled_dataset, tac.calc_local_timestamps_from_strings) # calculate local timestamps
    return    
    
def main():
    config = analysis_configuration.Config()
    
    print 
    print 'Culling and Preparing Data'
    
    try:
        culled_dataset = fad.load(config.culled_datasets_path + config.culled_dataset_name)
        print 'Loaded culled dataset'
    except:
        try:
            raw_dataset = fad.load(config.raw_datasets_path + config.raw_dataset_name)
            print 'Loaded raw dataset'
        except:
            print 'Cannot find dataset, run save_h5_to_dataset.py first'
                
        culled_dataset = culling_function(raw_dataset) 
        print 'Preparing culled dataset'
        prep_data(culled_dataset)
        fad.save(culled_dataset, config.culled_datasets_path + config.culled_dataset_name)
        print 'Saved culled dataset'
        
    prep_data(culled_dataset)
    
    return culled_dataset
    
if __name__ == '__main__':
    culled_dataset = main()
