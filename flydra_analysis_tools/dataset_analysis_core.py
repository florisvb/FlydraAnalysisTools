import flydra_analysis_dataset as fad
import trajectory_analysis_core as tac

###################################################################################################
# Commonly used, but specific, dataset functions
###################################################################################################

def cull_dataset_min_frames(dataset, min_length_frames=10, reset=True):
    if reset: fad.set_attribute_for_trajecs(dataset, 'cull', False)
    fad.iterate_calc_function(dataset, tac.mark_for_culling_based_on_min_frames, keys=None, min_length_frames=min_length_frames)
    return fad.make_dataset_with_attribute_filter(dataset, 'cull', False)

def cull_dataset_cartesian_volume(dataset, x_range, y_range, z_range, reset=True):
    if reset: fad.set_attribute_for_trajecs(dataset, 'cull', False)
    fad.iterate_calc_function(dataset, tac.mark_for_culling_based_on_cartesian_position, keys=None, ok_range=x_range, axis=0)
    fad.iterate_calc_function(dataset, tac.mark_for_culling_based_on_cartesian_position, keys=None, ok_range=y_range, axis=1)
    fad.iterate_calc_function(dataset, tac.mark_for_culling_based_on_cartesian_position, keys=None, ok_range=z_range, axis=2)
    return fad.make_dataset_with_attribute_filter(dataset, 'cull', False)
    
def cull_dataset_min_speed(dataset, min_speed=0.05, reset=True):
    if reset: fad.set_attribute_for_trajecs(dataset, 'cull', False)
    fad.iterate_calc_function(dataset, tac.mark_for_culling_based_on_speed, keys=None, min_speed=min_speed)
    return fad.make_dataset_with_attribute_filter(dataset, 'cull', False)
    
    
def calc_saccades_for_dataset(dataset):
    
    for key, trajec in dataset.trajecs.items():
        tac.calc_heading(trajec)
        tac.calc_saccades(trajec)
