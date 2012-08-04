import sys, os
from optparse import OptionParser

import flydra_analysis_tools as fat

def main(path, config):
    
    # some options
    kalman_smoothing = config.kalman_smoothing
    save_covariance = False
    kalmanized = config.kalmanized # use kalmanized files?
    
    # path stuff
    savename = os.path.join(path, config.raw_datasets_path, config.raw_dataset_name)
    h5_path = os.path.join(path, config.h5_path)
    tmp_path = os.path.join(path, config.tmp_data_path)
    
    if config.h5_files == 'all':
        fat.flydra_analysis_dataset.load_all_h5s_in_directory(h5_path, print_filenames_only=False, kalmanized=kalmanized, savedataset=True, savename=savename, kalman_smoothing=kalman_smoothing, dynamic_model=None, fps=None, info=config.info, save_covariance=save_covariance, tmp_path=tmp_path)

    else:
        dataset = fat.flydra_analysis_dataset.Dataset()
        for h5_file in config.h5_files:
            if config.h5_path not in h5_file:
                h5_file = os.path.join(path, config.h5_path, h5_file)
            dataset.load_data(h5_file, kalman_smoothing=kalman_smoothing, dynamic_model=None, fps=None, info=config.info, save_covariance=save_covariance)

        dataset.save(savename)
        


if __name__ == '__main__':
    
    parser = OptionParser()
    parser.add_option("--path", type="str", dest="path", default='',
                        help="path to empty data folder, where you have a configuration file")
    (options, args) = parser.parse_args()
    
    path = options.path
    sys.path.append(path)
    import analysis_configuration
    config = analysis_configuration.Config()
    
    main(path, config)
    
