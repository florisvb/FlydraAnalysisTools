import flydra_analysis_tools as fat
import analysis_configuration

def main():

    config = analysis_configuration.Config()
    kalman_smoothing = config.kalman_smoothing
    save_covariance = False
    savename = config.raw_datasets_path + config.raw_dataset_name
    
    if config.h5_files == 'all':
        fat.flydra_analysis_dataset.load_all_h5s_in_directory(config.h5_path, print_filenames_only=False, kalmanized=False, savedataset=True, savename=savename, kalman_smoothing=kalman_smoothing, dynamic_model=None, fps=None, info=config.info, save_covariance=save_covariance, tmp_path=config.tmp_data_path)

    else:
        dataset = fat.flydra_analysis_dataset.Dataset()
        for h5_file in config.h5_files:
            if config.h5_path not in h5_file:
                h5_file = config.h5_path + h5_file
            dataset.load_data(h5_file, kalman_smoothing=kalman_smoothing, dynamic_model=None, fps=None, info=config.info, save_covariance=save_covariance)

        dataset.save(savename)
        


if __name__ == '__main__':
    main()
    
