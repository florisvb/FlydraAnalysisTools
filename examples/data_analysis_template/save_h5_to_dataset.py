import flydra_analysis_tools as fat

path = '/h5_files'
savename = '/datasets/dataset.pickle'

fat.flydra_analysis_dataset.load_all_h5s_in_directory(path, print_filenames_only=False, kalmanized=True, savedataset=True, savename=savename, kalman_smoothing=True, dynamic_model=None, fps=None, info={}, save_covariance=False)
