import sys, os, shutil
from optparse import OptionParser

import save_h5_to_dataset
import prep_dataset
import make_plots
import generate_empty_data_directory

def main(path, config, options):
    h5 = options.h5
    
    # make file structure
    try:
        generate_empty_data_directory.main(path, config)
    except:
        'Could not make empty data directory -- probably exists already!'
        return
        
    # move h5 file
    h5_path = os.path.join(path, config.h5_path)
    shutil.move(h5, h5_path)
    
    # save_h5_to_dataset.py: do raw dataset analysis
    if config.h5_files is not None:
        save_h5_to_dataset.main(path, config)

    # prep_dataset.py: load raw dataset, cull data, apply other functions
    prep_dataset.main(path, config)
    
    # make_plots.py: make all the plots you want
    make_plots.main(path, config)
    
    

if __name__ == '__main__':
    
    parser = OptionParser()
    parser.add_option("--path", type="str", dest="path", default='',
                        help="path to empty data folder, where you have a configuration file")
    parser.add_option("--h5", type="str", dest="h5", default='',
                        help="h5 file you wish to analyze, it will get moved to the new directory")
                        
    (options, args) = parser.parse_args()
    
    path = options.path
    sys.path.append(path)
    import analysis_configuration
    config = analysis_configuration.Config()
    
    main(path, config, options)
