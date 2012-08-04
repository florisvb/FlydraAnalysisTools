import sys, os
from optparse import OptionParser

def main(path, config):

    os.mkdir(os.path.join(path, config.data_path))
    os.mkdir(os.path.join(path, config.raw_datasets_path))
    os.mkdir(os.path.join(path, config.culled_datasets_path))
    os.mkdir(os.path.join(path, config.h5_path))
    os.mkdir(os.path.join(path, config.tmp_data_path))

    os.mkdir(os.path.join(path, config.figure_path))
    for fig in config.figures:
        os.mkdir(os.path.join(path, config.figure_path, fig))

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
