import os
import analysis_configuration

def main():
    config = analysis_configuration.Config()
    os.mkdir(config.data_path)
    
    os.mkdir(config.raw_datasets_path)
    os.mkdir(config.culled_datasets_path)
    os.mkdir(config.h5_path)
    os.mkdir(config.tmp_data_path)
    os.mkdir(config.odor_control_path)

if __name__ == '__main__':
    main()
    
    print
    print 'Successfully created data directory'
