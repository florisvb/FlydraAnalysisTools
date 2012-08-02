import save_h5_to_dataset
import prep_dataset
import make_plots
import analysis_configuration
import generate_empy_data_directory

config = analysis_configuration.Config()

try:
    generate_empty_data_directory
except:
    'Could not make empty data directory -- probably exists already!'

if config.h5_files is not None:
    save_h5_to_dataset.main()

prep_dataset.main()
make_plots.main()


