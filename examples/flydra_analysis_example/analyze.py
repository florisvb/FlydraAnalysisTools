import save_h5_to_dataset
import prep_dataset
import make_plots
import analysis_configuration

config = analysis_configuration.Config()

if config.h5_files is not None:
    save_h5_to_dataset.main()

prep_dataset.main()
make_plots.main()


