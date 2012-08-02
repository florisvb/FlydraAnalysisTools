import flydra_analysis_tools.flydra_analysis_dataset as fad
import analysis_configuration
import prep_dataset

# plotting functions:
from plot_scripts import plot_heatmaps
from plot_scripts import plot_spagetti
from plot_scripts import plot_activity_histograms

def main():
    config = analysis_configuration.Config()
    culled_dataset = prep_dataset.main()
    
    plot_heatmaps.main(culled_dataset, save_figure_path='figures/heatmaps/')
    plot_spagetti.main(culled_dataset, save_figure_path='figures/spagetti/')
    plot_activity_histograms.main(culled_dataset, save_figure_path='figures/activity_histograms/')
    
if __name__ == '__main__':
    main()
