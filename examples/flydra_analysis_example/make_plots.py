import flydra_analysis_tools.flydra_analysis_dataset as fad
import analysis_configuration

# plotting functions:
from plot_scripts import plot_heatmaps
from plot_scripts import plot_spagetti

def main():
    config = analysis_configuration.Config()
    culled_dataset = fad.load(config.culled_dataset_name)
    
    plot_heatmaps.main(culled_dataset, save_figure_path='figures/heatmaps/')
    plot_spagetti.main(culled_dataset, save_figure_path='figures/spagetti/')
    
if __name__ == '__main__':
    main()
