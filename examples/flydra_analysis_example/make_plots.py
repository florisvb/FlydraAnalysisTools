from plot_scripts import plot_heatmaps
import analysis_configuration
import flydra_analysis_tools.flydra_analysis_dataset as fad

def main():
    config = analysis_configuration.Config()
    culled_dataset = fad.load(config.culled_dataset_name)
    
    plot_heatmaps.main(culled_dataset, save_figure_path='figures/heatmaps/')
    
if __name__ == '__main__':
    main()
