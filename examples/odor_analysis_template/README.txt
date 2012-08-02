This is an example directory containing all the necessary code to analyze flydra data using the flydra_analysis_tools toolkit. The filestructure is intended to help keep everything organized for when projects get huge.

From inside this directory, with flydra_analysis_tools and flydra installed, run:

>> python ./analyze.py

If you do not have flydra, you can still run the example:
- change analysis_configuration.Config.h5_files to None
- then run analyze.py

What analyze.py does:
- will first analyze all the h5 files and save them in the flydra_analysis_dataset format (unless analysis_configuration.Config.h5_files = None)
- Next the data is culled and additional functions can be run
- Lastly, a variety of plots can be generated

Use the analysis_configuration.py file to save a variety of settings that scripts throughout the analysis directory may use.

Recommended usage: git clone/fork this directory and use it as a template to build a more complex/custom analysis, making the appropriate commits. This way in future experiments you can cherry-pick a directory that minimizes the work you need to do.

====
Using the code with your data
====

The simplest usage is to clone/fork this entire directory and use it as a template. Replace the h5 files in the h5 directory with your h5 file(s), delete the dataset.pickle constructs in raw_datasets and culled_datasets, and run analyze.py. From there you can start making the appropriate changes.

Note: save_h5_to_dataset.py may take several hours to complete its work, however, once finished dealing with the new dataset construct is very fast. Thus, it's best to only run save_h5_to_dataset.py when you first start working with an h5 file. 
