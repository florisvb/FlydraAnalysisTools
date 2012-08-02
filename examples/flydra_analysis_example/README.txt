This is an example directory containing all the necessary code to analyze flydra data using the flydra_analysis_tools toolkit. The filestructure is intended to help keep everything organized for when projects get huge.

From inside this directory, with flydra_analysis_tools installed, run:

>> python ./analyze.py

This script will first analyze all the h5 files and save them in the flydra_analysis_dataset format
Next the data is culled and additional functions can be run
Lastly, a variety of plots can be generated

Recommended usage: git clone/fork this directory, and then build your analysis, making the appropriate commits. This way in future experiments you can cherry-pick a directory that minimizes the work you need to do.

