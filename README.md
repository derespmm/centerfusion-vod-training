# centerfusion-vod-training

use "python3 -m pip install -r requirements.txt" to install the necessary packages

install anaconda from the web
install mamba with "conda install mamba -n base -c conda-forge"
then run "mamba create -n view-of-delft-env python=3.7 --yes"
activate the env with "mamba activate"
comment out half of the .yml file underneath the existing comment
run "mamba env update -f environment.yml"
uncomment the bottom half
run "mamba env update -f environment.yml" again to get the remaining dependencies

change line 162 in vis_2d.py file to return fig, plt.gca()
change line 325 in transformations.py from np.int to int
