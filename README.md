# centerfusion-vod-training

## Set Up

**Create and activate virtual environment**

```shell
    python -m venv venv
    venv/bin/activate (macos/linux)
    venv\Scripts\activate (Windows bash)
    venv\Scripts\Activate.ps1 (Windows powershell)
```

**Install python dependencies**

```shell
    pip install -r requirements
```

**Clone into view of delft**

```shell
    # clone it into the data folder
    cd data
    git clone https://github.com/tudelft-iv/view-of-delft-dataset.git
```

### Changes to preprocess.py

Change the variables: `vod_repo_path`, `raw_root`, to the correct data locations

Set `output_root = r"./data/vod_processed"`

### Changes to vod

**vod/visualization/vis_2d.py**: change line 162 to `return fig, plt.gca()`

**vod/frame/transformations.py**: change line 325 from `uvs = np.round(uvs).astype(np.int)` to `uvs = np.round(uvs).astype(int)`
