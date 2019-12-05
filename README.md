# MapNet: An Allocentric Spatial Memory for Mapping Environments

This is a PyTorch re-implementation of MapNet, presented in:

> João F. Henriques and Andrea Vedaldi, "MapNet: An Allocentric Spatial Memory for Mapping Environments", CVPR 2018 ([PDF](https://robots.ox.ac.uk/~joao/publications/henriques_cvpr2018.pdf))

It reproduces all of the training from scratch for the mazes experiments, but not the Doom or AVD experiments; I hope to change that in the future.


## Requirements

Although it may work with older versions, this has mainly been tested with:

- PyTorch 1.3
- Python 3.7
- [OverBoard](https://pypi.org/project/overboard/) 0.1.4 (for plotting and visualization)


## Usage

The mazes are stored in a large text file (45 MB). For this reason, it is zipped in `data/maze/mazes-10-10-100000.zip` (6 MB), please extract its contents to the same directory.

Training can then be performed by running `train_mapnet.py`. Run `train_mapnet.py --help` for command-line options and their explanation.


## Visualization

Plots and tensor visualizations (mostly heatmaps of the joint position-orientation probability, as well as the maps) from OverBoard:

![Screenshot](https://github.com/jotaf98/mapnet/raw/master/data/screenshot.png)


# Author

[João F. Henriques](http://www.robots.ox.ac.uk/~joao/)

