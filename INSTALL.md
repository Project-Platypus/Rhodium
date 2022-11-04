## Installation Instructions ##

### Prerequisite Software ###

Please install all software listed below.  You should allow all three programs to update your PATH environment variable.

  * [Python 3.6](https://www.python.org/) or newer
  * [Git](https://git-scm.com/downloads)
  * [GraphViz](http://www.graphviz.org/Download.php) (for generating CART's tree views)

### Setting up Rhodium ###

```
pip install -U build setuptools
git clone https://github.com/Project-Platypus/Rhodium.git
cd Rhodium
python -m build
```

### Running Examples

Try running the examples:

```
cd examples/Basic
python example.py
```

### Running IPython Example ###

  1. From the Rhodium folder, run: `ipython notebook`
  2. Open `Rhodium.ipynb` and evaluate the cells

### Optional Dependencies ###

  * pywin32 - To connect to Excel models
  * images2gif - To save GIF animations of 3D plots
  * J3Py - Interactive 3D visualizations (powered by J3)