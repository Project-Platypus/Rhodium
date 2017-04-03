## Installation Instructions ##

### Prerequisite Software ###

Please install all software listed below.  You should allow all three programs to update your PATH environment variable.

  * [Python 3.5](https://www.continuum.io/downloads) (we strongly recommend using Anaconda Python, especially on Windows)
  * [Git](https://git-scm.com/downloads)
  * [GraphViz](http://www.graphviz.org/Download.php) (for generating CART's tree views)

Attention Mac users: See the troubleshooting section for information on installing GraphViz.

### Setting up Rhodium ###

  1. Clone the Git repositories

     * In the command prompt, create a folder where the code repositories will be stored
     * Run the following commands
     * git clone https://github.com/Project-Platypus/PRIM.git
     * git clone https://github.com/Project-Platypus/Platypus.git
     * git clone https://github.com/Project-Platypus/Rhodium.git

  2. Build the Git repositories (which will also install all Python dependencies)

     * In a command prompt window, navigate to the PRIM folder
     * Run: python setup.py develop
     * Repeat for Platypus and Rhodium (in order)

  3. Run Examples

     * E.g., navigate to PRIM folder and run: python example.py


### Running IPython Example ###

  1. In the command prompt, navigate to the Rhodium folder

  2. Run: ipython notebook

  3. Open Rhodium.ipynb and evaluate the cells


### Setting up a Development Environment in Eclipse ###

  1. Install the latest version of Java (http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)

  2. Download the latest version of the Eclipse IDE for Java Developers (https://www.eclipse.org/downloads/eclipse-packages/)

  3. Run Eclipse

  4. Install PyDev

     * Open Help > Eclipse Marketplace
     * Search for PyDev
     * Click Install and follow the instructions to complete the installation

  5. Configure PyDev

     * Open Window > Preferences
     * Selected PyDev > Interpreters > Python Interpreters
     * Click New
     * Click Browse, go to the Python/Anaconda installation folder, and select python.exe
     * Click Ok/Next until you return to the Preferences window, click Ok to close the Preferences window

  6. Create PyDev projects for the Git repositories

     * Within Eclipse, select File > New > Other
     * Select PyDev > PyDev Project
     * Uncheck "Use Default"
     * Enter the project name (e.g., PRIM)
     * Click Browse and select one of the Git folders (e.g., PRIM)
     * Change Grammar Version to 3.0-3.5
     * Click Finish
     * If it asks you to change to the PyDev perspective, click Yes
     * Repeat this process for the other repositories

  7. Test

     * Within Eclipse, run some of the examples
     * E.g., Find PRIM > example.py.  Right-click and select Run As > Python Run.

### Troubleshooting ###

  1. MacOS users may have trouble installing GraphVis on new versions of the operating system.  If using Anaconda, you can run the following command to install GraphViz:
  
       ```
       conda install -c rmg graphviz=2.38.0
       ```
     
  2. Older versions of scikit-learn do not support colors in graphs (e.g., CART trees).  To enable colors, upgrade the scikit-learn version >= 0.17.  For example:
  
       ```
       conda update conda
       conda install scikit-learn=0.18.1
       ```


### Optional Python Modules ###

  * images2gif
  * J3Py