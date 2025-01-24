# A Quick Check of VPIC Data
The GUI program is called `quick_check_vpic.py`. It is similar to the IDL program `diagnostic.pro` but does not have that many functionalities. It works for both 2D and 3D runs, but more tests are needed to make sure that it works smoothly.

## Usage
In the VPIC run directory,
```sh
git clone https://github.com/xiaocanli/quick_check_vpic
cd quick_check_vpic
module load python
python -m pip install .
```
It will install the required packages if they are already in the system.

- On Perlmutter@NERSC, I recommend creating a conda environment and then installing the packages. Please follow the instructions at [nersc-python](https://docs.nersc.gov/development/languages/python/nersc-python/). You can use the GUI program through [NERSC NoMachine/NX](https://docs.nersc.gov/connect/nx/), which is much faster than directly launching it through ssh connection in the terminal.

You need to change a few parameters near the top of the program: `hdf5_fields`, `smoothed_data`, `smooth_factor`, `dir_smooth_data`, `momentum_field`, `time_averaged_field`, `turbulence_mixing`, `tmin`, `tmax`, and `animation_tinterval`. After making the changes, you can launch the GUI program in the simulation directory.
```sh
python3 quick_check_vpic.py
```
or use `python` instead if `python3` is in default.

## Customization
If you are interested in modifying it, the layout `mainwindow.ui` is created using Qt Creator, which is free to use. You need to convert it to a Python file before using it in the GUI program.
```sh
pyside6-uic mainwindow.ui -o mainwindow.py
```
For particle tracers, please modify `tracer_filepath` and `tracer_filename`. We assume that the tracer trajectories are saved in an HDF5 file, and each tracer particle data is saved in an individual group. The group name does not matter because it is not used in the code by default. Since the tracer trajectory is overplotted on the canvas, you need to plot a field image first. You can use the spinbox to select the tracer index.

## Issues
Please submit issues if you find problems or directly contact me for help.