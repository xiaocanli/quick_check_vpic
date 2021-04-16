# A Quick Check of VPIC Data
The GUI program is called `quick_check_vpic.py`. It is similar to the IDL program `diagnostic.pro` but does not have that many functionalities. It works for both 2D and 3D runs, but more tests are needed to make sure that it works smoothly.
```sh
git clone https://github.com/xiaocanli/quick_check_vpic
```
Then, you need to copy `quick_check_vpic.py` and `mainwindow.py` to the simulation directory.

It requires `h5py`, `matplotlib`, `numpy`, and `PyQt5`.
- On Cori, I recommend creating a conda environment and then installing the packages. Please follow the instructions at [nersc-python](https://docs.nersc.gov/development/languages/python/nersc-python/). You can use the GUI program through [NERSC NoMachine/NX](https://docs.nersc.gov/connect/nx/), which is much faster than directly launching it through ssh connection in the terminal.
- On Frontera, please follow
    ```sh
    module load phdf5 python3
    module load qt5/5.14.2
    pip install --upgrade pip --user
    pip install PyQt5==5.14.2 --user
    pip install --upgrade matplotlib --user
    ```
    where the trick is to install the same version of `PyQt5` as the system `qt5`. You can also upgrade `h5py` and `numpy` similarly. Then, you can use the [TACC visualization portal](https://vis.tacc.utexas.edu/#) (VNC) to launch the GUI program.

You need to change a few parameters near the top of the program: `hdf5_fields`, `smoothed_data`, `smooth_factor`, `dir_smooth_data`, `momentum_field`, `time_averaged_field`, `tmin`, `tmax`, and `animation_tinterval`. After making the changes, you can launch the GUI program in the simulation directory.
```sh
python3 quick_check_vpic.py
```
or use `python` instead if `python3` is in default. If you are interested in modifying it, the layout `mainwindow.ui` is created using Qt4 Designer, which is free to use. You need to convert it to a Python file before using it in the GUI program.
```sh
pyuic5 mainwindow.ui -o mainwindow.py
```
