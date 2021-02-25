# A Quick Check of VPIC Data
The GUI program is called `quick_check_vpic.py`. It is similar to the IDL program `diagnostic.pro` but does not have that many functionalities. It works for both 2D and 3D runs, but more tests are needed to make sure that it works smoothly. Although there is a slot for GDA files, it can only read HDF5 data now.

You can change a few parameters near the top of the program: `smoothed_data`, `smooth_factor`, `dir_smooth_data`, `tmin`, `tmax`, and `animation_tinterval`. If you are interested in modifying it, the layout `mainwindow.ui` is created using Qt4 Designer, which is free to use. You need to convert it to a Python file before using it in the GUI program.
```sh
pyuic5 mainwindow.ui -o mainwindow.py
```