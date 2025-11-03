#!/usr/bin/env python3
"""
Quick check of VPIC output
"""
import argparse
import errno
import glob
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets

from mainwindow import Ui_MainWindow

mpl.use('qtagg')

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("Warning: PyYAML not installed. Using default configuration.")
    print("Install with: pip install pyyaml")


def get_vpic_info(info_file: str = '../info') -> Dict[str, float]:
    """Get information of the VPIC simulation from the info file.

    Args:
        info_file: Path to the VPIC info file

    Returns:
        Dictionary containing VPIC simulation parameters

    Raises:
        FileNotFoundError: If info file doesn't exist
        ValueError: If info file format is invalid
    """
    info_path = Path(info_file)
    if not info_path.exists():
        raise FileNotFoundError(
            f"VPIC info file not found: {info_file}\n"
            f"Please ensure the info file exists or specify the correct path."
        )

    try:
        with open(info_path) as f:
            content = f.readlines()
    except Exception as e:
        raise ValueError(f"Failed to read info file {info_file}: {e}")

    if len(content) < 2:
        raise ValueError(f"Info file {info_file} appears to be empty or invalid")

    vpic_info = {}
    for line in content[1:]:
        if "=" in line:
            line_splits = line.split("=")
        elif ":" in line:
            line_splits = line.split(":")
        else:
            continue

        try:
            tail = line_splits[1].split("\n")
            vpic_info[line_splits[0].strip()] = float(tail[0])
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not parse line in info file: {line.strip()}")
            continue

    return normalize_box_size_params(vpic_info)


def get_vpic_info_log(hdf5_fields: bool = True,
                       log_file: str = "../vpic.out",
                       dir_fields: str = "field_hdf5") -> Dict[str, float]:
    """Get information of the VPIC simulation from log file.

    This function reads VPIC parameters from the log file (vpic.out) when
    the info file is not available. This is useful for older simulations
    or hybridVPIC runs.

    Args:
        hdf5_fields: whether data is in HDF5 format
        log_file: path to the VPIC log file
        dir_fields: field directory name (field_hdf5, fields_hdf5, etc.)

    Returns:
        Dictionary containing VPIC simulation parameters

    Raises:
        FileNotFoundError: If log file doesn't exist
        ValueError: If log file format is invalid
    """
    log_path = Path(log_file)
    if not log_path.exists():
        raise FileNotFoundError(
            f"VPIC log file not found: {log_file}\n"
            f"Please ensure the log file exists or specify the correct path."
        )

    try:
        with open(log_path) as f:
            content = f.readlines()
    except Exception as e:
        raise ValueError(f"Failed to read log file {log_file}: {e}")

    vpic_info = {}
    iline = 0
    nlines = len(content)

    # Find the "System of units" section
    while iline < nlines:
        line = content[iline]
        if "System of units" in line:
            break
        iline += 1

    # Parse parameters
    while iline < nlines:
        line = content[iline]
        if "=" in line and "start" not in line:
            try:
                if "," not in line:
                    # Single variable per line
                    line_splits = line.split(":")
                    if len(line_splits) > 1:
                        var = line_splits[1].split("=")
                        if len(var) > 1:
                            vpic_info[var[0].strip()] = float(var[1].strip())
                else:
                    # Multiple variables per line
                    line_splits = line.split(":")
                    if len(line_splits) > 1:
                        data1 = line_splits[1].split("\n")
                        data2 = data1[0].split(",")
                        for var_data in data2:
                            var = var_data.split("=")
                            if len(var) > 1:
                                vpic_info[var[0].strip()] = float(var[1].strip())
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse line in log file: {line.strip()}")
        iline += 1

    # Get the intervals from output data
    try:
        if hdf5_fields:
            # Check for fields and hydro directories
            outputs = ["fields", "hydro"]
            for output in outputs:
                # Try different directory naming conventions
                possible_dirs = [
                    f"../{output}_hdf5",
                    f"../{dir_fields.replace('field', output)}",
                ]

                for dir_path in possible_dirs:
                    if os.path.exists(dir_path):
                        try:
                            file_list = os.listdir(dir_path)
                            tframes = []
                            for file_name in file_list:
                                if "T." in file_name:
                                    fsplit = file_name.split(".")
                                    tindex = int(fsplit[-1])
                                    tframes.append(tindex)
                            if len(tframes) > 1:
                                tframes = np.sort(np.asarray(tframes))
                                vpic_info[output + "_interval"] = int(tframes[1] - tframes[0])
                            break
                        except Exception as e:
                            print(f"Warning: Could not determine {output} interval: {e}")
        else:
            # For gda format
            bx_list = [os.path.basename(x) for x in glob.glob('../data/bx*.gda')]
            if len(bx_list) == 1:  # all steps in the same file
                vpic_info["fields_interval"] = 1
                vpic_info["hydro_interval"] = 1
            elif len(bx_list) > 1:
                tframes = []
                for file_name in bx_list:
                    fsplit = file_name.split(".")
                    fsplit2 = fsplit[0].split("_")
                    if len(fsplit2) > 1:
                        tindex = int(fsplit2[1])
                        tframes.append(tindex)
                if len(tframes) > 1:
                    tframes = np.sort(np.asarray(tframes))
                    vpic_info["fields_interval"] = int(tframes[1] - tframes[0])
                    vpic_info["hydro_interval"] = vpic_info["fields_interval"]
    except Exception as e:
        print(f"Warning: Could not determine output intervals: {e}")

    return normalize_box_size_params(vpic_info)


def normalize_box_size_params(vpic_info: Dict[str, float]) -> Dict[str, float]:
    """Normalize box size parameters to use /de convention.

    HybridVPIC simulations may use different conventions:
    - Full VPIC: Lx/de, Ly/de, Lz/de (electron scales)
    - HybridVPIC: Lx/di, Ly/di, Lz/di (ion scales) OR just Lx, Ly, Lz

    This function ensures Lx/de, Ly/de, Lz/de always exist by checking
    for alternative names and using them as fallbacks. It also sets
    'length_scale' to indicate which scale is being used ('de', 'di', or 'other').

    Args:
        vpic_info: Dictionary containing VPIC simulation parameters

    Returns:
        Updated dictionary with normalized parameter names and 'length_scale' key
    """
    # Detect which scale is used
    if 'Lx/de' in vpic_info:
        scale = 'de'
    elif 'Lx/di' in vpic_info:
        scale = 'di'
    elif 'Lx' in vpic_info:
        scale = 'other'
    else:
        scale = 'unknown'

    vpic_info['length_scale'] = scale

    # Normalize box size parameters (Lx, Ly, Lz)
    for dim in ['x', 'y', 'z']:
        key_de = f'L{dim}/de'
        key_di = f'L{dim}/di'
        key_plain = f'L{dim}'

        # If /de version doesn't exist, use /di or plain version
        if key_de not in vpic_info:
            if key_di in vpic_info:
                vpic_info[key_de] = vpic_info[key_di]
                if dim == 'x':  # Only print once
                    print(f"Info: Using ion scale (di) for length units")
            elif key_plain in vpic_info:
                vpic_info[key_de] = vpic_info[key_plain]
                if dim == 'x':  # Only print once
                    print(f"Info: Using plain length units (no scale suffix)")

    # Normalize cell size parameters (dx, dy, dz)
    for dim in ['x', 'y', 'z']:
        key_de = f'd{dim}/de'
        key_di = f'd{dim}/di'
        key_plain = f'd{dim}'

        # If /de version doesn't exist, use /di or plain version
        if key_de not in vpic_info:
            if key_di in vpic_info:
                vpic_info[key_de] = vpic_info[key_di]
            elif key_plain in vpic_info:
                vpic_info[key_de] = vpic_info[key_plain]

    return vpic_info


def mkdir_p(path: str) -> None:
    """Create directory with parents, ignore if exists.

    Args:
        path: Directory path to create
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


@dataclass
class Config:
    """Configuration parameters for VPIC visualization."""

    vpic_info: Dict[str, float] = field(default_factory=dict)
    info_file: str = "../info"
    log_file: str = "../vpic.out"  # VPIC log file (fallback if info file not found)
    hdf5_fields: bool = True
    smoothed_data: bool = True
    smooth_factor: int = 2
    dir_smooth_data: str = "data_smooth"
    dir_fields_hdf5: str = "field_hdf5"  # Field directory name (field_hdf5 or fields_hdf5)
    dir_beta_hdf5: str = "hydro-int-hdf5"  # Beta diagnostics directory name
    hdf5_data_order: str = "xyz"  # HDF5 data order: "xyz" (old) or "zyx" (new)
    momentum_field: bool = True
    time_averaged_field: bool = False
    turbulence_mixing: bool = False
    auto_detect_species: bool = True  # Auto-detect species from hydro files
    species_list: List[str] = field(default_factory=list)  # Manually specify species if not auto-detecting
    tmin: int = 0
    tmax: int = 52
    animation_tinterval: int = 100
    tracer_filepath: str = "/home/cannon/Research/quick_check_vpic/vpic-sorter/data/2D-Lx150-bg0.2-150ppc/"
    tracer_filename: str = "electrons_ntraj1000_10emax.h5p"

    def __post_init__(self):
        """Adjust configuration based on settings."""
        if not self.smoothed_data:
            self.smooth_factor = 1
        if self.time_averaged_field and self.tmin == 0:
            self.tmin = 1
        self.nt = self.tmax - self.tmin + 1

        # Load vpic_info if not already loaded
        if not self.vpic_info:
            try:
                self.vpic_info = get_vpic_info(self.info_file)
            except FileNotFoundError:
                # Try to read from log file as fallback
                print(f"Info file '{self.info_file}' not found, trying to read from log file '{self.log_file}'...")
                try:
                    self.vpic_info = get_vpic_info_log(
                        hdf5_fields=self.hdf5_fields,
                        log_file=self.log_file,
                        dir_fields=self.dir_fields_hdf5
                    )
                    print(f"Successfully loaded VPIC info from log file '{self.log_file}'.")
                except (FileNotFoundError, ValueError) as e:
                    print(f"Error: Could not read from log file '{self.log_file}': {e}")
                    print("Using empty vpic_info. Some features may not work correctly.")
                    self.vpic_info = {}

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Config instance
        """
        if not HAS_YAML:
            print(f"Cannot load {yaml_path}: PyYAML not installed")
            return cls()

        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            print(f"Config file not found: {yaml_path}")
            return cls()

        try:
            with open(yaml_file) as f:
                config_data = yaml.safe_load(f)

            # Flatten nested structure
            flat_config = {}
            if 'data' in config_data:
                flat_config.update(config_data['data'])
            if 'time' in config_data:
                flat_config.update(config_data['time'])
            if 'tracer' in config_data:
                tracer = config_data['tracer']
                flat_config['tracer_filepath'] = tracer.get('filepath', cls.tracer_filepath)
                flat_config['tracer_filename'] = tracer.get('filename', cls.tracer_filename)
            if 'info_file' in config_data:
                flat_config['info_file'] = config_data['info_file']

            return cls(**flat_config)
        except Exception as e:
            print(f"Error loading config from {yaml_path}: {e}")
            print("Using default configuration")
            return cls()

    @classmethod
    def auto_detect(cls, info_file: str = '../info') -> 'Config':
        """Auto-detect configuration from available files.

        Args:
            info_file: Path to VPIC info file

        Returns:
            Config instance with auto-detected settings
        """
        config_dict = {'info_file': info_file}

        # Auto-detect HDF5 vs GDA format and field directory name
        if Path('../fields_hdf5').exists():
            config_dict['hdf5_fields'] = True
            config_dict['dir_fields_hdf5'] = 'fields_hdf5'
            print("Auto-detected: HDF5 field format (fields_hdf5/)")
        elif Path('../field_hdf5').exists():
            config_dict['hdf5_fields'] = True
            config_dict['dir_fields_hdf5'] = 'field_hdf5'
            print("Auto-detected: HDF5 field format (field_hdf5/)")
        elif Path('../fields-avg-hdf5').exists():
            config_dict['hdf5_fields'] = True
            config_dict['dir_fields_hdf5'] = 'fields-avg-hdf5'
            print("Auto-detected: HDF5 field format (fields-avg-hdf5/)")
        elif Path('../data').exists():
            config_dict['hdf5_fields'] = False
            print("Auto-detected: GDA field format")

        # Auto-detect smoothed data
        if Path('../data_smooth').exists() or Path('../data-smooth').exists():
            config_dict['smoothed_data'] = True
            if Path('../data_smooth').exists():
                config_dict['dir_smooth_data'] = 'data_smooth'
            else:
                config_dict['dir_smooth_data'] = 'data-smooth'
            print(f"Auto-detected: Smoothed data in {config_dict['dir_smooth_data']}/")
        else:
            config_dict['smoothed_data'] = False

        # Auto-detect beta diagnostics directory
        if Path('../hydro-int-hdf5').exists():
            config_dict['dir_beta_hdf5'] = 'hydro-int-hdf5'
            print("Auto-detected: Beta diagnostics in hydro-int-hdf5/")
        elif Path('../hydro_int_hdf5').exists():
            config_dict['dir_beta_hdf5'] = 'hydro_int_hdf5'
            print("Auto-detected: Beta diagnostics in hydro_int_hdf5/")

        # Auto-detect turbulence mixing
        if Path('../hydro_hdf5').exists():
            # Check for turbulence mixing species
            hydro_dirs = list(Path('../hydro_hdf5').glob('T.*'))
            if hydro_dirs:
                sample_files = list(hydro_dirs[0].glob('hydro_*_*.h5'))
                species_names = [f.stem.split('_')[1] for f in sample_files]
                if any('Top' in s or 'Bot' in s for s in species_names):
                    config_dict['turbulence_mixing'] = True
                    print("Auto-detected: Turbulence mixing simulation")

        return cls(**config_dict)


def load_config(config_file: Optional[str] = None, auto_detect: bool = True) -> Config:
    """Load configuration with fallback chain.

    Priority order:
    1. Specified config file (if provided)
    2. config.yaml in current directory
    3. config_example.yaml
    4. Auto-detection (if enabled)
    5. Default values

    Args:
        config_file: Optional path to config file
        auto_detect: Whether to auto-detect settings from file system

    Returns:
        Config instance
    """
    # Try specified config file first
    if config_file and Path(config_file).exists():
        print(f"Loading configuration from: {config_file}")
        return Config.from_yaml(config_file)

    # Try config.yaml
    if Path('config.yaml').exists():
        print("Loading configuration from: config.yaml")
        return Config.from_yaml('config.yaml')

    # Try config_example.yaml
    if Path('config_example.yaml').exists():
        print("Loading configuration from: config_example.yaml")
        return Config.from_yaml('config_example.yaml')

    # Try auto-detection
    if auto_detect:
        print("Auto-detecting configuration...")
        return Config.auto_detect()

    # Use defaults
    print("Using default configuration")
    return Config()


# Global configuration instance - will be initialized in main()
config: Optional[Config] = None


class TracerData:
    """Handle tracer particle data reading and processing."""

    def __init__(self, tracer_filepath: str = "", tracer_filename: str = ""):
        """Initialize tracer data handler.

        Args:
            tracer_filepath: Path to tracer data directory
            tracer_filename: Name of tracer data file
        """
        self.tracer_filepath = tracer_filepath
        self.tracer_filename = tracer_filename
        self.file_to_read = ""
        self.ntracers = 0
        self.tracer_tags: List[str] = []
        self.tracer_data: Dict[str, np.ndarray] = {}
        self.tracer_adjusted: Dict[str, np.ndarray] = {}
        self.gamma: Optional[np.ndarray] = None
        self.kene: Optional[np.ndarray] = None
        self.segments_wo_boundary: Optional[np.ndarray] = None

    def get_tracer_numbers(self) -> None:
        """Get the number of tracers in the tracer file."""
        self.file_to_read = self.tracer_filepath + self.tracer_filename
        with h5py.File(self.file_to_read, "r") as fh:
            self.ntracers = len(fh)

    def get_tracer_tags(self) -> None:
        """Get the tags of the tracers."""
        self.tracer_tags = []
        with h5py.File(self.file_to_read, "r") as fh:
            self.tracer_tags = list(fh.keys())

    def read_tracer_data(self, tracer_tag: str, dt_tracer: float = 1.0) -> None:
        """Read tracer data from file.

        Args:
            tracer_tag: Tag identifying the tracer
            dt_tracer: Time step for tracer data
        """
        self.tracer_data = {}
        with h5py.File(self.file_to_read, "r") as fh:
            grp = fh[tracer_tag]
            for dset in grp:
                self.tracer_data[dset] = grp[dset][:]
        nsteps = len(self.tracer_data["dX"])
        self.tracer_data["t"] = np.arange(nsteps) * dt_tracer

    def calc_tracer_energy(self) -> None:
        """Calculate tracer kinetic energy (gamma - 1)."""
        self.gamma = np.sqrt(
            self.tracer_data['Ux']**2 +
            self.tracer_data['Uy']**2 +
            self.tracer_data['Uz']**2 + 1
        )
        self.kene = self.gamma - 1

    def find_crossings(self, pos: np.ndarray, length: float) -> Tuple[List[int], List[float]]:
        """Find the crossings of periodic boundaries.

        Args:
            pos: Position array along one axis
            length: Box size along that axis

        Returns:
            Tuple of (crossing indices, position offsets)
        """
        crossings = []
        offsets = []
        offset = 0.0
        nt = pos.shape[0]

        for i in range(nt - 1):
            if pos[i] - pos[i + 1] > 0.1 * length:
                crossings.append(i)
                offset += length
                offsets.append(offset)
            elif pos[i] - pos[i + 1] < -0.1 * length:
                crossings.append(i)
                offset -= length
                offsets.append(offset)

        return crossings, offsets

    def adjust_pos(self, pos: np.ndarray, length: float) -> np.ndarray:
        """Adjust position for periodic boundary conditions.

        Args:
            pos: Position array along one axis
            length: Box size along that axis

        Returns:
            Adjusted position array
        """
        crossings, offsets = self.find_crossings(pos, length)
        pos_b = np.copy(pos)
        nc = len(crossings)

        if nc > 0:
            crossings_arr = np.asarray(crossings)
            offsets_arr = np.asarray(offsets)
            for i in range(nc - 1):
                pos_b[crossings_arr[i] + 1:crossings_arr[i + 1] + 1] += offsets_arr[i]
            pos_b[crossings_arr[nc - 1] + 1:] += offsets_arr[nc - 1]

        return pos_b

    def adjust_tracer_pos(self, vpic_info: Dict[str, float]) -> None:
        """Adjust tracer positions for periodic boundaries.

        Args:
            vpic_info: Dictionary with VPIC simulation parameters
        """
        self.tracer_adjusted = {
            "dX": self.adjust_pos(self.tracer_data['dX'], config.vpic_info["Lx/de"]),
            "dY": self.adjust_pos(self.tracer_data['dY'], config.vpic_info["Ly/de"]),
            "dZ": self.adjust_pos(self.tracer_data['dZ'], config.vpic_info["Lz/de"])
        }

    def exclude_boundary_segments(
        self, vpic_info: Dict[str, float], dim_h: str = "x", dim_v: str = "z"
    ) -> None:
        """Exclude trajectory segments that cross boundaries.

        Args:
            vpic_info: Dictionary with VPIC simulation parameters
            dim_h: Horizontal dimension ('x', 'y', or 'z')
            dim_v: Vertical dimension ('x', 'y', or 'z')
        """
        hdata = self.tracer_data["d" + dim_h.upper()]
        vdata = self.tracer_data["d" + dim_v.upper()]
        lh_de = config.vpic_info["L" + dim_h + "/de"]
        lv_de = config.vpic_info["L" + dim_v + "/de"]

        crossings_h, _ = self.find_crossings(hdata, lh_de)
        crossings_v, _ = self.find_crossings(vdata, lv_de)
        crossings = np.unique(np.concatenate((crossings_h, crossings_v)))
        crossings = crossings.astype(int)

        points = np.array([hdata, vdata]).T.reshape((-1, 1, 2))
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        self.segments_wo_boundary = np.delete(segments, crossings, 0)


class MplCanvas(FigureCanvasQTAgg):
    """Custom matplotlib canvas for Qt integration."""

    def __init__(
        self,
        parent=None,
        width: int = 5,
        height: int = 4,
        dpi: int = 100,
        axes_h: str = "X",
        plot_type: str = "Contour"
    ):
        """Initialize the matplotlib canvas.

        Args:
            parent: Parent widget
            width: Figure width in inches
            height: Figure height in inches
            dpi: Dots per inch
            axes_h: Horizontal axis label
            plot_type: Type of plot ('Contour', 'Contour+X-Average', etc.)
        """
        self.fig = Figure(figsize=(width, height), constrained_layout=True, dpi=dpi)
        self.ax_main = None
        self.ax_cbar = None
        self.ax1d = None
        self.ax_tracer = None
        self.create_axes(axes_h, plot_type)
        super().__init__(self.fig)

    def update_axes(self, axes_h: str = "X", plot_type: str = "Contour") -> None:
        """Update axes configuration.

        Args:
            axes_h: Horizontal axis label
            plot_type: Type of plot
        """
        self.create_axes(axes_h, plot_type)

    def create_axes(self, axes_h: str = "X", plot_type: str = "Contour") -> None:
        """Create axes based on plot type.

        Args:
            axes_h: Horizontal axis label
            plot_type: Type of plot to create axes for
        """
        if plot_type == "Contour":
            widths = [4.8, 0.2]
            spec = self.fig.add_gridspec(nrows=1, ncols=2, width_ratios=widths)
            self.ax_main = self.fig.add_subplot(spec[0, 0])
            self.ax_cbar = self.fig.add_subplot(spec[0, 1])
        elif plot_type in [f"Contour+{axes_h}-Average", f"Contour+{axes_h}-Slice"]:
            widths = [0.2, 4.8, 1.2]
            spec = self.fig.add_gridspec(
                nrows=1, ncols=3, width_ratios=widths, wspace=0.0, hspace=0.0
            )
            self.ax_main = self.fig.add_subplot(spec[0, 1])
            self.ax1d = self.fig.add_subplot(spec[0, 2], sharey=self.ax_main)
            self.ax_cbar = self.fig.add_subplot(spec[0, 0])
        else:
            widths = [4.8, 0.2]
            heights = [4.8, 1.2]
            spec = self.fig.add_gridspec(
                nrows=2, ncols=2, width_ratios=widths, height_ratios=heights,
                wspace=0.0, hspace=0.0
            )
            self.ax_main = self.fig.add_subplot(spec[0, 0])
            self.ax_main.tick_params(axis='x', labelbottom=False)
            self.ax1d = self.fig.add_subplot(spec[1, 0], sharex=self.ax_main)
            self.ax_cbar = self.fig.add_subplot(spec[0, 1])

    def create_tracer_axes(self) -> None:
        """Create an inset axes for tracer energy plot."""
        pos = np.asarray(self.ax_main.get_position()).flatten()
        box_w = pos[2] - pos[0]
        box_h = pos[3] - pos[1]
        box_w1 = box_w * 0.4
        box_h1 = box_h * 0.4
        left = pos[0] + 0.05 * box_w
        bottom = pos[3] - 0.05 * box_h - box_h1
        rect = [left, bottom, box_w1, box_h1]
        self.ax_tracer = self.fig.add_axes(rect)
        self.ax_tracer.patch.set_alpha(0.6)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        # simulation domain
        self.get_domain()

        # check if the simulation is 2D
        self.coords = ["x", "y", "z"]
        self.norms_gda = {"z": 0, "y": 1, "x": 2}
        # Set HDF5 axis mapping based on data order configuration
        if config.hdf5_data_order == "zyx":
            # Newer VPIC: data stored as (nz, ny, nx)
            self.norms_hdf5 = {"z": 0, "y": 1, "x": 2}
        else:
            # Older VPIC: data stored as (nx, ny, nz)
            self.norms_hdf5 = {"x": 0, "y": 1, "z": 2}
        self.normal = "y"  # normal direction
        self.is_2d = False  # whether is a 2D simulation
        for c in self.coords:
            if self.vpic_domain["n" + c] == 1:
                self.normal = c
                self.is_2d = True
        self.hv = [c for c in self.coords if c != self.normal]  # in-plane

        # the file type (HDF5 or gda)
        if config.hdf5_fields:
            self.filetype_comboBox.setCurrentText("HDF5")
        else:
            self.filetype_comboBox.setCurrentText("gda")
            if config.smoothed_data:
                self.gda_path = "../" + config.dir_smooth_data + "/"
            else:
                self.gda_path = "../data/"
            self.tframe_loaded = -1
            self.var_loaded = "random_var"

        # whether to automatically update the plot
        self.autoplot_checkBox.setChecked(False)
        self.auto_update = False
        self.autoplot_checkBox.stateChanged.connect(
            self.autoplot_checkBox_change)

        # whether to integrate along normal direction
        self.integrate_checkBox.setChecked(False)
        self.integrate_normal = False
        self.integrate_checkBox.stateChanged.connect(
            self.integrate_checkBox_change)

        # plot type
        self.plottype_comboBox.currentTextChanged.connect(
            self.plottype_comboBox_change)

        # Raw plot variables
        self.raw_plot_variables()
        self.rawplot_comboBox.currentTextChanged.connect(
            self.rawplot_comboBox_vchange)

        # Diagnostics plot variables
        self.diag_plot = False  # whether to plot diagnostics
        self.diag_var_name = ""
        self.diag_plot_variables()
        self.diagplot_comboBox.currentTextChanged.connect(
            self.diagplot_comboBox_vchange)

        # Overplot variables
        self.overplot_comboBox.setDisabled(True)
        self.over_plot = False  # whether to overplot another variable
        self.over_var_name = ""
        self.over_plot_variables()
        self.tracer_plot = None
        self.field_line_plot = None
        self.overplot_comboBox.currentTextChanged.connect(
            self.overplot_comboBox_vchange)

        # Create toolbar and canvas
        self.margin = 30
        self.top = 320
        self.middle = self.width() // 2
        self.width_max = self.width() - 2 * self.margin
        self.height_max = self.height() - 2 * self.margin - self.top
        self.plot_vLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.plot_vLayoutWidget.setGeometry(
            QtCore.QRect(self.margin, self.top, self.width_max,
                         self.height_max))
        self.plot_vLayoutWidget.setObjectName("plot_vLayoutWidget")
        self.plot_verticalLayout = QtWidgets.QVBoxLayout(
            self.plot_vLayoutWidget)
        self.plot_verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.plot_verticalLayout.setObjectName("plot_verticalLayout")
        self.canvas = MplCanvas(self,
                                width=8,
                                height=8,
                                dpi=100,
                                axes_h="X",
                                plot_type=self.plottype_comboBox.currentText())
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.plot_verticalLayout.addWidget(self.toolbar)
        self.plot_verticalLayout.addWidget(self.canvas)

        # Time slice
        self.tframe_hSlider.setMinimum(config.tmin)
        self.tframe_hSlider.setMaximum(config.tmax)
        self.tframe_SpinBox.setMinimum(self.tframe_hSlider.minimum())
        self.tframe_SpinBox.setMaximum(self.tframe_hSlider.maximum())
        self.tframe_hSlider.valueChanged.connect(self.tframe_hSlider_vchange)
        self.tframe_hSlider.sliderPressed.connect(
            self.tframe_hSlider_disconnect)
        self.tframe_hSlider.sliderReleased.connect(
            self.tframe_hSlider_reconnect)
        self.tframe_SpinBox.setKeyboardTracking(False)
        self.tframe_SpinBox.valueChanged.connect(self.tframe_SpinBox_vchange)
        self.var_name = self.rawplot_comboBox.currentText()
        self.plot_type = self.plottype_comboBox.currentText()
        self.tframe = self.tframe_hSlider.value()
        if "fields_interval" not in config.vpic_info:
            config.config.vpic_info["fields_interval"], _ = QtWidgets.QInputDialog.getInt(
                self, "Get fields interval", "Fields interval:", 100, 0,
                10000000, 1)
        self.tindex = self.tframe * int(config.vpic_info["fields_interval"])

        # x-range
        self.xmin_hScrollBar.valueChanged.connect(self.xmin_hScrollBar_vchange)
        self.xmin_SpinBox.valueChanged.connect(self.xmin_SpinBox_vchange)
        self.xmax_hScrollBar.valueChanged.connect(self.xmax_hScrollBar_vchange)
        self.xmax_SpinBox.valueChanged.connect(self.xmax_SpinBox_vchange)
        self.xmin_hScrollBar.setMinimum(int(self.vpic_domain["xmin"]))
        self.xmin_hScrollBar.setMaximum(int(self.vpic_domain["xmax"]))
        self.xmin_SpinBox.setMinimum(self.vpic_domain["xmin"])
        self.xmin_SpinBox.setMaximum(self.vpic_domain["xmax"])
        self.xmax_hScrollBar.setMinimum(self.xmin_hScrollBar.minimum())
        self.xmax_hScrollBar.setMaximum(self.xmin_hScrollBar.maximum())
        self.xmax_SpinBox.setMinimum(self.xmin_SpinBox.minimum())
        self.xmax_SpinBox.setMaximum(self.xmin_SpinBox.maximum())
        self.xmin_hScrollBar.setSliderPosition(self.xmin_hScrollBar.minimum())
        self.xmax_hScrollBar.setSliderPosition(self.xmax_hScrollBar.maximum())

        # y-range
        self.ymin_hScrollBar.valueChanged.connect(self.ymin_hScrollBar_vchange)
        self.ymin_SpinBox.valueChanged.connect(self.ymin_SpinBox_vchange)
        self.ymax_hScrollBar.valueChanged.connect(self.ymax_hScrollBar_vchange)
        self.ymax_SpinBox.valueChanged.connect(self.ymax_SpinBox_vchange)
        self.ymin_hScrollBar.setMinimum(int(self.vpic_domain["ymin"]))
        self.ymin_hScrollBar.setMaximum(int(self.vpic_domain["ymax"]))
        self.ymin_SpinBox.setMinimum(self.vpic_domain["ymin"])
        self.ymin_SpinBox.setMaximum(self.vpic_domain["ymax"])
        self.ymax_hScrollBar.setMinimum(self.ymin_hScrollBar.minimum())
        self.ymax_hScrollBar.setMaximum(self.ymin_hScrollBar.maximum())
        self.ymax_SpinBox.setMinimum(self.ymin_SpinBox.minimum())
        self.ymax_SpinBox.setMaximum(self.ymin_SpinBox.maximum())
        self.ymin_hScrollBar.setSliderPosition(self.ymin_hScrollBar.minimum())
        self.ymax_hScrollBar.setSliderPosition(self.ymax_hScrollBar.maximum())

        # z-range
        self.zmin_hScrollBar.valueChanged.connect(self.zmin_hScrollBar_vchange)
        self.zmin_SpinBox.valueChanged.connect(self.zmin_SpinBox_vchange)
        self.zmax_hScrollBar.valueChanged.connect(self.zmax_hScrollBar_vchange)
        self.zmax_SpinBox.valueChanged.connect(self.zmax_SpinBox_vchange)
        self.zmin_hScrollBar.setMinimum(int(self.vpic_domain["zmin"]))
        self.zmin_hScrollBar.setMaximum(int(self.vpic_domain["zmax"]))
        self.zmin_SpinBox.setMinimum(self.vpic_domain["zmin"])
        self.zmin_SpinBox.setMaximum(self.vpic_domain["zmax"])
        self.zmax_hScrollBar.setMinimum(self.zmin_hScrollBar.minimum())
        self.zmax_hScrollBar.setMaximum(self.zmin_hScrollBar.maximum())
        self.zmax_SpinBox.setMinimum(self.zmin_SpinBox.minimum())
        self.zmax_SpinBox.setMaximum(self.zmin_SpinBox.maximum())
        self.zmin_hScrollBar.setSliderPosition(self.zmin_hScrollBar.minimum())
        self.zmax_hScrollBar.setSliderPosition(self.zmax_hScrollBar.maximum())

        # create a dictionary for the ranges
        self.range_dist = {
            "xmin_bar": self.xmin_hScrollBar,
            "xmin_box": self.xmin_SpinBox,
            "ymin_bar": self.ymin_hScrollBar,
            "ymin_box": self.ymin_SpinBox,
            "zmin_bar": self.zmin_hScrollBar,
            "zmin_box": self.zmin_SpinBox,
            "xmax_bar": self.xmax_hScrollBar,
            "xmax_box": self.xmax_SpinBox,
            "ymax_bar": self.ymax_hScrollBar,
            "ymax_box": self.ymax_SpinBox,
            "zmax_bar": self.zmax_hScrollBar,
            "zmax_box": self.zmax_SpinBox
        }

        # 1D slices
        self.xslice_hScrollBar.valueChanged.connect(
            self.xslice_hScrollBar_vchange)
        self.xslice_SpinBox.valueChanged.connect(self.xslice_SpinBox_vchange)
        self.yslice_hScrollBar.valueChanged.connect(
            self.yslice_hScrollBar_vchange)
        self.yslice_SpinBox.valueChanged.connect(self.yslice_SpinBox_vchange)
        self.zslice_hScrollBar.valueChanged.connect(
            self.zslice_hScrollBar_vchange)
        self.zslice_SpinBox.valueChanged.connect(self.zslice_SpinBox_vchange)
        self.xslice_hScrollBar.setMinimum(int(self.vpic_domain["xmin"]))
        self.xslice_hScrollBar.setMaximum(int(self.vpic_domain["xmax"]))
        self.xslice_SpinBox.setMinimum(self.vpic_domain["xmin"])
        self.xslice_SpinBox.setMaximum(self.vpic_domain["xmax"])
        xmid = 0.5 * (self.vpic_domain["xmin"] + self.vpic_domain["xmax"])
        self.xslice_hScrollBar.setSliderPosition(int(xmid))
        self.yslice_hScrollBar.setMinimum(int(self.vpic_domain["ymin"]))
        self.yslice_hScrollBar.setMaximum(int(self.vpic_domain["ymax"]))
        self.yslice_SpinBox.setMinimum(self.vpic_domain["ymin"])
        self.yslice_SpinBox.setMaximum(self.vpic_domain["ymax"])
        ymid = 0.5 * (self.vpic_domain["ymin"] + self.vpic_domain["ymax"])
        self.yslice_hScrollBar.setSliderPosition(int(ymid))
        self.zslice_hScrollBar.setMinimum(int(self.vpic_domain["zmin"]))
        self.zslice_hScrollBar.setMaximum(int(self.vpic_domain["zmax"]))
        self.zslice_SpinBox.setMinimum(self.vpic_domain["zmin"])
        self.zslice_SpinBox.setMaximum(self.vpic_domain["zmax"])
        zmid = 0.5 * (self.vpic_domain["zmin"] + self.vpic_domain["zmax"])
        self.zslice_hScrollBar.setSliderPosition(int(zmid))

        # create a dictionary for the slices
        self.slice_dist = {
            "xslice_bar": self.xslice_hScrollBar,
            "xslice_box": self.xslice_SpinBox,
            "yslice_bar": self.yslice_hScrollBar,
            "yslice_box": self.yslice_SpinBox,
            "zslice_bar": self.zslice_hScrollBar,
            "zslice_box": self.zslice_SpinBox
        }

        # 2D plane
        if self.is_2d:
            self.plane_comboBox.setDisabled(True)
            self.plane_hScrollBar.setDisabled(True)
            self.plane_SpinBox.setDisabled(True)
        self.plane_comboBox.currentTextChanged.connect(
            self.plane_comboBox_vchange)
        self.plane_hScrollBar.sliderPressed.connect(
            self.plane_hScrollBar_disconnect)
        self.plane_hScrollBar.sliderReleased.connect(
            self.plane_hScrollBar_reconnect)
        self.plane_hScrollBar.valueChanged.connect(
            self.plane_hScrollBar_vchange)
        self.plane_SpinBox.setKeyboardTracking(False)
        self.plane_SpinBox.valueChanged.connect(self.plane_SpinBox_vchange)
        self.set_normal_plane()
        self.set_plane_index()

        # read the field data
        self.read_data(self.var_name, self.tindex)

        # plot button
        self.plotButton.clicked.connect(self.update_plot)

        # checkbox for saving JPEGs during animation
        self.savejpg_checkBox.setChecked(False)
        self.save_jpegs = False
        self.savejpg_checkBox.stateChanged.connect(
            self.savejpg_checkBox_change)

        # animation buttons
        self.start_animateButton.clicked.connect(self.start_animation)
        self.stop_animateButton.clicked.connect(self.stop_animation)
        self.continue_animateButton.clicked.connect(self.continue_animation)
        self.stop_animateButton.setDisabled(True)
        self.continue_animateButton.setDisabled(True)
        self.is_animation = False

        # checkbox for fixing colormap
        self.cmap_checkBox.setChecked(False)
        self.fix_cmap = False
        self.cmap_checkBox.stateChanged.connect(self.cmap_checkBox_change)
        self.cmap_LineEdit.setText("coolwarm")

        # checkbox for fixing the colorbar range
        self.cbar_checkBox.setChecked(False)
        self.fix_cbar_range = False
        self.cbar_checkBox.stateChanged.connect(self.cbar_checkBox_change)

        # tracer particle index
        self.tracer = TracerData(config.tracer_filepath, config.tracer_filename)
        self.tracer_index_SpinBox.setMinimum(0)
        self.tracer_index_SpinBox.setMaximum(100)
        self.tracer_index_SpinBox.setKeyboardTracking(False)
        self.tracer_index_SpinBox.valueChanged.connect(
            self.tracer_index_SpinBox_vchange)
        self.tracer_index = self.tracer_index_SpinBox.value()

    def plottype_comboBox_change(self, value):
        self.plot_type = value
        self.canvas.fig.clf()
        self.canvas.update_axes(self.hv[0].upper(), value)
        self.update_plot()

    def rawplot_comboBox_vchange(self, value):
        if not self.diag_plot:
            self.var_name = self.rawplot_comboBox.currentText()
            self.read_data(self.var_name, self.tindex)
            self.update_plot()
        self.overplot_comboBox.setDisabled(False)

    def diagplot_comboBox_vchange(self, value):
        self.diag_var_name = self.diagplot_comboBox.currentText()
        if self.diag_var_name != "":
            self.diag_plot = True
            self.var_name = self.diag_var_name
            self.read_data(self.var_name, self.tindex)
            self.update_plot()
        else:
            self.diag_plot = False
            self.var_name = self.rawplot_comboBox.currentText()
            self.read_data(self.var_name, self.tindex)
            self.update_plot()
        self.overplot_comboBox.setDisabled(False)

    def overplot_comboBox_vchange(self, value):
        self.over_var_name = self.overplot_comboBox.currentText()
        if not self.over_var_name:
            self.over_plot = False
            if self.field_line_plot:
                self.field_line_plot.lines.remove()
                for art in self.canvas.ax_main.get_children():
                    if not isinstance(art, mpl.patches.FancyArrowPatch):
                        continue
                    art.remove()
                self.field_line_plot = None
                self.canvas.draw()
            if self.tracer_plot:
                self.tracer_plot.remove()
                self.tracer_dot_plot.remove()
                self.tracer_plot = None
                self.canvas.ax_tracer.remove()
                self.canvas.draw()
        elif self.over_var_name == "field line":
            self.over_plot = True
            self.read_magnetic_field(self.tindex)
            self.plot_field_line()
        elif self.over_var_name == "trajectory":
            self.over_plot = True
            self.over_var_name = self.over_var_name
            self.tracer.get_tracer_numbers()
            self.tracer_index_SpinBox.setMaximum(self.tracer.ntracers)
            self.tracer.get_tracer_tags()
            self.tracer_index = self.tracer_index_SpinBox.value()
            self.tracer_tag = self.tracer.tracer_tags[self.tracer_index]
            self.dt_tracer = config.vpic_info["tracer_interval"] * config.vpic_info["dt*wpe"]
            self.tracer.read_tracer_data(self.tracer_tag, self.dt_tracer)
            self.tracer.calc_tracer_energy()
            self.tracer.adjust_tracer_pos(config.vpic_info)
            self.tracer.exclude_boundary_segments(config.vpic_info, self.hv[0], self.hv[1])
            self.canvas.create_tracer_axes()
            self.plot_tracer()
        else:
            self.over_plot = False
            self.var_name = self.rawplot_comboBox.currentText()
            self.read_data(self.var_name, self.tindex)
            self.update_plot()

    def savejpg_checkBox_change(self):
        self.save_jpegs = not self.save_jpegs

    def cmap_checkBox_change(self):
        self.fix_cmap = not self.fix_cmap

    def cbar_checkBox_change(self):
        self.fix_cbar_range = not self.fix_cbar_range
        if self.fix_cbar_range:
            self.cbar_min_LineEdit.setText(str(np.min(self.field_2d)))
            self.cbar_max_LineEdit.setText(str(np.max(self.field_2d)))

    def autoplot_checkBox_change(self):
        self.auto_update = not self.auto_update

    def integrate_checkBox_change(self):
        self.integrate_normal = not self.integrate_normal
        if self.integrate_normal:
            self.plane_hScrollBar.setDisabled(True)
            self.plane_SpinBox.setDisabled(True)
        else:
            if not self.is_2d:
                self.plane_hScrollBar.setDisabled(False)
                self.plane_SpinBox.setDisabled(False)
        self.read_data(self.var_name, self.tindex)
        self.update_plot()

    def tframe_hSlider_vchange(self, value):
        self.tframe_SpinBox.setValue(value)
        self.tframe = value
        self.tindex = self.tframe * int(config.vpic_info["fields_interval"])
        self.read_data(self.var_name, self.tindex)
        if self.field_line_plot:
            self.read_magnetic_field(self.tindex)
        if self.auto_update:
            self.update_plot()

    def tframe_hSlider_disconnect(self):
        self.sender().valueChanged.disconnect()

    def tframe_hSlider_reconnect(self):
        self.sender().valueChanged.connect(self.tframe_hSlider_vchange)
        self.sender().valueChanged.emit(self.sender().value())

    def tframe_SpinBox_vchange(self, value):
        self.tframe_hSlider.setValue(value)

    def xmin_hScrollBar_vchange(self, value):
        self.xmin_SpinBox.setValue(float(value))

    def xmin_SpinBox_vchange(self, value):
        self.xmin_hScrollBar.setValue(int(value))
        if self.auto_update:
            self.update_plot()

    def ymin_hScrollBar_vchange(self, value):
        self.ymin_SpinBox.setValue(float(value))

    def ymin_SpinBox_vchange(self, value):
        self.ymin_hScrollBar.setValue(int(value))
        if self.auto_update:
            self.update_plot()

    def zmin_hScrollBar_vchange(self, value):
        self.zmin_SpinBox.setValue(float(value))

    def zmin_SpinBox_vchange(self, value):
        self.zmin_hScrollBar.setValue(int(value))
        if self.auto_update:
            self.update_plot()

    def xmax_hScrollBar_vchange(self, value):
        self.xmax_SpinBox.setValue(float(value))

    def xmax_SpinBox_vchange(self, value):
        self.xmax_hScrollBar.setValue(int(value))
        if self.auto_update:
            self.update_plot()

    def ymax_hScrollBar_vchange(self, value):
        self.ymax_SpinBox.setValue(float(value))

    def ymax_SpinBox_vchange(self, value):
        self.ymax_hScrollBar.setValue(int(value))
        if self.auto_update:
            self.update_plot()

    def zmax_hScrollBar_vchange(self, value):
        self.zmax_SpinBox.setValue(float(value))

    def zmax_SpinBox_vchange(self, value):
        self.zmax_hScrollBar.setValue(int(value))
        if self.auto_update:
            self.update_plot()

    def xslice_hScrollBar_vchange(self, value):
        self.xslice_SpinBox.setValue(float(value))

    def xslice_SpinBox_vchange(self, value):
        self.xslice_hScrollBar.setValue(int(value))
        if self.plot_type == "Contour+X-Slice":
            if self.auto_update:
                self.update_plot()

    def yslice_hScrollBar_vchange(self, value):
        self.yslice_SpinBox.setValue(float(value))

    def yslice_SpinBox_vchange(self, value):
        self.yslice_hScrollBar.setValue(int(value))
        if self.plot_type == "Contour+Y-Slice":
            if self.auto_update:
                self.update_plot()

    def zslice_hScrollBar_vchange(self, value):
        self.zslice_SpinBox.setValue(float(value))

    def zslice_SpinBox_vchange(self, value):
        self.zslice_hScrollBar.setValue(int(value))
        if self.plot_type == "Contour+Z-Slice":
            if self.auto_update:
                self.update_plot()

    def plane_comboBox_vchange(self, value):
        current_plane = self.plane_comboBox.currentText()
        self.normal = current_plane[0].lower()
        self.set_normal_plane()

    def set_normal_plane(self):
        self.hv = [c for c in self.coords if c != self.normal]  # in-plane
        self.plane_comboBox.setCurrentText(self.normal.upper() + "-Plane")
        self.plane_hScrollBar.setMinimum(
            int(self.vpic_domain[self.normal + "min"]))
        self.plane_hScrollBar.setMaximum(
            int(self.vpic_domain[self.normal + "max"]))
        old_plane = self.plane_hScrollBar.value()
        self.plane_SpinBox.setMinimum(self.vpic_domain[self.normal + "min"])
        self.plane_SpinBox.setMaximum(self.vpic_domain[self.normal + "max"])
        mid = 0.5 * (self.vpic_domain[self.normal + "min"] +
                     self.vpic_domain[self.normal + "max"])
        self.plane_hScrollBar.setSliderPosition(int(mid))
        if int(mid) == old_plane:  # force update
            self.set_plane_index()
            self.read_data(self.var_name, self.tindex)
            if self.field_line_plot:
                self.read_magnetic_field(self.tindex)
            if self.auto_update:
                self.update_plot()
        self.plottype_comboBox.setItemText(
            1, "Contour+" + self.hv[0].upper() + "-Average")
        self.plottype_comboBox.setItemText(
            2, "Contour+" + self.hv[0].upper() + "-Slice")
        self.plottype_comboBox.setItemText(
            3, "Contour+" + self.hv[1].upper() + "-Slice")
        self.integrate_checkBox.setText("Integrate along " + self.normal.upper())

    def set_plane_index(self):
        plane_coord = self.plane_SpinBox.value()
        cmin = self.vpic_domain[self.normal + "min"]
        self.plane_index = int(
            (plane_coord - cmin) / self.vpic_domain["d" + self.normal])
        if self.plane_index == self.vpic_domain["n" + self.normal]:
            self.plane_index = self.vpic_domain["n" + self.normal] - 1

    def plane_hScrollBar_disconnect(self):
        self.sender().valueChanged.disconnect()

    def plane_hScrollBar_reconnect(self):
        self.sender().valueChanged.connect(self.plane_hScrollBar_vchange)
        self.sender().valueChanged.emit(self.sender().value())

    def plane_hScrollBar_vchange(self, value):
        self.plane_SpinBox.setValue(float(value))

    def plane_SpinBox_vchange(self, value):
        self.plane_hScrollBar.setValue(int(value))
        self.set_plane_index()
        self.read_data(self.var_name, self.tindex)
        if self.field_line_plot:
            self.read_magnetic_field(self.tindex)
        if self.auto_update:
            self.update_plot()

    def raw_plot_variables(self):
        if config.hdf5_fields:
            # Read fields from HDF5 file dynamically
            if config.smoothed_data:
                fdir = "../" + config.dir_smooth_data + "/"
            else:
                if config.time_averaged_field:
                    fdir = "../fields-avg-hdf5/T.0/"
                else:
                    fdir = "../" + config.dir_fields_hdf5 + "/T.0/"
            fname = fdir + "fields_0.h5"

            try:
                with h5py.File(fname, "r") as fh:
                    group = fh["Timestep_0"]
                    self.fields_list = list(group.keys())
                self.fields_list.append("absb")  # Computed field
            except Exception as e:
                print(f"Warning: Could not read fields file {fname}: {e}")
                print("Using default field list")
                self.fields_list = ["cbx", "cby", "cbz", "absb", "ex", "ey", "ez"]

            # Detect species and read hydro data
            if config.smoothed_data:
                hydro_dir = "../" + config.dir_smooth_data + "/"
            else:
                if config.time_averaged_field:
                    hydro_dir = "../hydro-avg-hdf5/T.0/"
                else:
                    hydro_dir = "../hydro_hdf5/T.0/"

            self.hydro_list = []
            self.ehydro_list = []  # Electron hydro variables

            if config.auto_detect_species:
                # Auto-detect species from hydro files
                detected_species = []
                if os.path.exists(hydro_dir):
                    try:
                        files = os.listdir(hydro_dir)
                        # Look for hydro_<species>_0.h5 files
                        for fname in files:
                            if fname.startswith("hydro_") and fname.endswith("_0.h5"):
                                # Extract species name
                                species = fname.replace("hydro_", "").replace("_0.h5", "")
                                detected_species.append(species)
                        detected_species = sorted(set(detected_species))
                        if detected_species:
                            print(f"Auto-detected species: {', '.join(detected_species)}")
                    except Exception as e:
                        print(f"Warning: Could not auto-detect species from {hydro_dir}: {e}")

                if not detected_species:
                    print("Warning: No species detected, using default: electron, H")
                    detected_species = ["electron", "H"]
            else:
                # Use manually specified species list
                detected_species = config.species_list if config.species_list else ["electron", "H"]
                print(f"Using configured species: {', '.join(detected_species)}")

            # Store species list for later use (e.g., diagnostic plots)
            self.species_list = detected_species

            # Read variables from each species' hydro file
            for species in detected_species:
                fname = hydro_dir + f"hydro_{species}_0.h5"
                try:
                    with h5py.File(fname, "r") as fh:
                        group = fh["Timestep_0"]
                        for var in group.keys():
                            var_name = species + "-" + var
                            self.hydro_list.append(var_name)
                            # Track electron hydro variables separately
                            if species.lower() == "electron":
                                self.ehydro_list.append(var_name)
                except Exception as e:
                    print(f"Warning: Could not read hydro file for species '{species}': {e}")
                    print(f"Skipping species '{species}'")

            self.var_list = self.fields_list + self.hydro_list
            self.var_list = sorted(set(self.var_list))
        else:
            # For GDA format, use default species list
            self.species_list = ["electron", "ion"]
            flist = [
                _ for _ in os.listdir(self.gda_path) if _.endswith(".gda")
            ]
            if len([name for name in flist if name[:2] == "bx"]) == 1:
                # all frames are save in the same file
                self.var_list = sorted([f[:-4] for f in flist])
                self.single_gda = True  # all time frames are in the same file
            else:
                var_list = []
                for f in flist:
                    var_list.append(f.split("_")[0])
                self.var_list = sorted(set(var_list))
                self.single_gda = False
        _translate = QtCore.QCoreApplication.translate
        for ivar, var in enumerate(self.var_list):
            self.rawplot_comboBox.addItem("")
            self.rawplot_comboBox.setItemText(ivar,
                                              _translate("MainWindow", var))

    def diag_plot_variables(self):
        self.diag_var_list = [""]
        # Add per-species j.E diagnostics
        if hasattr(self, 'species_list'):
            for species in self.species_list:
                self.diag_var_list.append(f"{species}-j.E")
        self.diag_var_list.append("beta")
        if config.turbulence_mixing:
            self.diag_var_list.append("emix")
        _translate = QtCore.QCoreApplication.translate
        for ivar, var in enumerate(self.diag_var_list):
            self.diagplot_comboBox.addItem("")
            self.diagplot_comboBox.setItemText(ivar,
                                               _translate("MainWindow", var))

    def over_plot_variables(self):
        self.over_var_list = ["", "field line", "trajectory"]
        _translate = QtCore.QCoreApplication.translate
        for ivar, var in enumerate(self.over_var_list):
            self.overplot_comboBox.addItem("")
            self.overplot_comboBox.setItemText(ivar,
                                               _translate("MainWindow", var))

    def tracer_index_SpinBox_vchange(self, value):
        self.tracer_index = value
        self.tracer_tag = self.tracer.tracer_tags[self.tracer_index]
        self.tracer.read_tracer_data(self.tracer_tag, self.dt_tracer)
        self.tracer.calc_tracer_energy()
        self.tracer.adjust_tracer_pos(config.vpic_info)
        self.tracer.exclude_boundary_segments(config.vpic_info, self.hv[0], self.hv[1])
        self.plot_tracer()

    def get_domain(self):
        """Get VPIC simulation domain
        """
        self.vpic_domain = {}
        self.vpic_domain["xmin"] = 0.0
        self.vpic_domain["xmax"] = config.vpic_info["Lx/de"]
        self.vpic_domain["ymin"] = -0.5 * config.vpic_info["Ly/de"]
        self.vpic_domain["ymax"] = 0.5 * config.vpic_info["Ly/de"]
        self.vpic_domain["zmin"] = -0.5 * config.vpic_info["Lz/de"]
        self.vpic_domain["zmax"] = 0.5 * config.vpic_info["Lz/de"]
        self.vpic_domain["nx"] = int(config.vpic_info["nx"]) // config.smooth_factor
        self.vpic_domain["ny"] = int(config.vpic_info["ny"]) // config.smooth_factor
        self.vpic_domain["nz"] = int(config.vpic_info["nz"]) // config.smooth_factor
        self.vpic_domain["dx"] = config.vpic_info["dx/de"] * config.smooth_factor
        self.vpic_domain["dy"] = config.vpic_info["dy/de"] * config.smooth_factor
        self.vpic_domain["dz"] = config.vpic_info["dz/de"] * config.smooth_factor
        for i in ["x", "y", "z"]:
            n = "n" + i
            d = "d" + i
            if config.vpic_info[n] < config.smooth_factor:
                self.vpic_domain[n] = 1
                self.vpic_domain[d] = config.vpic_info[d + "/de"] * config.vpic_info[n]
        hdx = 0.5 * self.vpic_domain["dx"]
        hdy = 0.5 * self.vpic_domain["dy"]
        hdz = 0.5 * self.vpic_domain["dz"]
        self.vpic_domain["xgrid"] = np.linspace(self.vpic_domain["xmin"] + hdx,
                                                self.vpic_domain["xmax"] - hdx,
                                                self.vpic_domain["nx"])
        self.vpic_domain["ygrid"] = np.linspace(self.vpic_domain["ymin"] + hdy,
                                                self.vpic_domain["ymax"] - hdy,
                                                self.vpic_domain["ny"])
        self.vpic_domain["zgrid"] = np.linspace(self.vpic_domain["zmin"] + hdz,
                                                self.vpic_domain["zmax"] - hdz,
                                                self.vpic_domain["nz"])

    def get_length_scale_label(self):
        """Get the appropriate length scale label for plot axes.

        Returns:
            str: Length scale label ('d_e', 'd_i', or empty string)
        """
        scale = config.vpic_info.get('length_scale', 'de')
        if scale == 'de':
            return 'd_e'
        elif scale == 'di':
            return 'd_i'
        else:
            return ''  # No scale suffix for plain length units

    def read_gda_file(self, vname, tindex):
        """read fields or hydro in gda format

        Args:
            vname (string): variable name
            tindex (int): time index
        """
        if self.is_2d:
            nh = self.vpic_domain["n" + self.hv[0]]
            nv = self.vpic_domain["n" + self.hv[1]]
            if self.single_gda:
                fname = self.gda_path + vname + ".gda"
                fdata = np.fromfile(fname,
                                    dtype=np.float32,
                                    count=nh * nv,
                                    offset=nh * nv * self.tframe * 4)
            else:
                fname = self.gda_path + vname + "_" + str(self.tindex) + ".gda"
                fdata = np.fromfile(fname, dtype=np.float32, count=-1)
            field_2d = fdata.reshape([nv, nh]).T
            return field_2d, field_2d
        else:
            nx = self.vpic_domain["nx"]
            ny = self.vpic_domain["ny"]
            nz = self.vpic_domain["nz"]
            ntot = nx * ny * nz
            # Since slicing of gda file is incontinent, we read all the 3D
            # data cube and take slices in the memory.
            if self.tframe_loaded != self.tframe or self.var_loaded != vname:
                if self.single_gda:
                    fname = self.gda_path + vname + ".gda"
                    field_3d = np.fromfile(fname,
                                           dtype=np.float32,
                                           count=ntot,
                                           offset=ntot * self.tframe *
                                           4).reshape([nz, ny, nx])
                else:
                    fname = self.gda_path + vname + "_" + str(
                        self.tindex) + ".gda"
                    field_3d = np.fromfile(fname, dtype=np.float32,
                                           count=-1).reshape([nz, ny, nx])
            else:
                field_3d = self.field_3d
            self.tframe_loaded = self.tframe
            self.var_loaded = vname
            if self.normal == 'x':
                field_2d = field_3d[:, :, self.plane_index].T
            elif self.normal == 'y':
                field_2d = field_3d[:, self.plane_index, :].T
            else:
                field_2d = field_3d[self.plane_index, :, :].T
            return field_2d, field_3d

    def read_fields(self, vname, tindex):
        """read electric and magnetic fields in HDF5 format

        Args:
            vname (string): variable name
            tindex (int): time index
        """
        if config.smoothed_data:
            fname = ("../" + config.dir_smooth_data + "/fields_" + str(tindex) + ".h5")
        else:
            if config.time_averaged_field:
                fdir = "../fields-avg-hdf5/T." + str(tindex) + "/"
            else:
                fdir = "../" + config.dir_fields_hdf5 + "/T." + str(tindex) + "/"
            fname = fdir + "fields_" + str(tindex) + ".h5"
        with h5py.File(fname, 'r') as fh:
            group = fh["Timestep_" + str(tindex)]
            if vname == "absb":
                bvec = {}
                if self.integrate_normal:
                    dcell = self.vpic_domain["d" + self.normal]
                    for var in ["cbx", "cby", "cbz"]:
                        dset = group[var]
                        bvec[var] = dset[:, :, :]
                    field_2d = dcell * np.sum(np.sqrt(bvec["cbx"]**2 +
                                                      bvec["cby"]**2 +
                                                      bvec["cbz"]**2),
                                              axis=self.norms_hdf5[self.normal])
                else:
                    # Slice based on data order (norms_hdf5 maps direction to axis)
                    axis_idx = self.norms_hdf5[self.normal]
                    for var in ["cbx", "cby", "cbz"]:
                        dset = group[var]
                        if axis_idx == 0:
                            bvec[var] = dset[self.plane_index, :, :]
                        elif axis_idx == 1:
                            bvec[var] = dset[:, self.plane_index, :]
                        else:  # axis_idx == 2
                            bvec[var] = dset[:, :, self.plane_index]
                    field_2d = np.sqrt(bvec["cbx"]**2 + bvec["cby"]**2 +
                                       bvec["cbz"]**2)
            else:
                dset = group[vname]
                if self.integrate_normal:
                    dcell = self.vpic_domain["d" + self.normal]
                    field_2d = dcell * np.sum(dset[:, :, :],
                                              axis=self.norms_hdf5[self.normal])
                else:
                    # Slice based on data order (norms_hdf5 maps direction to axis)
                    axis_idx = self.norms_hdf5[self.normal]
                    if axis_idx == 0:
                        field_2d = dset[self.plane_index, :, :]
                    elif axis_idx == 1:
                        field_2d = dset[:, self.plane_index, :]
                    else:  # axis_idx == 2
                        field_2d = dset[:, :, self.plane_index]
        return field_2d

    def read_current_density_species(self, vname, tindex, species):
        """Read current density associated with one species

        Args:
            vname (string): variable name
            tindex (int): time index
            species (string): particle species
        """
        # Determine file path
        if config.smoothed_data:
            fname = ("../" + config.dir_smooth_data + "/hydro_" + species + "_" +
                     str(tindex) + ".h5")
        else:
            if config.time_averaged_field:
                fdir = "../hydro-avg-hdf5/T." + str(tindex) + "/"
            else:
                fdir = "../hydro_hdf5/T." + str(tindex) + "/"
            fname = fdir + "hydro_" + species + "_" + str(tindex) + ".h5"

        jsp = {}
        with h5py.File(fname, 'r') as fh:
            group = fh["Timestep_" + str(tindex)]
            # Read specific current density component
            dset = group[vname]
            if self.integrate_normal:
                jsp["jdir"] = dset[:, :, :]
            else:
                # Slice based on data order (norms_hdf5 maps direction to axis)
                axis_idx = self.norms_hdf5[self.normal]
                if axis_idx == 0:
                    jsp["jdir"] = dset[self.plane_index, :, :]
                elif axis_idx == 1:
                    jsp["jdir"] = dset[:, self.plane_index, :]
                else:  # axis_idx == 2
                    jsp["jdir"] = dset[:, :, self.plane_index]
        return jsp

    def read_current_density(self, vname, tindex):
        """read current density

        Args:
            vname (string): variable name
            tindex (int): time index
        """
        # Electron
        if config.turbulence_mixing:
            jsp = self.read_current_density_species(vname, tindex,
                                                    "electronTop")
            jtmp = self.read_current_density_species(vname, tindex,
                                                     "electronBot")
            for var in jsp:
                jsp[var] += jtmp[var]
        else:
            jsp = self.read_current_density_species(vname, tindex, "electron")

        # Ion
        if config.turbulence_mixing:
            jtmp = self.read_current_density_species(vname, tindex, "ionTop")
            for var in jsp:
                jsp[var] += jtmp[var]
            jtmp = self.read_current_density_species(vname, tindex, "ionBot")
            for var in jsp:
                jsp[var] += jtmp[var]
        else:
            jtmp = self.read_current_density_species(vname, tindex, "ion")
            for var in jsp:
                jsp[var] += jtmp[var]

        if self.integrate_normal:
            dcell = self.vpic_domain["d" + self.normal]
            field_2d = np.sum(jsp["jdir"],
                              axis=self.norms_hdf5[self.normal]) * dcell
        else:
            field_2d = jsp["jdir"]

        return field_2d

    def read_hydro_species(self, vname, tindex, species):
        """Read the hydro data of one species

        Args:
            vname (string): variable name
            tindex (int): time index
            species (string): particle species
        """
        if vname in self.ehydro_list:
            if config.smoothed_data:
                fname = ("../" + config.dir_smooth_data + "/hydro_" + species + "_" +
                         str(tindex) + ".h5")
            else:
                if config.time_averaged_field:
                    fdir = "../hydro-avg-hdf5/T." + str(tindex) + "/"
                else:
                    fdir = "../hydro_hdf5/T." + str(tindex) + "/"
                fname = fdir + "hydro_" + species + "_" + str(tindex) + ".h5"
        else:
            if config.smoothed_data:
                fname = ("../" + config.dir_smooth_data + "/hydro_" + species + "_" +
                         str(tindex) + ".h5")
            else:
                if config.time_averaged_field:
                    fdir = "../hydro-avg-hdf5/T." + str(tindex) + "/"
                else:
                    fdir = "../hydro_hdf5/T." + str(tindex) + "/"
                fname = fdir + "hydro_" + species + "_" + str(tindex) + ".h5"

        hydro = {}
        keys = []
        with h5py.File(fname, 'r') as fh:
            group = fh["Timestep_" + str(tindex)]
            # Extract the actual variable name (remove species prefix like "electron-")
            if "-" in vname:
                var_only = vname.split("-", 1)[1]
            else:
                var_only = vname
            
            if var_only[0] == "n":
                keys.append("rho")
            elif var_only[0] == 'v':  # for velocity
                keys.append("rho")
                keys.append("j" + var_only[-1])
            elif var_only[0] == 'u':  # for four-velocity
                keys.append("rho")
                keys.append("p" + var_only[-1])
            else:  # for pressure tensor
                keys.append("rho")
                keys.append("j" + var_only[-2])
                keys.append("p" + var_only[-1])
                vtmp = "t" + var_only[2:]
                if vtmp in group:
                    keys.append(vtmp)
                else:
                    keys.append("t" + var_only[-1] + var_only[-2])
            if self.integrate_normal:
                for key in keys:
                    if key in group:
                        dset = group[key]
                        hydro[key] = dset[:, :, :]
            else:
                # Slice based on data order (norms_hdf5 maps direction to axis)
                axis_idx = self.norms_hdf5[self.normal]
                for key in keys:
                    if key in group:
                        dset = group[key]
                        if axis_idx == 0:
                            hydro[key] = dset[self.plane_index, :, :]
                        elif axis_idx == 1:
                            hydro[key] = dset[:, self.plane_index, :]
                        else:  # axis_idx == 2
                            hydro[key] = dset[:, :, self.plane_index]

        return hydro

    def read_hydro(self, vname, tindex):
        """Read the hydro data of one species

        Args:
            vname (string): variable name in format "species-var" (e.g., "electron-txx")
            tindex (int): time index
        """
        # Split variable name to get species and actual variable
        splits = vname.split("-")
        species = splits[0]
        var = splits[1]

        # Determine file path
        if config.smoothed_data:
            fdir = "../" + config.dir_smooth_data + "/"
        else:
            if config.time_averaged_field:
                fdir = "../hydro-avg-hdf5/T." + str(tindex) + "/"
            else:
                fdir = "../hydro_hdf5/T." + str(tindex) + "/"
        fname = fdir + "hydro_" + species + "_" + str(tindex) + ".h5"

        # Read the variable directly from the file
        with h5py.File(fname, "r") as fh:
            group = fh["Timestep_" + str(tindex)]
            dset = group[var]
            if self.integrate_normal:
                dcell = self.vpic_domain["d" + self.normal]
                field_2d = np.sum(dset[:, :, :],
                                  axis=self.norms_hdf5[self.normal]) * dcell
            else:
                # Slice based on data order (norms_hdf5 maps direction to axis)
                axis_idx = self.norms_hdf5[self.normal]
                if axis_idx == 0:
                    field_2d = dset[self.plane_index, :, :]
                elif axis_idx == 1:
                    field_2d = dset[:, self.plane_index, :]
                else:  # axis_idx == 2
                    field_2d = dset[:, :, self.plane_index]

        return field_2d

    def get_jdote_species(self, tindex, species):
        """Get j.E for a specific species

        Args:
            tindex (int): time index
            species (string): particle species name

        Returns:
            tuple: (field_2d, field_3d) - 2D and 3D j.E data
        """
        if config.hdf5_fields:
            # Read current density components for this species
            jx_sp = self.read_current_density_species("jx", tindex, species)
            jy_sp = self.read_current_density_species("jy", tindex, species)
            jz_sp = self.read_current_density_species("jz", tindex, species)

            # Handle integration along normal direction
            if self.integrate_normal:
                dcell = self.vpic_domain["d" + self.normal]
                jx = np.sum(jx_sp["jdir"], axis=self.norms_hdf5[self.normal]) * dcell
                jy = np.sum(jy_sp["jdir"], axis=self.norms_hdf5[self.normal]) * dcell
                jz = np.sum(jz_sp["jdir"], axis=self.norms_hdf5[self.normal]) * dcell
            else:
                jx = jx_sp["jdir"]
                jy = jy_sp["jdir"]
                jz = jz_sp["jdir"]

            # Read electric field components
            ex = self.read_fields("ex", tindex)
            ey = self.read_fields("ey", tindex)
            ez = self.read_fields("ez", tindex)

            # Calculate j.E
            field_2d = jx * ex + jy * ey + jz * ez
            return field_2d, field_2d
        else:
            # For GDA format, fall back to total current (not per-species)
            j2d, j3d = self.read_gda_file("jx", tindex)
            e2d, e3d = self.read_gda_file("ex", tindex)
            field_2d = j2d * e2d
            field_3d = j2d * e3d
            j2d, j3d = self.read_gda_file("jy", tindex)
            e2d, e3d = self.read_gda_file("ey", tindex)
            field_2d += j2d * e2d
            field_3d += j2d * e3d
            j2d, j3d = self.read_gda_file("jz", tindex)
            e2d, e3d = self.read_gda_file("ez", tindex)
            field_2d += j2d * e2d
            field_3d += j2d * e3d
            if self.integrate_normal:
                dcell = self.vpic_domain["d" + self.normal]
                field_2d = dcell * np.sum(field_3d,
                                          axis=self.norms_gda[self.normal]).T
            return field_2d, field_3d

    def get_beta(self, tindex):
        """Get beta production diagnostics data.

        Beta production is typically stored in hydro-int-hdf5 directory
        for hybridVPIC simulations.

        Args:
            tindex (int): time index

        Returns:
            tuple: (field_2d, field_3d) for 2D and 3D data
        """
        if config.hdf5_fields:
            fdir = "../" + config.dir_beta_hdf5 + "/T." + str(tindex) + "/"
            fname = fdir + "hydro_beta_" + str(tindex) + ".h5"

            try:
                with h5py.File(fname, "r") as fh:
                    group = fh["Timestep_" + str(tindex)]
                    dset = group["beta"]
                    if self.integrate_normal:
                        dcell = self.vpic_domain["d" + self.normal]
                        field_2d = dcell * np.sum(dset[:, :, :],
                                                  axis=self.norms_hdf5[self.normal])
                    else:
                        if config.hdf5_data_order == "zyx":
                            # Newer VPIC format (nz, ny, nx)
                            if self.normal == 'x':
                                field_2d = dset[:, :, self.plane_index]
                            elif self.normal == 'y':
                                field_2d = dset[:, self.plane_index, :]
                            else:
                                field_2d = dset[self.plane_index, :, :]
                        else:
                            # Older VPIC format (nx, ny, nz)
                            if self.normal == 'x':
                                field_2d = dset[self.plane_index, :, :]
                            elif self.normal == 'y':
                                field_2d = dset[:, self.plane_index, :]
                            else:
                                field_2d = dset[:, :, self.plane_index]
                return field_2d, field_2d
            except FileNotFoundError:
                print(f"Warning: Beta diagnostics file not found: {fname}")
                print("Beta diagnostics are typically available in hybridVPIC simulations")
                print("with hydro-int-hdf5 directory containing hydro_beta_*.h5 files")
                # Return zeros with appropriate shape
                if self.is_2d:
                    nh = self.vpic_domain["n" + self.hv[0]]
                    nv = self.vpic_domain["n" + self.hv[1]]
                    field_2d = np.zeros((nh, nv))
                else:
                    nx = self.vpic_domain["nx"]
                    ny = self.vpic_domain["ny"]
                    field_2d = np.zeros((nx, ny))
                return field_2d, field_2d
            except Exception as e:
                print(f"Error reading beta diagnostics: {e}")
                # Return zeros with appropriate shape
                if self.is_2d:
                    nh = self.vpic_domain["n" + self.hv[0]]
                    nv = self.vpic_domain["n" + self.hv[1]]
                    field_2d = np.zeros((nh, nv))
                else:
                    nx = self.vpic_domain["nx"]
                    ny = self.vpic_domain["ny"]
                    field_2d = np.zeros((nx, ny))
                return field_2d, field_2d
        else:
            # GDA format support (for older simulations)
            try:
                field_2d, field_3d = self.read_gda_file("ni-beta3", tindex)
                if self.integrate_normal:
                    dcell = self.vpic_domain["d" + self.normal]
                    field_2d = dcell * np.sum(field_3d,
                                              axis=self.norms_gda[self.normal]).T
                return field_2d, field_3d
            except Exception as e:
                print(f"Error reading beta from GDA files: {e}")
                if self.is_2d:
                    nh = self.vpic_domain["n" + self.hv[0]]
                    nv = self.vpic_domain["n" + self.hv[1]]
                    field_2d = np.zeros((nh, nv))
                else:
                    nx = self.vpic_domain["nx"]
                    ny = self.vpic_domain["ny"]
                    field_2d = np.zeros((nx, ny))
                return field_2d, field_2d

    def electron_mixing_fraction(self, tindex):
        """get the electron mixing fraction

        (ne_bot - ne_top) / (ne_bot + ne_top)
        """
        ne_top = self.read_hydro_species("ne", tindex, "electronTop")
        ne_bot = self.read_hydro_species("ne", tindex, "electronBot")
        field_2d = (ne_bot["rho"] - ne_top["rho"]) / (ne_bot["rho"] +
                                                      ne_top["rho"])
        return field_2d, field_2d

    def read_data(self, vname, tindex):
        """Read data from file

        Args:
            vname (string): variable name
            tindex (int): time index
        """
        if self.diag_plot:
            if "-j.E" in self.diag_var_name:
                # Extract species name from "<species>-j.E"
                species = self.diag_var_name.replace("-j.E", "")
                self.field_2d, self.field_3d = self.get_jdote_species(tindex, species)
            elif self.diag_var_name == "beta":
                self.field_2d, self.field_3d = self.get_beta(tindex)
            elif self.diag_var_name == "emix":
                self.field_2d, self.field_3d = self.electron_mixing_fraction(
                    tindex)
        else:
            if config.hdf5_fields:
                if vname in self.fields_list:  # electric and magnetic fields
                    self.field_2d = self.read_fields(vname, tindex)
                else:  # density, velocity, momentum, pressure tensor
                    self.field_2d = self.read_hydro(vname, tindex)
            else:
                self.field_2d, self.field_3d = self.read_gda_file(
                    vname, tindex)
                if self.integrate_normal:
                    dcell = self.vpic_domain["d" + self.normal]
                    self.field_2d = dcell * np.sum(self.field_3d,
                                                   axis=self.norms_gda[self.normal]).T

    def read_magnetic_field(self, tindex):
        """Read magnetic field from file

        Args:
            tindex (int): time index
        """
        if config.hdf5_fields:
            self.bh_2d = self.read_fields("cb" + self.hv[0], tindex)
            self.bv_2d = self.read_fields("cb" + self.hv[1], tindex)
            # Try to read cb*0 fields if they exist (background/initial fields)
            try:
                self.bh_2d += self.read_fields("cb" + self.hv[0] + "0", tindex)
                self.bv_2d += self.read_fields("cb" + self.hv[1] + "0", tindex)
            except KeyError:
                # cb*0 fields don't exist, skip them
                pass
        else:
            self.bh_2d, self.bh_3d = self.read_gda_file(
                "b" + self.hv[0], self.gda_path, tindex)
            self.bv_2d, self.bv_3d = self.read_gda_file(
                "b" + self.hv[1], self.gda_path, tindex)
            if self.integrate_normal:
                dcell = self.vpic_domain["d" + self.normal]
                self.bh_2d = dcell * np.sum(self.bh_3d,
                                            axis=self.norms_gda[self.normal]).T
                self.bv_2d = dcell * np.sum(self.bv_3d,
                                            axis=self.norms_gda[self.normal]).T

    def plot_field_line(self):
        """Plot magnetic field lines using streamplot"""
        if self.field_line_plot:
            self.field_line_plot.lines.remove()
            for art in self.canvas.ax_main.get_children():
                if not isinstance(art, mpl.patches.FancyArrowPatch):
                    continue
                art.remove()

        # Handle transpose based on data order
        # For "zyx" order: data is already in correct shape (ny, nx) for streamplot
        # For "xyz" order: data needs transpose from (nx, ny) to (ny, nx)
        if config.hdf5_data_order == "zyx":
            bh_plot = self.bh_2d
            bv_plot = self.bv_2d
        else:
            bh_plot = self.bh_2d.T
            bv_plot = self.bv_2d.T

        # Create grids that match the actual data dimensions
        # This handles cases where data has been smoothed or processed
        h_coord = self.hv[0]
        v_coord = self.hv[1]

        # Get the actual data shape (streamplot expects shape (len(Y), len(X)))
        ny_data, nx_data = bh_plot.shape

        # Create coordinate grids matching the actual data dimensions
        h_min = self.vpic_domain[h_coord + "min"]
        h_max = self.vpic_domain[h_coord + "max"]
        v_min = self.vpic_domain[v_coord + "min"]
        v_max = self.vpic_domain[v_coord + "max"]

        xgrid = np.linspace(h_min, h_max, nx_data)
        ygrid = np.linspace(v_min, v_max, ny_data)

        self.field_line_plot = self.canvas.ax_main.streamplot(
            xgrid,
            ygrid,
            bh_plot,
            bv_plot,
            density=1.0,
            linewidth=0.5,
            arrowsize=0.5,
            broken_streamlines=True,
            color='k')
        self.canvas.draw()

    def update_plot(self):
        if self.fix_cbar_range:
            dmin = float(self.cbar_min_LineEdit.text())
            dmax = float(self.cbar_max_LineEdit.text())
        else:
            if np.any(self.field_2d < 0) and np.any(self.field_2d > 0):
                dmax = min(-self.field_2d.min(), self.field_2d.max())
                dmin = -dmax
            else:
                dmin, dmax = self.field_2d.min(), self.field_2d.max()

        if self.fix_cmap:
            cmap = self.cmap_LineEdit.text()
        else:
            if dmin < 0 and dmax > 0:
                cmap = "coolwarm"
            else:
                cmap = "viridis"

        h = self.hv[0]
        v = self.hv[1]
        hmin = self.range_dist[h + "min_box"].value()
        hmax = self.range_dist[h + "max_box"].value()
        vmin = self.range_dist[v + "min_box"].value()
        vmax = self.range_dist[v + "max_box"].value()
        lhp = hmax - hmin
        lvp = vmax - vmin

        hmin_s = hmin - self.vpic_domain[h + "min"]
        hmax_s = hmax - self.vpic_domain[h + "min"]
        vmin_s = vmin - self.vpic_domain[v + "min"]
        vmax_s = vmax - self.vpic_domain[v + "min"]

        # Calculate indices, accounting for actual field dimensions
        h_range = self.vpic_domain[h + "max"] - self.vpic_domain[h + "min"]
        v_range = self.vpic_domain[v + "max"] - self.vpic_domain[v + "min"]
        ihs = int((hmin_s / h_range) * self.field_2d.shape[0])
        ihe = int(math.ceil((hmax_s / h_range) * self.field_2d.shape[0]))
        ivs = int((vmin_s / v_range) * self.field_2d.shape[1])
        ive = int(math.ceil((vmax_s / v_range) * self.field_2d.shape[1]))

        # Clamp to actual field bounds
        ihs = max(0, min(ihs, self.field_2d.shape[0]))
        ihe = max(0, min(ihe, self.field_2d.shape[0]))
        ivs = max(0, min(ivs, self.field_2d.shape[1]))
        ive = max(0, min(ive, self.field_2d.shape[1]))

        # rescale the plot when additional panels are included
        orientation = "vertical"
        if ("Contour+" + self.hv[0].upper()) in self.plot_type:
            lhp *= 1.25
        elif self.plot_type == "Contour+" + self.hv[1].upper() + "-Slice":
            lvp *= 1.25
        denp = max(lhp / self.width_max, lvp / self.height_max)
        canvas_h = int(lhp / denp)
        canvas_v = int(lvp / denp)
        if canvas_h < self.width_max // 3:
            canvas_h = self.width_max // 3
        if canvas_v < self.height_max // 3:
            canvas_v = self.height_max // 3
        if self.plot_type == "Contour+" + self.hv[1].upper() + "-Slice":
            if canvas_v < self.height_max // 2:
                canvas_v = self.height_max // 2
        canvas_l = self.middle - canvas_h // 2
        self.canvas.ax_main.clear()
        self.plot_vLayoutWidget.setGeometry(
            QtCore.QRect(canvas_l, self.top, canvas_h, canvas_v))

        im = self.canvas.ax_main.imshow(self.field_2d[ihs:ihe, ivs:ive].T,
                                        extent=[hmin, hmax, vmin, vmax],
                                        vmin=dmin,
                                        vmax=dmax,
                                        cmap=cmap,
                                        aspect='auto',
                                        origin='lower',
                                        interpolation='none')
        if self.field_line_plot:
            self.plot_field_line()
        if self.tracer_plot:
            self.plot_tracer()
        self.canvas.ax_cbar.clear()
        self.canvas.fig.colorbar(im,
                                 cax=self.canvas.ax_cbar,
                                 orientation=orientation)

        # 1D plot
        if self.plot_type == "Contour+" + self.hv[0].upper() + "-Average":
            self.field_1d = np.mean(self.field_2d[ihs:ihe, :], axis=0)
        elif self.plot_type == "Contour+" + self.hv[0].upper() + "-Slice":
            hslice = self.slice_dist[h + "slice_box"].value()
            # Map slice position to actual array index
            h_range = self.vpic_domain[h + "max"] - self.vpic_domain[h + "min"]
            h_fraction = (hslice - self.vpic_domain[h + "min"]) / h_range
            ih_slice = int(h_fraction * self.field_2d.shape[0])
            # Clamp to actual array bounds
            ih_slice = min(max(0, ih_slice), self.field_2d.shape[0] - 1)
            self.field_1d = self.field_2d[ih_slice, :]
            ylim = self.canvas.ax_main.get_ylim()
            self.canvas.ax_main.plot([hslice, hslice], ylim, color='w')
            self.canvas.ax_main.set_ylim(ylim)
        elif self.plot_type == "Contour+" + self.hv[1].upper() + "-Slice":
            vslice = self.slice_dist[v + "slice_box"].value()
            # Map slice position to actual array index
            v_range = self.vpic_domain[v + "max"] - self.vpic_domain[v + "min"]
            v_fraction = (vslice - self.vpic_domain[v + "min"]) / v_range
            iv_slice = int(v_fraction * self.field_2d.shape[1])
            # Clamp to actual array bounds
            iv_slice = min(max(0, iv_slice), self.field_2d.shape[1] - 1)
            self.field_1d = self.field_2d[:, iv_slice]
            xlim = self.canvas.ax_main.get_xlim()
            self.canvas.ax_main.plot(xlim, [vslice, vslice], color='w')
            self.canvas.ax_main.set_xlim(xlim)
        if ("Contour+" + self.hv[0].upper()) in self.plot_type:
            self.canvas.ax1d.clear()
            # Ensure indices are within field_1d bounds
            ivs_clamp = min(ivs, len(self.field_1d))
            ive_clamp = min(ive, len(self.field_1d))
            # Use actual field grid that matches field_1d size
            v_grid = np.linspace(vmin, vmax, len(self.field_1d))
            self.canvas.ax1d.plot(self.field_1d[ivs_clamp:ive_clamp],
                                  v_grid[ivs_clamp:ive_clamp])
            self.canvas.ax1d.set_ylim(self.canvas.ax_main.get_ylim())
        elif self.plot_type == "Contour+" + self.hv[1].upper() + "-Slice":
            self.canvas.ax1d.clear()
            # Ensure indices are within field_1d bounds
            ihs_clamp = min(ihs, len(self.field_1d))
            ihe_clamp = min(ihe, len(self.field_1d))
            # Use actual field grid that matches field_1d size
            h_grid = np.linspace(hmin, hmax, len(self.field_1d))
            self.canvas.ax1d.plot(h_grid[ihs_clamp:ihe_clamp],
                                  self.field_1d[ihs_clamp:ihe_clamp])
            self.canvas.ax1d.set_xlim(self.canvas.ax_main.get_xlim())

        # Get the appropriate length scale label
        scale_label = self.get_length_scale_label()
        xlabel_suffix = f"/{scale_label}" if scale_label else ""
        ylabel_suffix = f"/{scale_label}" if scale_label else ""

        if self.plot_type == "Contour+" + self.hv[1].upper() + "-Slice":
            self.canvas.ax1d.set_xlabel(r"$" + h + xlabel_suffix + "$", fontsize=12)
        else:
            self.canvas.ax_main.set_xlabel(r"$" + h + xlabel_suffix + "$", fontsize=12)
        self.canvas.ax_main.set_ylabel(r"$" + v + ylabel_suffix + "$", fontsize=12)

        # Trigger the canvas to update and redraw.
        self.canvas.draw()
        # plt.tight_layout()

        # save the figure
        if self.save_jpegs and self.is_animation:
            img_dir = ("./img/" + self.var_name + "/" + self.normal + "_" +
                       str(self.plane_index) + "/")
            mkdir_p(img_dir)
            fname = img_dir + self.var_name + "_" + str(self.tframe) + ".jpg"
            self.canvas.fig.savefig(fname)

    def plot_tracer(self):
        if self.tracer_plot:
            self.tracer_plot.remove()
            self.tracer_dot_plot.remove()
            self.canvas.ax_tracer.clear()
            self.canvas.ax_tracer.patch.set_alpha(0.6)
        segments = self.tracer.segments_wo_boundary
        norm = plt.Normalize(self.tracer.kene.min(), self.tracer.kene.max())
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(self.tracer.kene)
        lc.set_linewidth(2)
        self.tracer_plot = self.canvas.ax_main.add_collection(lc)
        # plot the current point
        hdata = self.tracer.tracer_data["d" + self.hv[0].upper()]
        vdata = self.tracer.tracer_data["d" + self.hv[1].upper()]
        self.tindex_tracer = int(self.tindex / config.vpic_info["tracer_interval"])
        if self.tindex_tracer > len(hdata):
            self.tindex_tracer = len(hdata) - 1
        self.tracer_dot_plot, = self.canvas.ax_main.plot(
            hdata[self.tindex_tracer],
            vdata[self.tindex_tracer],
            markersize=10,
            marker='o',
            color='r')
        # plot time vs. energy
        ttracer = self.tracer.tracer_data["t"]
        kene_tracer = self.tracer.kene
        points = np.array([ttracer, kene_tracer]).T.reshape((-1, 1, 2))
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(kene_tracer)
        lc.set_linewidth(1)
        self.tracer_time_energy = self.canvas.ax_tracer.add_collection(lc)
        self.canvas.ax_tracer.set_xlim([ttracer.min(), ttracer.max()])
        ylim = [0, kene_tracer.max()]
        twpe = self.tindex * config.vpic_info["dt*wpe"]
        self.tracer_time_indicator, = self.canvas.ax_tracer.plot([twpe, twpe],
                                                                 ylim,
                                                                 linewidth=1,
                                                                 color='k')
        self.canvas.ax_tracer.set_ylim(ylim)
        self.canvas.ax_tracer.set_xlabel(r"$t\omega_{pe}$", fontsize=12)
        self.canvas.ax_tracer.set_ylabel(r"$\gamma - 1$", fontsize=12)
        self.canvas.draw()

    def start_animation(self):
        self.timer = QtCore.QTimer()
        self.timer.setInterval(config.animation_tinterval)
        self.timer.timeout.connect(self.tick_timer)
        self.timer.start()
        self.tframe_hSlider.setValue(config.tmin)
        self.stop_animateButton.setDisabled(False)
        self.continue_animateButton.setDisabled(False)
        self.is_animation = True
        self.auto_update_old = self.auto_update
        self.autoplot_checkBox.setChecked(True)

    def tick_timer(self):
        tframe = ((self.tframe - config.tmin + 1) % config.nt) + config.tmin
        if self.tframe == config.tmax:
            self.savejpg_checkBox.setChecked(False)
        self.tframe_hSlider.setValue(tframe)

    def stop_animation(self):
        self.timer.stop()
        self.is_animation = False
        self.autoplot_checkBox.setChecked(self.auto_update_old)

    def continue_animation(self):
        self.timer.start()
        self.is_animation = True
        self.auto_update_old = self.auto_update
        self.autoplot_checkBox.setChecked(True)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Quick Check VPIC - Interactive visualization tool for VPIC simulation data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s                          # Auto-detect configuration
  %(prog)s --config my_config.yaml  # Use specific config file
  %(prog)s --hdf5                   # Force HDF5 format
  %(prog)s --gda --no-smooth        # Force GDA format without smoothing
  %(prog)s --tmin 10 --tmax 100     # Set time range

For more information, see the README or config_example.yaml
        '''
    )

    parser.add_argument('-c', '--config', type=str,
                        help='Path to configuration YAML file')
    parser.add_argument('--info-file', type=str, default='../info',
                        help='Path to VPIC info file (default: ../info)')
    parser.add_argument('--log-file', type=str, default='../vpic.out',
                        help='Path to VPIC log file (default: ../vpic.out)')
    parser.add_argument('--hdf5', action='store_true',
                        help='Force HDF5 field format')
    parser.add_argument('--gda', action='store_true',
                        help='Force GDA field format')
    parser.add_argument('--smooth', action='store_true',
                        help='Force smoothed data mode')
    parser.add_argument('--no-smooth', action='store_true',
                        help='Force non-smoothed data mode')
    parser.add_argument('--beta-dir', type=str,
                        help='Beta diagnostics directory name (e.g., "hydro-int-hdf5")')
    parser.add_argument('--tmin', type=int,
                        help='Minimum time frame')
    parser.add_argument('--tmax', type=int,
                        help='Maximum time frame')
    parser.add_argument('--no-auto-detect', action='store_true',
                        help='Disable auto-detection')

    return parser.parse_args()


def main():
    """Main entry point for the application."""
    global config

    # Parse command-line arguments
    args = parse_args()

    # Load configuration
    config = load_config(
        config_file=args.config,
        auto_detect=not args.no_auto_detect
    )

    # Override config with command-line arguments
    if args.info_file:
        config.info_file = args.info_file
        # Reload vpic_info if info file changed
        try:
            config.vpic_info = get_vpic_info(args.info_file)
        except Exception as e:
            print(f"Warning: Could not load info file: {e}")

    if args.log_file:
        config.log_file = args.log_file

    if args.hdf5:
        config.hdf5_fields = True
    elif args.gda:
        config.hdf5_fields = False

    if args.smooth:
        config.smoothed_data = True
    elif args.no_smooth:
        config.smoothed_data = False
        config.smooth_factor = 1

    if args.beta_dir:
        config.dir_beta_hdf5 = args.beta_dir

    if args.tmin is not None:
        config.tmin = args.tmin
    if args.tmax is not None:
        config.tmax = args.tmax

    # Recalculate nt
    config.nt = config.tmax - config.tmin + 1

    # Print final configuration
    print("\n" + "="*60)
    print("VPIC Quick Check - Configuration")
    print("="*60)
    print(f"File format: {'HDF5' if config.hdf5_fields else 'GDA'}")
    print(f"Smoothed data: {config.smoothed_data}")
    if config.smoothed_data:
        print(f"  Smooth factor: {config.smooth_factor}")
        print(f"  Data directory: ../{config.dir_smooth_data}/")
    print(f"Time range: {config.tmin} - {config.tmax} ({config.nt} frames)")
    print(f"Info file: {config.info_file}")
    if config.vpic_info:
        scale = config.vpic_info.get('length_scale', 'de')
        scale_label = 'de' if scale == 'de' else ('di' if scale == 'di' else 'code units')
        print(f"Simulation domain: {config.vpic_info.get('Lx/de', '?')} x "
              f"{config.vpic_info.get('Ly/de', '?')} x "
              f"{config.vpic_info.get('Lz/de', '?')} {scale_label}")
    print("="*60 + "\n")

    # Start Qt application
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
