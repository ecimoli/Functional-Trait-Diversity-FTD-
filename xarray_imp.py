import logging
from pathlib import Path

import numpy as np
import xarray as xr
import rioxarray as rioxr
import sys
import os
from osgeo import gdal, gdalconst
from osgeo.gdalconst import *
import numpy as np
import spectral.io.envi as envi
import numpy as np

logger = logging.getLogger(__name__)

FIELDS = [
    'samples',
    'lines',
    'bands',
    'wavelength',
    'interleave',
    'data type'
]

def read_header(fpath, process=True) -> dict:
    '''subfunctions used to read header file information'''
    with open(fpath, 'r') as fp:
        lines = fp.readlines()

    HDR = {}
    reading = False

    for line in lines:
        # logger.debug(f'Line: {line}')
        if not reading:
            parts = [s.strip() for s in line.split('=')]
            try:
                key, value = parts
                if key not in FIELDS:
                    continue

                if value.startswith('{'):
                    reading = True
                    reading_vals = [value.strip()]
                    reading_key = key
                    logger.info(f"Reading {key}={reading} from {value}")
                else:
                    HDR[key] = value
                    logger.info(f'Read "{key}" = "{value}"')
            except ValueError:
                logger.debug('Invalid header entry')

        else:
            line = line.strip()
            reading_vals.append(line.strip())
            if line.endswith('}'):
                HDR[reading_key] = ' '.join(reading_vals)
                logger.info(f'Done reading "{key}" = "{value}"')
                reading = False
                del reading_key, reading_vals

    if process:
        return _process_envi_fields(HDR)
    else:
        return HDR


def _process_envi_fields(HDR: dict) -> dict:
    '''subfunctions used to read header file information'''
    lines, samples, bands = [int(HDR[_]) for _ in (
        'lines',
        'samples',
        'bands')
                             ]
    interleave = HDR['interleave']

    if interleave == 'bil':
        shape = (lines, bands, samples)
        dims = ('y', 'wavelength', 'x')

    elif interleave == 'bip':
        shape = (lines, samples, bands)
        dims = ('y', 'x', 'wavelength')

    elif interleave == 'bsq':
        shape = (bands, lines, samples)
        dims = ('wavelength', 'y', 'x')

    data_type = int(HDR['data type'])
    if data_type == 4:
        dtype = 'float32'
    elif data_type == 8:
        dtype = 'uint8'
    elif data_type > 8:
        dtype = 'uint16'
    elif data_type == 2:
        dtype = 'int16'

    wavelength_arr = HDR['wavelength']
    # remove the { & } at the start and end, and interpret as float
    wavelengths = np.array([
        float(_) for _ in wavelength_arr[1:-1].split(',')])

    HDR['wavelength'] = wavelengths
    HDR['shape'] = shape
    HDR['dtype'] = dtype
    HDR['dims'] = dims

    return HDR


def load_envi_xr(header_file, data_file=None, data_suffix=None) -> xr.DataArray:
    '''Good for non-georeferenced images as geoinformation is lost'''
    logger.info(f'Loading ENVI from {header_file}')
    header = read_header(header_file, process=True)

    header_file = Path(header_file)
    if not data_file:
        data_file = header_file.with_suffix(data_suffix)
    else:
        data_file = Path(data_file)

    if not data_file.is_file():
        raise FileNotFoundError(f'Could not find {data_file}')

    _mmmap_params = dict(dtype=header['dtype'], mode='r', shape=header['shape'])
    logger.debug(f'Memmap with {_mmmap_params}')
    cube_array = np.memmap(data_file, **_mmmap_params)

    cube = xr.DataArray(cube_array, dims=header['dims'],
                        coords=dict(wavelength=header['wavelength']))

    cube = cube.transpose('wavelength', "x", "y")

    logger.debug(f'Loaded cube: {cube}')
    # # renames coordinates
    # # cube.coords['wavelength'] = ('band', header['wavelength'])
    # cube = cube.rename({'x': x_label, 'y': y_label})
    # cube.attrs['long_name'] = 'data'

    return cube


def load_envi_rioxr(header_file, data_file=None, band=None, x=None, y=None, x_label=None, y_label=None) -> xr.DataArray:
    """Loads an envi file using rasterio xarray, adds useful metadata to the xarrays
    such as wavelengths"""

    header = read_header(header_file, process=True)

    cube = rioxr.open_rasterio(data_file, chunks={'band': band, 'x': x, 'y': y})

    # renames coordinates
    cube.coords['wavelength'] = ('band', header['wavelength'])
    cube = cube.swap_dims({'band': 'wavelength'})
    cube = cube.rename({'x': x_label, 'y': y_label})
    cube.attrs['long_name'] = 'data'
    # cube = cube.reset_coords(names=['band', 'spatial_ref'], drop=True) # to drop non needed coordinates in case needed
    return cube

def load_tif_rioxr(data_file, wavelength=None, band=None, x=None, y=None):
    """Loads an tif file using rasterio xarray, adds useful metadata to the xarrays
    such as wavelengths. This fuction is useful for micasense data, and you define wevelengths in the arguments"""

    cube = rioxr.open_rasterio(data_file, chunks={'band': band, 'x': x, 'y': y})
    cube.coords['wavelength'] = ('band', wavelength)
    cube = cube.swap_dims({'band': 'wavelength'})
    cube = cube.rename({'x': 'longitude', 'y': 'latitude'})
    cube.attrs['long_name'] = 'reflectance'
    # cube = cube.reset_coords(names=['band', 'spatial_ref'], drop=True) # to drop non needed coordinates in case needed
    return cube

def load_traits_rioxr(data_file, traits=None, band=None, x=None, y=None):
    """Loads an tif file using rasterio xarray, adds useful metadata to the xarrays
    such as wavelengths. This fuction is useful for micasense data, and you define wevelengths in the arguments"""

    cube = rioxr.open_rasterio(data_file, chunks={'band': band, 'x': x, 'y': y})
    cube.coords['traits'] = ('band', traits)
    cube = cube.swap_dims({'band': 'traits'})
    cube = cube.rename({'x': 'longitude', 'y': 'latitude'})
    # cube.attrs['long_name'] = 'units'

    # cube = cube.reset_coords(names=['band', 'spatial_ref'], drop=True) # to drop non needed coordinates in case needed

    return cube

# %% EXPORT ##########################################################################

def write_envi(hdr_out, img_array, dtype=None, ext=None, interleave=None, force=None):
    # TODO ADD WAVELENGTHS INTO HDR FILE AND IMPROVE HDR FILE
    img_array = np.transpose(img_array.data)
    envi.save_image(hdr_out, img_array, dtype=dtype, ext=ext, interleave=interleave, force=force)
