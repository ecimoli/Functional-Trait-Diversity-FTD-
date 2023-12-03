'''
Script to explore functional diversity metrics Richness, Divergence and Evenness
rescale_method = Method you want to use to rescale
cutoff = Cutoff threshold for normalisation as a tuple
kernel_size =
bandwidth = bandwidth for kernel density estimation (smoothing parameter)
voxel_size = size of the 3d voxel, i.e. number of points to sample in the kde
prob_threshold = base probability threshold for TPD
input_folder = folder containing input datasets

by Emiliano Cimoli @EC_lambda
'''

import pandas as pd
import matplotlib.pyplot as plt
from dask.distributed import Client
from diversity.data_manager import *
from diversity.div_algorithms import *
from data_imp.xarray_imp import *
from modules.visual import trait_rgb, trait_composite, trait_hist, trait_scatter, kde_plot

# %% Set directories prepare file
# select folder with traits GeoTIFF data
in_dir = Path(r"C:\Users\ecimoli\PycharmProjects\HSS\diversity\test_set")

# create output folder in case not existent
if not os.path.exists(os.path.join(in_dir, 'out')):
    os.mkdir(os.path.join(in_dir, 'out'))

# select folder and files names for the outputs
out_dir = Path(r"C:\Users\ecimoli\PycharmProjects\HSS\diversity\test_set\out")


file_out_fric = out_dir / "new_test_ric_UAS_61_005_10_10_2_70.TIFF"  # Functional Richness map output
file_out_fdiv = out_dir / "new_test_div_UAS_61_005_10_10_2_70.TIFF"  # Functional Richness map output
# file_out_feve = out_dir / "eve_swan_airborne_31_005_10_10_2_100.TIFF"  # TODO finish implementation of FEve
nan_mask_out = out_dir / "tumba_nan_mask.TIFF"  # nan_mask_utility map output

# select traits you are going to analyze
trait_1_file = str(in_dir / "triPRI_proc.TIFF")  #
trait_2_file = str(in_dir / "Datt_Cab_proc.TIFF")  #
trait_3_file = str(in_dir / "GIT_THREEBAND_ANT_proc.TIFF")  #
# ... add more traits if needed replicating the same line

# name the traits for plotting and analysis in the same order as above
trait_names = ['trait_1', 'trait_2', 'trait_3']

# name output file for stack array of traits selected
outfile = str(out_dir / 'stacked_traits_3d_triPRI_Datt_Cab_mARI_5.TIFF')
outfile_norm = str(out_dir / 'stacked_traits_3d_norm_triPRI_Datt_Cab_mARI_5.TIFF')

# define variables
scaling_method = 'normalise'
norm_min = 0.
norm_max = 1.
kernel_size = int(61)
kernel_type = 'gaussian'
bandwidth = float(0.05)
lat_stride = 30
long_stride = 30
n_cells = int(10)
percentiles = (0.5, 99.5)
prob_thresh = float(2)
cell_size = (norm_max - norm_min) / n_cells
ndim = 3
kernel_nan_thresh = 0  # percentage of non-NaN pixels inside a kernel beyond which calculating the kernel TPD is omitted

# dask array chunking
band = 3
x = 300
y = 300

# %% stacks traits tiffs together and normalize, also produces file in the outfile folder

# prompts link to Dask dashboard for monitoring processing
client = Client(processes=False)
print(client)
print("Open browser and enter the following URL for Dask dashboard: " + str(client.dashboard_link))

# load trait stack as Xarray format
GTiff_stack(trait_1_file, trait_2_file, trait_3_file, outfile=outfile, model=trait_1_file)

traits_xr = load_traits_rioxr(outfile, traits=trait_names)

# normalizes trait values over the specifies min and max
traits_xr_norm = xr.apply_ufunc(normalise_traits_xr, traits_xr.groupby('traits'), dask='parallelized', keep_attrs=True,
                                kwargs={"percentiles": percentiles})

# outputs a copy of the normalized stacked trait matrix
traits_xr_norm.rio.to_raster(outfile_norm)

# re-import import in dask array format
traits_xr = load_traits_rioxr(outfile_norm, traits=trait_names, band=band, x=x, y=y)

# re-chunk to fit all dimensions
traits_xr = traits_xr.chunk({'traits': len(trait_names)})

# generate voxel space to fit the data through, ndim is represented in dimension 1 of each array chunk
cells_xr = generate_ndcell_xr(n_cells, norm_min, norm_max, ndim=traits_xr.shape[0], trait_names=trait_names)

# %% Stack and Rolling approach
# 2d rolling window approach to construct a database based Xarray methods and

rolling_window = traits_xr.rolling(latitude=kernel_size, longitude=kernel_size,
                                   center={'latitude': True, 'longitude': True}, min_periods=1)

window_construct = rolling_window.construct(latitude='x_window', longitude='y_window',
                                            stride={'latitude': lat_stride, 'longitude': long_stride})

# threshold aims to drop empty areas at the boundaries of the raster, can add threshold
kernels_xr = window_construct.stack(gridcell=("latitude", "longitude"), window=["x_window", "y_window"]).transpose(
    'gridcell', 'window', 'traits').dropna('gridcell', how='all',
                                           thresh=((kernel_nan_thresh * kernel_size ** 2) / 100))
# %% kernel density operation

# A work around that saves gridcell information to be reused later
sav_gridcell_dim = kernels_xr.indexes["gridcell"]

# A work around that saves gridcell information to be reused later
print("Please ignore this warning")
kernels_xr.coords['gridcell'] = np.linspace(0, kernels_xr.shape[0], kernels_xr.shape[0])


kernels_xr = kernels_xr.fillna(kernels_xr.mean(dim='gridcell')) # fill any of the few remaining NaNs with mean within each kernel, so it does not affect metric value

# kernels_xr = kernels_xr.fillna(0)
# kernels_xr = kernels_xr.fillna(np.random.random()) # fill any of the few remaining NaNs with mean within each kernel, so it does not affect metric value

# grouping by gridcell so
kernel_groups = kernels_xr.groupby('gridcell')

# compute kernel density
t0 = time.perf_counter()
kernel_density = xr.apply_ufunc(kernel_density_pix,
                                kernel_groups,
                                cells_xr,
                                input_core_dims=[['window', 'traits'], ['nd_cell', 'traits']],
                                output_core_dims=[['nd_cell']],
                                exclude_dims=set(('window',)),
                                join='exact',
                                dask='allowed',
                                keep_attrs=True,
                                kwargs={'bandwidth': bandwidth},
                                vectorize=False)
t1 = time.perf_counter()
tdiff = t1 - t0

kernel_density = kernel_density.assign_coords(gridcell=sav_gridcell_dim.set_names(["latitude", "longitude"]))

# %% Apply functional metrics to kernels by groups

# %% Compute and export FRic
prob_thresh_kernels = np.quantile(kernel_density, 0.95, axis=1)

prob_thresh_kernels.mean()

kde_fric_uf = xr.apply_ufunc(kde_fric_pix,
                             kernel_density.groupby('gridcell'),
                             input_core_dims=[['nd_cell', 'gridcell']],
                             exclude_dims=set(('nd_cell',)),
                             join='exact',
                             dask='allowed',
                             keep_attrs=True,
                             kwargs={'prob_thresh': None, 'norm': 1},
                             vectorize=False)

kde_fric_map = kde_fric_uf.unstack('gridcell')
kde_fric_map.plot()

# %% Compute and export FDiv
kde_fdiv_uf = xr.apply_ufunc(kde_fdiv_pix,
                             kernel_density.groupby('gridcell'),
                             cells_xr,
                             input_core_dims=[['gridcell', 'nd_cell'], ['nd_cell', 'traits']],
                             exclude_dims=set(('nd_cell',)),
                             join='exact',
                             dask='allowed',
                             keep_attrs=True,
                             kwargs={'ndim': kernels_xr.shape[2], "n_cells": n_cells, 'prob_thresh': prob_thresh},
                             vectorize=False)

kde_fdiv_map = kde_fdiv_uf.unstack('gridcell')
kde_fdiv_map.plot()

# %% Compute and export FEve
# # TODO Implement computation of Functional Evenness (FEve). Needs to be completed.
#
# kde_feve_uf = xr.apply_ufunc(kde_feve_pix,
#                              kernel_density.groupby('gridcell'),
#                              input_core_dims=[['nd_cell']],
#                              exclude_dims=set(('nd_cell',)),
#                              join='exact',
#                              dask='allowed',
#                              keep_attrs=True,
#                              kwargs={'ndim': kernels_xr.shape[2], "n_cells": n_cells, 'prob_thresh': prob_thresh},
#                              vectorize=False)
#
# kde_feve_map = kde_feve_uf.unstack('gridcell')


# %% create mask file to remove accidental nans

coars_mask = traits_xr.isel(traits=1).coarsen(latitude=lat_stride, longitude=long_stride, boundary="pad").mean()
coars_mask.plot()

mask = coars_mask * 0

mask = mask.assign_coords({"longitude": kde_fric_map.coords["longitude"].values})
mask = mask.assign_coords({"latitude": kde_fric_map.coords["latitude"].values})

mask = mask.T

mask = xr.DataArray(np.rot90(mask.data), dims=("latitude", "longitude"))
mask.plot()

nan_mask = mask.where(mask != 0, 1).where(mask == 0, 0)
nan_mask.plot()

nan_mask = kde_fric_map.copy(deep=True, data=nan_mask)

# %% export files
# export mask file for
nan_mask.rio.to_raster(nan_mask_out)

# export FRic mask file for
masked_fric = nan_mask * kde_fric_map # applies mask
masked_fric = masked_fric.where(masked_fric != 0, np.nan)
plt.figure()
masked_fric.plot()
masked_fric.rio.to_raster(file_out_fric)

# export FDiv file for
masked_fdiv = nan_mask * kde_fdiv_map # applies mask
masked_fdiv = masked_fdiv.where(masked_fdiv != 0, np.nan)
plt.figure()
masked_fdiv.plot()
masked_fdiv.rio.to_raster(file_out_fdiv)

# # export FEve mask file for
# # TODO Implement computation of Functional Evenness (FEve). Needs to be completed.
# masked_feve = nan_mask * kde_feve_map # applies mask
# masked_feve = masked_fdiv.where(masked_feve != 0, np.nan)
# plt.figure()
# masked_feve.plot()
# masked_feve.rio.to_raster(file_out_feve)
