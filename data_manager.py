import os
import numpy as np
from numpy.lib import stride_tricks
import sys
from sklearn.impute import KNNImputer
import multiprocessing as mp
import rasterio as rio
from osgeo import gdal


def gather_input_files(folder):
    """
    Finds all files in the given folder that have a 'tif' extension.
    Checks all the files for consistency, and returns a stacked numpy array
    of all of the files.

    :param folder: String. Input directory containing the files.
    :return: Array. Stacked n dimensional array, where n is the number of files input.
    """
    input_files = [f for f in os.listdir(folder) if f.endswith('.TIFF')]
    # open and stack all the input files
    data_array = sanity_check_rasters(input_files)
    return data_array


def prepare_datasets(input_folder, rescale_method, cutoff_percentile, kernel_dim):
    """
    Main function used to execute the steps of the processing chain

    :param input_folder: String. Directory containing the files to process
    :param rescale_method: String. Method used to rescale the datasets
    :param cutoff_percentile: Tuple. Upper and lower bounds to scale the dataset to.
    :param kernel_dim: Integer. Size of the kernel to generate for the KDE calculation
    :return: List. Chunked portions of the rescaled array. Number of chunks is equal to the CPU count of the computer
    """
    os.chdir(input_folder)
    input_array, profile = gather_input_files(input_folder)

    rescaled = rescale_datasets(rescale_method, cutoff_percentile, input_array)
    kernel = generate_strides(rescaled,
                              kernel_dim)  # breaks down the file into an array where the 9x9 kernel is saved as a dimension on each pixel
    chunked_arrays = split_arrays(kernel)  # chunk the array for splitting into processes on each CPU core
    return [chunked_arrays, profile]


def sanity_check_rasters(input_files, base_projection=None):
    """
    Designed to sanity check most common errors when joining rasters. Will check against the
    metadata of the first file it finds (they all need to be the same anyway).

    :param input_files: List. All file names to use in the analysis, all end with '.tif' extension
    :return: Array of floats. Stacked and masked dataset as an array
    """
    file_list = []
    for index, file in enumerate(input_files):
        # want to add functionality to check extent, projections, no data value are consistent
        if index == 0:
            # This sets up a baseline to check all the other files against to make sure everything is consistent
            base_mtd = rio.open(file)
        else:
            input_mtd = rio.open(file)
            # check all the variables match
            if input_mtd.crs != base_mtd.crs:
                # I'm just writing this to exit if it doesn't match for now, the reprojection can be added in later
                print("Input coordinate systems do not match, converting image to %s" % base_projection)
                sys.exit(1)
            if input_mtd.bounds != base_mtd.bounds:
                print("Extent of band {!} does not equal bounds of band 1".format(str(index)))
                sys.exit(1)
            if input_mtd.nodata != base_mtd.nodata:
                print("No data value of band {!} does not equal no data value of band 1".format(str(index)))
                sys.exit(1)
        # open the file to a numpy array
        opened_file = rio.open(file).read(1, masked=True)
        file_list.append(opened_file)
    # stack all of the masked arrays
    stack = np.ma.dstack(file_list)
    stack_mask = np.ma.filled(stack.astype(float), np.nan)
    return [stack_mask, base_mtd.profile]


def rescale_datasets(input_array, method, percentiles):
    """
    Rescales an array using a user-defined methodology.

    :param method: String. Rescaling method to use. Accepted options are: normalise, cutoff, standardise, rescale
    :param percentiles: Tuple. If rescaling to a specific range using 'cutoff', upper and lower bounds to use.
    :param input_array: Array. Dataset to rescale
    :return: Array. Rescaled and filled array.
    """
    # flatten out the array
    reshaped_array = input_array.reshape(-1, input_array.shape[-1])
    if method == "normalise":
        print("Normalising file")
        # if we set the percentiles argument to none we can use the same function twice
        normalised_array = np.apply_along_axis(normalise_array, 0, reshaped_array, None)
        normalised_array = np.clip(normalised_array, 0., 1.)
    elif method == 'cutoff':
        print("Normalising file to cutoff values")
        normalised_array = np.apply_along_axis(normalise_array, 0, reshaped_array, percentiles)
        normalised_array = np.clip(normalised_array, 0., 1.)
    elif method == "rescale":
        print("Robust rescaling file")
        normalised_array = np.apply_along_axis(robust_rescale_array, 0, reshaped_array)
    elif method == "standardise":
        print("Standardising file")
        normalised_array = np.apply_along_axis(standardise_array, 0, reshaped_array)
    # fill the nan values to give a continuous variable
    original_dims = normalised_array.reshape(input_array.shape)
    return original_dims


def normalise_array(array, percentiles):
    """
    Calculates percentiles of an array based on a lower and upper bounding tuple

    :param array: Array. Input dataset to normalise. Given as a 2d slice of larger 3d array.
    :param percentiles: Tuple. Upper and lower range to rescale to if using 'cutoff' method
    :return: Array. Normalised dataset scaled between 0 and 1 as a floating point.
    """
    if percentiles:
        lower, upper = np.nanpercentile(array, percentiles)
    else:
        lower = np.nanmin(array)
        upper = np.nanmax(array)
    normalised = (array - lower) / (upper - lower)
    return normalised


def normalise_traits_xr(array, percentiles):
    """
    Calculates percentiles of an array based on a lower and upper bounding tuple
    """
    lower, upper = np.nanpercentile(array, percentiles)
    normalised = (array - lower) / (upper - lower)
    normalised = np.clip(normalised, 0., 1.)
    return normalised


def normalise_ndarray(ndarray, percentiles):
    """
    Calculates percentiles of an array based on a lower and upper bounding tuple

    :param ndarray: Array. Input dataset to normalise. Given as a numpy array of any dimension
    :param percentiles: Tuple. Upper and lower range to rescale to if using 'cutoff' method
    :return: Array. Normalised dataset scaled between 0 and 1 as a floating point.
    """
    stack_norm = np.empty([ndarray.shape[0], ndarray.shape[1], ndarray.shape[-1]])
    for n in range(ndarray.shape[-1]):
        if percentiles:
            lower, upper = np.nanpercentile(ndarray[:, :, n], percentiles)
        else:
            lower = np.nanmin(ndarray[:, :, n])
            upper = np.nanmax(ndarray[:, :, n])
        normalised = (ndarray[:, :, n] - lower) / (upper - lower)
        stack_norm[:, :, n] = normalised
    return stack_norm


def robust_rescale_array(array):
    """
    Rescaling method using the 'robust rescaler'

    :param array: Array. Dataset to rescale, given as a 2D slice of a 3D array
    :return: Array. Rescaled dataset as an array of floating points.
    """
    median = np.nanmedian(array)
    q25, q75 = np.nanpercentile(array, (25, 75))
    scaled_array = (array - median) / (q75 - q25)
    return scaled_array


def standardise_array(array):
    """
    Standarises an array using the mean and standard deviation. Results in an array with a mean of 0 and an std of 1

    :param array: Array. Input array given as a 2D slice of a 3D array
    :return: Array. Standardised array.
    """
    standardised = (array - np.nanmean(array)) / (np.nanstd(array))
    return standardised


def fill_nan(in_array, n_neigh=None, weights=None):
    """
    Interpolates over an array and fills any 'no data' values using a 3x3 nearest neighbour approach

    :param in_array: Array. Dataset containing data values to fill
    :return: Array. Filled dataset.
    """
    if n_neigh and weights is None:
        n_neigh = 3
        weights = "uniform"
    imputer = KNNImputer(n_neighbors=n_neigh, weights=weights)
    filled = imputer.fit_transform(in_array)
    return filled


def fill_ndnan(ndarray, n_neigh=None, weights=None):
    """
    Interpolates over an array and fills any 'no data' values using a 3x3 nearest neighbour approach

    :param in_array: Array. Dataset containing data values to fill
    :return: Array. Filled dataset.
    """
    if n_neigh is None:
        n_neigh = 3
    if weights is None:
        weights = "uniform"
    stack_filled = np.empty([ndarray.shape[0], ndarray.shape[1], ndarray.shape[-1]])
    for n in range(ndarray.shape[-1]):
        imputer = KNNImputer(n_neighbors=n_neigh, weights=weights)
        filled = imputer.fit_transform(ndarray[:, :, n])
        stack_filled[:, :, n] = filled
    return stack_filled

def generate_strides(filled_array, kernel_dim):
    """
    Function used to extract an nxn kernel for each pixel in an array, and project the pixels to a higher dimensional level.
    This is a super handy trick as it lets us apply functions along the axis containing the kernel values rather than having to
    slice the array iteratively and calculate on each sub array.

    :param filled_array: Array. Dataset to derive kernel pixels from, no data values need to be filled.
    :param kernel_dim: Integer. Size of the kernel to extract. The total number of neighbouring pixels will be equal to kernel_dim x kernel_dim
    :return: Array. Reshaped array where the x dimension is the number of input files, y dimension is the pixels of a kernel, and z dimension is a single pixel in the array
    """
    # this needs work but meh
    kernel_list = []
    # TODO Get rid of this loop - format to a 5d array where arr[-1] is the number of input variables
    for x in list(range(filled_array.shape[-1])):
        reshaped_array = filled_array[:, :, x]
        vector_shape = (
            reshaped_array.shape[0] - kernel_dim + 1, reshaped_array.shape[1] - kernel_dim + 1, kernel_dim, kernel_dim)
        # strides represents the number of bits that need to be skipped to get to the next item
        strides = 2 * reshaped_array.strides
        patched_array = stride_tricks.as_strided(reshaped_array, shape=vector_shape, strides=strides)
        # flatten the dimension containing the kernel
        kernel_3D = patched_array.reshape(patched_array.shape[0], patched_array.shape[1], -1)
        flattened_kernel = kernel_3D.reshape(-1, kernel_3D.shape[-1]).T
        kernel_list.append(flattened_kernel)
    # join the variables into a 3d array where x = variables, y= kernel, and z= individual pixels to compute values for
    kernel = np.stack(kernel_list)
    return kernel

def split_arrays(array):
    """
    Splits an array into n chunks along the given axis. Chunks will be equal size if possible, but if not the last chunk will have a different size.
    The array is split into n chunks, where n is the number of CPU cores available.

    :param Array. Strided array to break into chunks.
    :return: List. List containing chunks of array in order.
    """
    num_cores = mp.cpu_count()  # find the number of cores available
    split_array = np.array_split(array, num_cores, axis=-1)
    return split_array

def pad_extent(array, kernel_size):
    """
    Used to pad the array back to the original shape, because when you filter you use kernel width on each side.

    :param array: Array to pad
    :param kernel_size: Kernel size used in the filtering bit
    :return:
    """
    pad = np.floor_divide(kernel_size, 2)
    padded_array = np.pad(array, ((pad, pad), (pad, pad)), "constant", constant_values=0)
    return padded_array


def write_files_out(array, profile, kernel_size, name):
    """
    Saves file, basically what it says on the tin

    :param array: Array. Result file to write out
    :param profile: Dictionary. Profile of raster paramenters taken from original dataset
    :param kernel_size: Integer. Kernel Size
    :param name: String. File name to write out.
    :return: None.
    """
    stacked = np.hstack(array)
    reshaped = stacked.reshape(profile['height'] - kernel_size + 1, profile['width'] - kernel_size + 1)
    padded = pad_extent(reshaped, kernel_size)
    with rio.open(name, 'w', **profile) as dst:
        dst.write(padded.astype(rio.float32), 1)

def kernel_img_reshape(array, profile, kernel_size):
    """
    reshapes the kernel np array to the original trait image shape

    :param array: Array. Result file to write out
    :param profile: Dictionary. Profile of raster paramenters taken from original dataset
    :param kernel_size: Integer. Kernel Size
    :return: None.
    """
    stacked = np.hstack(array)
    reshaped = stacked.reshape(profile['height'] - kernel_size + 1, profile['width'] - kernel_size + 1)
    padded = pad_extent(reshaped, kernel_size)
    return padded

def raster_stack(*args):
    """
    Stacks multiple GeoTIFF image into one stacked file with multiple bands readable by rasterio
    """
    file_list = []
    for arg in args:
        print(f"Import {arg}")
        file_list = file_list + [arg]

    # read shape of raster map
    raster_1 = rio.open(file_list[0])
    stack_1 = raster_1.read(1)
    stack = np.empty([stack_1.shape[0], stack_1.shape[1], len(file_list)])

    # loop to create a stack of of all rasters
    for n in range(1, len(file_list)):
        n_raster = rio.open(file_list[n])
        n_stack = n_raster.read(1)
        stack[:, :, n] = n_stack

    # out_img = "stack.tif"
    #
    # out_meta = raster_1.meta.copy()
    # out_meta.update({"count": len(file_list),
    #                  "nodata": 0})
    #
    # with rio.open(out_img, "w", **out_meta) as dest:
    #     dest.write(stack, 4)

    return stack


def gdal_vrt_stack(*args, out_file):
    """
    Stacks multiple GeoTIFF image into one stacked file with multiple bands readable by rasterio
    Generates a files in the designated folder
    """
    file_list = []
    for arg in args:
        print(f"Import {arg}")
        file_list = file_list + [arg]

    outvrt = '/vsimem/stacked.vrt'  # /vsimem is special in-memory virtual "directory"

    outds_gg = gdal.BuildVRT(outvrt, file_list, separate=True)
    outds = gdal.Translate(out_file, outds_gg)
    return outds, outds_gg

def export_gtif_stack(array, out_file, profile):
    array = array.transpose(2, 0, 1)
    with rio.open(out_file,
                   'w',
                   driver='GTiff',
                   height=array.shape[1],
                   width=array.shape[2],
                   count=array.shape[0],
                   dtype=array.dtype,
                   crs=profile['crs'],
                   nodata=None,  # change if data has nodata value
                   transform=profile['transform']) as dst:
        dst.write(array)

def GTiff_stack(*args, outfile, model):
    """
    Stacks multiple GeoTIFF image into one stacked file with multiple bands readable by rasterio
    Generates a files in the designated folder
    """
    file_list = []
    for arg in args:
        print(f"Import {arg}")
        trait_raster = rio.open(arg)
        arg = trait_raster.read(1, masked=True)
        file_list = file_list + [arg]

    raster_model = rio.open(model)
    out_meta = raster_model.meta.copy()
    out_meta.update({"count": len(file_list)}) # changes raster metadata to update to the number of bands in the new raster

    with rio.open(outfile, 'w', **out_meta) as dest:
        for band_nr, src in enumerate(file_list, start=1):
            dest.write(src, band_nr)
