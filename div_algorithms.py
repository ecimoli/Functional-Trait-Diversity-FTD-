import multiprocessing as mp
import argparse

import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.spatial import ConvexHull
import os
from scipy.spatial import distance
import subprocess
import sys
import time
import xarray as xr
import dask.array as da
# %% import HSS modules

# %%
start = time.time()
np.set_printoptions(suppress=True)


def generate_ndcell_xr(n_cells, min, max, ndim, trait_names):
    """
    Generates a 3D voxel space that represents regularly spaced intervals along a dimension.
    Values represent each possible combination of values between the number of input axes.

    :param voxel_size:
    :param ndim: number of input variables (number of input files originally)
    :return: Array. N-dimensional array
    """
    cells_range = np.linspace(min, max, n_cells + 1)
    cells_label = np.linspace(min, (n_cells + 1) ** ndim, (n_cells + 1) ** ndim)
    gridmesh = np.array(np.meshgrid(*[cells_range] * ndim))
    cells = gridmesh.T.reshape(-1, ndim)
    cells_xr = xr.DataArray(cells, dims=['nd_cell', 'traits'],
                            coords={'nd_cell': cells_label, 'traits': trait_names})
    return cells_xr

def kernel_density_xr(kernels_xr, cells_xr, bandwidth):
    """
    Calculates the kernel density estimate and returns the log density for each pixel in each voxel interval

    :param kernel: Array. Kernel to calculate the density for
    :param kernel_bandwidth: Float/Integer. Factor used to smooth the KDE
    :param voxgrid: Array. Voxel space representing a sample of the probability space, density is fit through this space
    :return: Array. Log density for each pixel in each voxel interval
    """
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kernel_group = kernels_xr.groupby('gridcell', squeeze=True)
    kernel_pix = kernel_group.map(kde.fit)
    kde.fit(kernel_pix)
    density_pix = np.exp(kde.score_samples(cells_xr))

    # this will return an array where the columns represent the pixels, and the rows are the densities across the voxel
    return density_pix


def kernel_density_pix(kernel_group, cells_xr, bandwidth):
    """
    Calculates the kernel density estimate and returns the log density for each pixel in each voxel interval

    :param kernel: Array. Kernel to calculate the density for
    :param kernel_bandwidth: Float/Integer. Factor used to smooth the KDE
    :param voxgrid: Array. Voxel space representing a sample of the probability space, density is fit through this space
    :return: Array. Log density for each pixel in each voxel interval
    """
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    # fills any missing nan values with random choices from the same kernel
    kernel_group = np.where(np.isnan(kernel_group), np.random.choice(kernel_group [~np.isnan(kernel_group)]), kernel_group )
    kde.fit(kernel_group)
    density_pix = np.exp(kde.score_samples(cells_xr))
    # this will return an array where the columns represent the pixels, and the rows are the densities across the voxel
    return density_pix

def kde_fric_pix(density_pix, norm=None, prob_thresh=None):
    """
    Calculates the functional richness index

    :param density: Array. Calculated kernel density estimate for a pixel given as a log density
    :param prob: User defined probability threshold
    :return: Float. Functional Richness value for pixel
    """
    if prob_thresh is None:
        prob_thresh = 0

    richness_pix = np.count_nonzero(density_pix > prob_thresh)
    # fric_cell = richness_pix / np.count_nonzero(density_pix)
    fric_cell = richness_pix / len(density_pix)
    fric_perc = (richness_pix / np.count_nonzero(density_pix)) * 100

    if norm is None:
        return richness_pix
    elif norm == 1:
        return fric_cell
    elif norm == 100:
        return fric_perc

def kde_feve_pix(density_pix, ndim, n_cells, prob_thresh=None):
    """
    Calculatues functional evenness
    :param density: Array. Calculated kernel density estimate for a pixel given as a log density
    :param n_cells: Integer. number of dimensions in the voxel space
    :return: Float. Function evenness estimate
    """
    if prob_thresh is None:
        prob_thresh = 0

    # every pixel has a value in each of the 1331 combination s
    cell_dim = 1 / n_cells
    cell_density = (cell_dim ** ndim * density_pix.T)

    # create an imaginary trait distribution occupying the same functional volume with uniform probabilities throughout.
    cell_non_zero = cell_dim * (np.count_nonzero(density_pix.T > prob_thresh, axis=0).astype(float) ** -1)
    cell_non_zero = cell_non_zero[np.newaxis]
    # create an array with the values repeated from cell_non_zero along the y axis
    cell_constants = np.repeat(cell_non_zero, cell_density.shape[-1], axis=0)
    # overlap between the TPD of the considered unit and the imaginary trait distribution
    min_values = np.minimum(cell_density, cell_constants)
    evenness_pix = np.sum(min_values, axis=0)
    return evenness_pix

def kde_fdiv_pix(density_pix, cells_xr, ndim, n_cells, prob_thresh=None):
    """
    Calculates the functional diversity for a pixel based on the probability density function
    See calculate fde for description of parameters
    """
    if prob_thresh is None:
        prob_thresh = 0

    # density_pix[density_pix < float(prob_thresh)] = 0 # TODO check if this makes sense or remove
    mask = density_pix != 0

    cell_dim = 1 / n_cells
    cell_density = cell_dim ** ndim * density_pix

    # find the center of gravity for the hypervolume grid space
    gravity = np.average(cells_xr[mask[0, :], :], axis=0,
                         weights=density_pix[mask])  # TODO check if weights are ok, density or cell density?
    # calculate distance between each cell and the center of gravity (Carmona 2019 eq 5)
    # creates a tile of replicates of the center of gravity that matches the number of pixels in the kernel
    # that are above the threshold
    eucdist_pix_kernel = distance.cdist(cells_xr[mask[0, :], :],
                                        np.tile(gravity, (cells_xr[mask[0, :], :].shape[0], 1)))
    # Carmona 2019 eq 6
    mean_dist_pix_kernel = np.mean(eucdist_pix_kernel[:, 0])
    # Carmona 2019 eq 7
    delta_diff = np.sum(cell_density[mask] * (eucdist_pix_kernel[:, 0] - mean_dist_pix_kernel))
    # Carmona 2019 eq 8
    delta_diff_abs = np.sum(cell_density[mask] * np.abs((eucdist_pix_kernel[:, 0] - mean_dist_pix_kernel)))
    # final divergence metric  Carmona 2019 eq 9
    divergence_pix = (delta_diff + mean_dist_pix_kernel) / (delta_diff_abs + mean_dist_pix_kernel)
    return divergence_pix


def calculate_kde_feve(density, ndim, n_cells, prob_thresh=None):
    """
    Calculatues functional evenness
    :param density: Array. Calculated kernel density estimate for a pixel given as a log density
    :param n_cells: Integer. number of dimensions in the voxel space
    :return: Float. Function evenness estimate
    """
    if prob_thresh is None:
        prob_thresh = 0

    # convert to nan any

    # density[density < prob_thresh] = 0 # TODO verify if this makes sense

    # every pixel has a value in each of the 1331 combination s
    cell_dim = 1 / n_cells
    cell_density = cell_dim ** ndim * density.T  # TODO verify if cell_dim needs to account for dimensionality

    # create an imaginary trait distribution occupying the same functional volume with uniform probabilities throughout.
    cell_non_zero = cell_dim ** ndim * (np.count_nonzero(density.T > prob_thresh, axis=1).astype(float) ** -1)
    cell_non_zero = cell_non_zero[:, np.newaxis]
    # create an array with the values repeated from cell_non_zero along the y axis
    cell_constants = np.repeat(cell_non_zero, cell_density.shape[-1], axis=1)
    # overlap between the TPD of the considered unit and the imaginary trait distribution
    min_values = np.minimum(cell_density, cell_constants)
    summed = np.sum(min_values, axis=1)
    return summed


def calculate_kde_fdiv(density, ndim, n_cells, cells_xr, prob_thresh=None, old_dim=None):
    """
    Calculates the functional diversity for a pixel based on the probability density function
    See calculate fde for description of parameters
    """
    if prob_thresh is None:
        prob_thresh = 0

    # density[density < float(prob_thresh)] = 0 # TODO check this makes sense
    mask = density >= prob_thresh

    cell_dim = 1 / n_cells
    cell_density = cell_dim ** ndim * density

    # find the center of gravity for the hypervolume grid space
    gravity = np.zeros((ndim, density.shape[-1]))
    mean_dist = []
    divergence = []
    # Keeping the for loop here is marginally faster than using apply_along_axis, so it's here for now
    for pix in list(range(density.shape[-1])):
        gravity[:, pix] = np.average(cells_xr[mask[:, pix]], axis=0, weights=density[
            mask[:, pix], pix])  # TODO check if weights are ok, density or cell density?
        # calculate distance between each cell and the center of gravity (Carmona 2019 eq 5)
        # creates a tile of replicates of the center of gravity that matches the number of pixels in the kernel
        # that are above the threshold
        eucdist_pix_kernel = distance.cdist(cells_xr[mask[:, pix]],
                                            np.tile(gravity[:, pix], (cells_xr[mask[:, pix]].shape[0], 1)))
        # Carmona 2019 eq 6
        mean_dist_pix_kernel = np.mean(eucdist_pix_kernel[:, 0])
        # Carmona 2019 eq 7
        delta_diff = np.sum(cell_density[mask[:, pix], pix] * (eucdist_pix_kernel[:, 0] - mean_dist_pix_kernel))
        # Carmona 2019 eq 8
        delta_diff_abs = np.sum(
            cell_density[mask[:, pix], pix] * np.abs((eucdist_pix_kernel[:, 0] - mean_dist_pix_kernel)))
        # final divergence metric  Carmona 2019 eq 9
        divergence_pix = (delta_diff + mean_dist_pix_kernel) / (delta_diff_abs + mean_dist_pix_kernel)
        divergence.append(divergence_pix)
    if old_dim is None:
        combined = xr.concat(divergence, dim='gridcell')
    else:
        combined = xr.concat(divergence, dim='gridcell')
        combined = combined.assign_coords(gridcell=old_dim.set_names(["latitude", "longitude"]))

    return combined


# TODO add dissimilarity matrix
'''
# Dissimilary matrix to measure association between spectral diversity original proxies of diversity
# We assessed the association between species spectral, functional and phylogenetic distance using spectral,
# functional and phylogenetic dissimilarity matrices. The spectral dissimilarity matrix
# was based on Manhattan distances among species' mean spectra acquired at the leaf level.

# Diversity metrics 
# We calculated spectral, functional and phylogenetic diversity based on qD(TM)40, which uses a dissimilarity matrix and
# a community matrix of species presence/absence or abundance weights as input
# data. Defined as a quantity of distance, qD(TM) calculates the effective number of
# spectrally, functionally or phylogenetically distinct units in a community (which
# can be species, other phylogenetic or functional groups, individuals or image
# pixels) based on the number of units, the regularity (evenness) and dispersion
# of their distribution in mathematical space (see below).I shall make it
'''


# TODO add other metrics

def convex_hull_fric(stack, colormap=None):
    if colormap is None:
        colormap = None

    fig = plt.figure()

    x = stack[:, :, 0].flatten()
    y = stack[:, :, 1].flatten()
    z = stack[:, :, 2].flatten()

    array_tuple = (x, y, z)
    traits_flatten = np.vstack(array_tuple).T

    hull = ConvexHull(traits_flatten)
    # find centroid
    cx = np.mean(hull.points[hull.vertices, 0])
    cy = np.mean(hull.points[hull.vertices, 1])
    cz = np.mean(hull.points[hull.vertices, 2])


def convex_hull_feve(stack, colormap=None):
    if colormap is None:
        colormap = None

    fig = plt.figure()

    x = stack[:, :, 0].flatten()
    y = stack[:, :, 1].flatten()
    z = stack[:, :, 2].flatten()

    array_tuple = (x, y, z)
    traits_flatten = np.vstack(array_tuple).T

    hull = ConvexHull(traits_flatten)


def convex_hull_fdiv(stack, colormap=None):
    if colormap is None:
        colormap = None

    fig = plt.figure()

    x = stack[:, :, 0].flatten()
    y = stack[:, :, 1].flatten()
    z = stack[:, :, 2].flatten()

    array_tuple = (x, y, z)
    traits_flatten = np.vstack(array_tuple).T

    hull = ConvexHull(traits_flatten)


# Rossi et all
# Convex hull
# volume
# (CHV)
# CHV calculates the volume of
# pixels forming a convex hull,
# using the first three principal
# components of the reflectance
# data.
# Dahlin (2016)
#
# Coefficient
# of variation
# (CV)
# CV calculates the ratio between
# the standard deviation and the
# mean of the reflectance value
# at a specific wavelength,
# averaged over all wavelengths.
# Wang, Gamon,
# Cavender-Bares,
# et al. (2018)
#
# Spectral
# species
# richness
# Defines the number of spectral
# species based on clustering of
# the reflectance signal.Feret and Asner
# (2014)
