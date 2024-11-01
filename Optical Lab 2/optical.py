import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import astropy
import astropy.io.fits as fits
import uwastro465isos
import glob
from uwastro465isos import isochrones
import photutils as ph
import astropy.stats
import astrometry
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry
import astropy.coordinates as astcoo
import astropy.units as u
import astropy.wcs
import warnings
warnings.filterwarnings('ignore')
import keyring
from astroquery.astrometry_net import AstrometryNet
from astropy.wcs import WCS
from astroML.crossmatch import crossmatch_angular

# step 2 - reduce your science images using your calibration products
class Reduce:
    
    def make_master_bias(bias_directory_path, mean = False):

        bias_files_list = glob.glob(bias_directory_path)

        bias_images_stack = []

        for filename in bias_files_list:
            image = fits.open(filename)[0].data.astype(float)
            bias_images_stack.append(image)

        bias_stack = np.array(bias_images_stack)

        if mean == True:
            master_bias = np.mean(bias_stack, axis = 0)
            fits.PrimaryHDU(data = master_bias).writeto("master_bias_mean.fits", overwrite = True)
            return master_bias

        if mean == False:
            master_bias = np.median(bias_stack, axis = 0)
            fits.PrimaryHDU(data = master_bias).writeto("master_bias_median.fits", overwrite = True)
            return master_bias

    def make_master_flat_field(flat_field_directory_path, master_bias, g = True, i = True):

        flats_files_list = glob.glob(flat_field_directory_path)

        flats_images_stack = []

        for filename in flats_files_list:
            image = fits.open(filename)[0].data.astype(float)
            image = image - master_bias

            mean_level = np.mean(image)
            image = image / mean_level

            flats_images_stack.append(image)

        flats_stack = np.array(flats_images_stack)
        master_flat = np.median(flats_stack, axis = 0)

        # g filter
        if g == True and i == False:
            fits.PrimaryHDU(data = master_flat).writeto("masterflat_g.fits", overwrite = True)

        # r filter
        if g == False and i == False:
            fits.PrimaryHDU(data = master_flat).writeto("masterflat_r.fits", overwrite = True)

        # i filter
        if g == False and i == True:
            fits.PrimaryHDU(data = master_flat).writeto("masterflat_i.fits", overwrite = True)

        return master_flat

    def reduce_science_image(science_image_path, master_bias, master_flat):

        image = fits.open(science_image_path)[0].data.astype(float)

        image = image - master_bias

        image = image / master_flat
        
        return image

# step 3 - extracting photometry for all sources of interest in your image
class Extract:

    def estimate_background(image_data):
        
        # this is using the astropy.stats package
        mean, background, std = sigma_clipped_stats(image_data, sigma=3.0)
        
        background_subtracted_image = image_data - background
        
        return background_subtracted_image, std, background

    def detect_source(background_subtracted_image, std, threshold):
        
        daofind = DAOStarFinder(fwhm=3.0, threshold=threshold*std)

        sources = daofind(background_subtracted_image)

        for col in sources.colnames:
            if col not in ('id', 'npix'):
                sources[col].info.format = '%.2f'

        sources = sources.to_pandas()

        return sources

    def estimate_source_fluxes(sources, background_subtracted_image):

        # provide source positions in the format the aperture tool requires
        positions = [ (sources.iloc[i]['xcentroid'], sources.iloc[i]['ycentroid']) for i in range(len(sources))]

        # define an aperture circular with radius equal to 4 pixels
        apertures = CircularAperture(positions, r = 4.0)

        # use aperture photometry tool
        photometry_table = aperture_photometry(background_subtracted_image, apertures)
        photometry_table['aperture_sum'].info.format = '%.8g'

        # source flux in  magnitudes = 2.5 log_10 (aperture sum)
        sources['aperture_sum'] = photometry_table['aperture_sum']
        sources['mag'] = -2.5*np.log10(sources['aperture_sum'])

        # print(sources['aperture_sum'])
        
        return sources

    def get_flux_uncertainties(sources, background):
        
        R = 9.904
        gain = 1.0
        npix = np.pi * 4**2
        sources['flux_err'] = np.sqrt(gain * sources['aperture_sum'] + npix*(gain*background + R**2)) / gain
        # converting it to a numpy array from an astropy table column

        return sources
