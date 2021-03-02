# Author: Evert Nasedkin
# email: nasedkinevert@gmail.com

import sys,os
import shutil
os.environ["OMP_NUM_THREADS"] = "1"
import glob
import argparse
import numpy as np

# Weird matplotlib imports for cluster use
import matplotlib
matplotlib.use('Agg') # set the backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib import rcParams
from matplotlib import rc

from astropy.io import fits
from photutils import CircularAperture, CircularAnnulus, aperture_photometry
from typing import List, Optional, Tuple

# My own files
from Astrometry import get_astrometry, read_astrometry, init_sphere, init_gpi, set_psfs

# Exoplanet stuff
import pyklip.instruments.GPI as GPI
import vip_hci as vip
from vip_hci.preproc import cube_recenter_2dfit, cube_recenter_dft_upsampling
from pynpoint import Pypeline, \
                     FitsReadingModule,\
                     ParangReadingModule,\
                     PcaPsfSubtractionModule,\
                     AttributeReadingModule, \
                     BadPixelSigmaFilterModule,\
                     SimplexMinimizationModule,\
                     WavelengthReadingModule

# Matplotlib styling
rc('font',**{'family':'serif','serif':['Computer Modern']},size = 24)
rc('text', usetex=True)  

# Data dir must contain ALL files
# This means: science cubes (wlen,nframes,x,y)
#             psf cubes
#             wavelength
#             parangs
# There is a preprocessing function to help sort everything into the correct formats (GPI and SPHERE)
data_dir = "/u/nnas/data/HR8799/HR8799_AG_reduced/GPIK2/" #SPHERE-0101C0315A-20/channels/

# Instrument name, and optionally the band (ie GPIH, SPHEREYJ)
instrument = "GPI"
planet_name = "HR8799e" # Name to give to all outputs
distance = 41.2925 #pc
pcas = [4,5,6,8,10,12,15,20]
fwhm = 3.5*0.01414 # fwhm will be recalculated
pixscale = 0.00746 # pixscale is updated depending on instrument
numthreads = 15 # Not important, read from config file

DIT_SCIENCE = 64.0 # Set with argparse
DIT_FLUX = 4.0 # Set with argparse
NORMFACTOR = 1.0 # updated based on instrument and/or DITS

def main(args):
    """
    This script will produce residuals and contrasts using Pynpoint for
    an IFU dataset
    """
    # necessary for running in cluster
    sys.path.append(os.getcwd())

    global data_dir 
    global instrument
    global planet_name
    global DIT_SCIENCE
    global DIT_FLUX
    # Let's read in what we need
    parser = argparse.ArgumentParser()
    # path to the data
    parser.add_argument("path", type=str, default= "/u/nnas/data/")
    # What instrument are we using - expects: SPHEREYJH, SPHEREYJ, GPIH, GPIK1, GPIK2
    parser.add_argument("instrument", type=str, default= "GPI")
    # Name of the planet we're looking at 
    parser.add_argument("name", type=str, default= "HR8799")   
    # Separation in mas and posn in PA (two floats for input)
    parser.add_argument("posn", type=float, nargs = "+")
    # OBJECT/SCIENCE and OBJECT/FLUX integration times for normalisation
    parser.add_argument("-ds","--ditscience", type=float, required=False)
    parser.add_argument("-df","--ditflux", type=float, required=False)
    args = parser.parse_args(args)

    # Set up from args
    data_dir = args.path
    instrument = args.instrument
    planet_name = args.name
    guesssep, guesspa = args.posn
    guessflux=5e-5 # as long as it's within an order of magnitude or so it's fine
    base_name = "HR8799_" + instrument
    if args.ditscience is not None:
        DIT_SCIENCE = args.ditscience
    if args.ditflux is not None:
        DIT_FLUX = args.ditflux

    # Setup directories
    if not data_dir.endswith("/"):
        data_dir += "/"
    if not os.path.isdir(data_dir + "pynpoint/"):
        os.makedirs(data_dir + "pynpoint", exist_ok=True) 

    # Instrument parameters
    if "sphere" in instrument.lower():
        nChannels = 39
        pixscale = 0.00746
        shutil.copy("config/Pynpoint_config_SPHERE.ini",data_dir + "PynPoint_config.ini")
    elif "gpi" in instrument.lower():
        nChannels = 37
        pixscale = 0.0162
        shutil.copy("config/Pynpoint_config_GPI.ini",data_dir + "PynPoint_config.ini")

    # Preprocess the data - get files into correct shapes and sizes
    data_shape = preproc_files()

    # Check for KLIP astrometry and either read in or create
    if not os.path.exists(data_dir + "pyklip/"+ planet_name + "_astrometry.txt"):
        if "gpi" in instrument.lower():
            dataset = init_gpi(data_dir)
        elif "sphere" in instrument.lower():
            dataset = init_sphere(data_dir)
        PSF_cube,cal_cube,spot_to_star_ratio = init_psfs(dataset)
        # posn is in sep [mas] and PA [degree], we need offsets in x and y px
        posn = get_astrometry(dataset, PSF_cube, guesssep, guesspa, guessflux,data_dir,planet_name)

    # read_astrometry gives offsets in x,y, need to compute absolute posns
    posn_dict = read_astrometry(data_dir,planet_name)
    posn = (-1*posn_dict["Px RA offset [px]"][0], -1*posn_dict["Px Dec offset [px]"][0])
    posn_pyn = (posn[0] + data_shape[-2]/2.0,posn[1]+data_shape[-1]/2.0)

    # Sanity chec the posn
    print(posn_pyn)

    # Run ADI for each channel individually
    run_all_channels(nChannels,
                 base_name,
                 instrument + "_" + planet_name,
                 posn_pyn)

    # Save outputs to numpy arrays
    contrasts = save_contrasts(nChannels,
                            base_name,
                            data_dir + "pynpoint/",
                            instrument + "_" + planet_name)

    # Use stellar model to convert to flux units
    save_flux(contrasts)


# Saves PCA residuals to fits file
def save_residuals(residuals,name,output_place):   
    hdu = fits.PrimaryHDU(residuals)
    hdul = fits.HDUList([hdu])
    hdul.writeto(output_place + name + '.fits',overwrite = True)

# Do a full ADI + SDI analysis to check that inputs work correctly
def test_analysis(input_name,psf_name,output_name,posn,working_dir,waffle = False):
    # 
    set_fwhm(0)
    test_pipeline = Pypeline(working_place_in=working_dir,
                        input_place_in=data_dir,
                        output_place_in=data_dir + "pynpoint/")

    module = FitsReadingModule(name_in="read_science",
                               input_dir=data_dir,
                               image_tag="science",
                               ifs_data = True,
                               filenames = sorted(glob.glob(data_dir + input_name)))
    test_pipeline.add_module(module)

    module = WavelengthReadingModule(name_in="read_wlen",
                               input_dir=data_dir,
                               data_tag="science",
                               file_name = "wavelength.fits")

    test_pipeline.add_module(module)

    module = FitsReadingModule(name_in="read_psf",
                               input_dir=data_dir,
                               image_tag="psf",
                               ifs_data = False,
                               filenames = sorted(glob.glob(data_dir + psf_name)))

    test_pipeline.add_module(module)
    module = WavelengthReadingModule(name_in="read_wlen_psf",
                               input_dir=data_dir,
                               data_tag="psf",
                               file_name = "wavelength.fits")

    test_pipeline.add_module(module)
    # might need to glob parang files
    module = ParangReadingModule(file_name="parangs.fits",
                                 input_dir=data_dir,
                                 name_in="parang",
                                 data_tag = 'science',
                                 overwrite=True)
    test_pipeline.add_module(module)
    
    module = PcaPsfSubtractionModule(name_in = "test_analysis",
                                     images_in_tag = "science",
                                     reference_in_tag = "science",
                                     res_median_tag = "median",
                                     pca_numbers = [1,3,5,8,10,12,20],
                                     processing_type = "ADI",
                                     )
    test_pipeline.add_module(module)
    test_pipeline.run()
    residuals = test_pipeline.get_data("median")
    save_residuals(residuals,output_name +"_residuals_ADISDI", data_dir+ "pynpoint/" )

# Run Simplex minimization on a single channel to get the contrast
def simplex_one_channel(channel,input_name,psf_name,output_name,posn,working_dir):
    set_fwhm(channel)
    pipeline = Pypeline(working_place_in=working_dir,
                        input_place_in=data_dir,
                        output_place_in=data_dir + "pynpoint/")

    module = FitsReadingModule(name_in="read_science",
                               input_dir=data_dir,
                               image_tag="science",
                               ifs_data = False,
                               filenames = [data_dir + input_name])

    pipeline.add_module(module)
    module = FitsReadingModule(name_in="read_center",
                               input_dir=data_dir,
                               image_tag="center",
                               ifs_data = False,
                               filenames = [data_dir + input_name])

    pipeline.add_module(module)

    module = FitsReadingModule(name_in="read_psf",
                               input_dir=data_dir,
                               image_tag="psf",
                               ifs_data = False,
                               filenames = [data_dir + psf_name])

    pipeline.add_module(module)
    # might need to glob parang files
    module = ParangReadingModule(file_name="parangs.fits",
                                 input_dir=data_dir,
                                 name_in="parang",
                                 data_tag = 'science',
                                 overwrite=True)
    pipeline.add_module(module)
    module = ParangReadingModule(file_name="parangs.fits",
                                 input_dir=data_dir,
                                 name_in="parang_cent",
                                 data_tag = 'center',
                                 overwrite=True)
    pipeline.add_module(module)

    module = BadPixelSigmaFilterModule(name_in='bad',
                                  image_in_tag='science',
                                  image_out_tag='science_bad',
                                  map_out_tag=None,
                                  box=9,
                                  sigma=5.,
                                  iterate=3)

    pipeline.add_module(module)

    module = SimplexMinimizationModule(name_in = 'simplex',
                                       image_in_tag = 'science_bad',
                                       psf_in_tag = 'psf',
                                       res_out_tag = 'flux_channel_' + channel+"_",
                                       flux_position_tag = 'flux_pos_channel_' + channel +"_",
                                       position = posn,
                                       magnitude = 16.0, # approximate planet contrast in mag
                                       psf_scaling = -1, # deal with spectrum mormalization later
                                       merit = 'gaussian', #better than hessian
                                       aperture = 0.07, # in arcsec
                                       tolerance = 0.005, # tighter tolerance is good
                                       pca_number = pcas, #listed above
                                       cent_size = 0.15, # how much to block out 
                                       offset = 1.0) #use fixed astrometry from KLIP

    pipeline.add_module(module)
    pipeline.run()
    for pca in pcas:
        flux = pipeline.get_data('flux_pos_channel_' + str(channel).zfill(3) + "_" + str(pca).zfill(3))
        np.savetxt(data_dir+ "pynpoint/" + output_name + "_ch" + str(channel).zfill(3) +"_flux_pos_out_pca_" +str(pca)+ ".dat",flux)
        residuals = pipeline.get_data('flux_channel_' + str(channel).zfill(3) + "_" + str(pca).zfill(3))
        save_residuals(residuals[-1],output_name +"_residuals_" + str(channel).zfill(3) + "_pca_" + str(pca), data_dir+ "pynpoint/" )

# Run Pynpoint
def run_all_channels(nchannels, base_name, output_name, posn):
    if "sphere" in instrument.lower():
        # use 'frames removed' files
        test_analysis("*_removed.fits","*_PSF.fits",output_name,posn,data_dir)
    elif "gpi" in instrument.lower():
        # reshape data into a single file to read in for correct shape
        test_analysis("*science_full.fits","*_PSF.fits",output_name,posn,data_dir)

    # Loop over all channels
    for channel in range(nchannels):
        working_dir = data_dir + "pynpoint/CH" + str(channel).zfill(3) +"/"

        # Have to copy the config file to the working dir 
        # Using separate dirs to get unique databases for each wavelength
        # Might be unnecessary, but then I don't have a 50GB hdf5 file to deal with
        if not os.path.isdir(working_dir):
            os.makedirs(working_dir, exist_ok=True) 
        if "sphere" in instrument.lower():
            shutil.copy("config/Pynpoint_config_SPHERE.ini", working_dir + "PynPoint_config.ini")
        elif "gpi" in instrument.lower():
            shutil.copy("config/Pynpoint_config_GPI.ini", working_dir + "PynPoint_config.ini")
        
        # Naming everything consistently
        # Channel must be zfilled for sorting 
        # Better option would be using fits headers, but that's a pain.
        name = base_name +"_" + str(channel).zfill(3) + '_reduced.fits'
        psf_name = base_name +"_" + str(channel).zfill(3) + '_PSF.fits'
        output_place = data_dir+"pynpoint/"
        if os.path.isfile(working_dir + "PynPoint_database.hdf5"):
            os.remove(working_dir + "PynPoint_database.hdf5")

        # Run Simplex Minimization module in pynpoint
        simplex_one_channel(str(channel).zfill(3),name,psf_name,output_name,posn,working_dir)

    # Save residuals to a more manageable file
    residuals = [] #pynpoint mag units
    contrasts = [] #in actual contrast units, only needs stellar spectrum normalization
    for pca in pcas:
        rpcas = []
        contrast = []
        for channel in range(nchannels):
            hdul = fits.open(data_dir + "pynpoint/"+output_name+"_residuals_" + str(channel).zfill(3) + "_pca_" + str(pca)+".fits")
            data = hdul[0].data
            rpcas.append(data)
            contrast.append(np.power(10.0,data/-2.5)*NORMFACTOR)
            hdul.close()
        rpcas = np.array(rpcas)
        contrast = np.array(contrast)
        residuals.append(rpcas)
        contrasts.append(contrast)
    residuals = np.array(residuals)
    contrasts = np.array(contrasts)

    # Write residuals in magnitude units
    hdu = fits.PrimaryHDU(residuals)
    hdul = fits.HDUList([hdu])
    hdul.writeto(data_dir+"pynpoint/" + instrument+ "_"+ planet_name + '_magnitudes.fits',overwrite = True)

    # Write contrast in contrast units (surprise)
    hdu = fits.PrimaryHDU(contrasts)
    hdul = fits.HDUList([hdu])
    hdul.writeto(data_dir+"pynpoint/" + instrument+ "_"+ planet_name + '_residuals.fits',overwrite = True)

# Save contrasts to a useable array
def save_contrasts(nchannels,base_name,output_place,output_name):
    contrasts = [] # the contrast of the planet itself
    for pca in pcas:
        contrast = []
        for channel in range(nchannels):
            samples = np.genfromtxt(data_dir + "pynpoint/"+ output_name + "_ch" + str(channel).zfill(3) +"_flux_pos_out_pca_" +str(pca)+ ".dat")
            contrast.append(samples[-1][4])
        contrasts.append(contrast)
    cont = np.array(contrasts)
    np.save(output_place + output_name +  "contrasts",cont) # saved in magnitude units
    return cont

# Normalize with stellar flux
def save_flux(contrasts):
    stellar_model = np.genfromtxt("/Users/nasedkin/data/HR8799/stellar_model/hr8799_star_spec_"+ instrument.upper() +"_fullfit_10pc.dat").T
    fluxes = []
    for i in range(len(pcas)):
        fluxes.append(stellar_model[1]*10**((contrasts[i])/-2.5)* NORMFACTOR)
    fluxes = np.array(fluxes)
    np.save(data_dir + "pynpoint/" + instrument + "_" + planet_name + "_flux",fluxes) # Saved in W/m2/micron at 10pc

# Reshape science and PSF files for PCA, 
def preproc_files():
    global NORMFACTOR
    # If everything is already processed, continue.
    if os.path.exists(data_dir + "HR8799_"+instrument+"_000_reduced.fits"):
        hdul = fits.open(data_dir + "HR8799_"+instrument+"_000_reduced.fits")
        cube = hdul[0].data
        return cube.shape

    data_shape = None
    if "sphere" in instrument.lower():
        science_name = "frames_removed.fits"
        psf_name = "psf_satellites_calibrated.fits"

        hdul = fits.open(data_dir + science_name)
        cube = hdul[0].data
        NORMFACTOR = DIT_SCIENCE/DIT_FLUX

        # Data shape is used to calculate image center, so it's returned
        if data_shape is None:
            data_shape = cube.shape
        # Separate full cube into wavelength channels
        # SimplexMinimization doesn't work on IFU data naturally
        for channel,frame in enumerate(cube[:]):
            hdu = fits.PrimaryHDU(frame)
            hdul_new = fits.HDUList([hdu])
            hdul_new.writeto(data_dir + "HR8799_"+instrument+"_" + str(channel).zfill(3) + '_reduced.fits',overwrite=True)
        hdul.close()
        hdul = fits.open(data_dir + psf_name)
        cube = hdul[0].data

        # Individual PSFs
        for channel,frame in enumerate(cube[:]):
            hdu = fits.PrimaryHDU(frame)
            hdul_new = fits.HDUList([hdu])
            hdul_new.writeto(data_dir + "HR8799_"+instrument+"_" + str(channel).zfill(3) + '_PSF.fits',overwrite = True)

    elif "gpi" in instrument.lower():
        science_name = "*distorcorr.fits"
        psf_name = glob.glob(data_dir + "*-original_PSF_cube.fits")[0]
        psf_hdul = fits.open(psf_name)
        psfs = psf_hdul[0].data

        # Filelist MUST be sorted for PAs and frames to be in correct order for pynpoint
        # Assuming standard GPI naming scheme
        filelist = sorted(glob.glob(data_dir +science_name))
        dataset = GPI.GPIData(filelist, highpass=True, PSF_cube = psfs,recalc_centers=True)
        dataset.generate_psf_cube(41)

        band = dataset.prihdrs[0]['APODIZER'].split('_')[1]
        spot_to_star_ratio = dataset.spot_ratio[band]
        NORMFACTOR = spot_to_star_ratio

        # Need to order the GPI data for pynpoint
        shape = dataset.input.shape
        science = dataset.input.reshape(len(filelist),37,shape[-2],shape[-1])
        science = np.swapaxes(science,0,1)
        science_pyn = []
        if data_shape is None:
                data_shape = science.shape
        for channel,frame in enumerate(science[:]):
            # The PSF center isn't aligned with the image center, so let's fix that
            centx = dataset.centers.reshape(len(filelist),37,2)[:,channel,0]
            centy = dataset.centers.reshape(len(filelist),37,2)[:,channel,1]
            shiftx,shifty = (int((frame.shape[-2]/2))*np.ones_like(centx) - centx,
                             (int(frame.shape[-1]/2))*np.ones_like(centy) - centy)
            shifted = vip.preproc.recentering.cube_shift(frame,shiftx,shifty)

            # Copy the GPI header, and add some notes of our own
            header_hdul = fits.open(filelist[0])
            # Save channel by channel files
            hdu = fits.PrimaryHDU(shifted)
            hdu.header = header_hdul[0].header
            hdu.header.update(header_hdul[1].header)
            hdu.header['NDIT'] = science.shape[1] 
            hdu.header['NAXIS3'] = science.shape[1]
            hdu.header['CDELT3'] = np.mean(np.diff(dataset.PAs.reshape(len(filelist),37)[:,0]))
            hdu.header['WAVELENGTH'] = dataset.wvs[channel]
            hdu.header['ESO ADA POSANG'] = (dataset.PAs.reshape(len(filelist),37)[:,0][0]+ 180.0)
            hdu.header['ESO ADA POSANG END'] = (dataset.PAs.reshape(len(filelist),37)[:,0][-1]+ 180.0 )
            hdul_new = fits.HDUList([hdu])
            hdul_new.writeto(data_dir + "HR8799_"+instrument+"_" + str(channel).zfill(3) + '_reduced.fits',overwrite=True)
            header_hdul.close()

            # Save for a full file, not channel by channel
            science_pyn.append(shifted)
        # Save the full file (wlens,nframes,x,y)
        hdu = fits.PrimaryHDU(np.array(science_pyn))
        header_hdul = fits.open(filelist[0])
        hdu.header = header_hdul[0].header
        hdu.header.update(header_hdul[1].header)
        hdu.header['NDIT'] = science.shape[1]
        hdu.header['NAXIS3'] = science.shape[1] # Time/PA
        hdu.header['NAXIS4'] = science.shape[0] # WLEN
        hdu.header['CDELT3'] = np.mean(np.diff(dataset.PAs.reshape(len(filelist),37)[:,0]))
        hdu.header['CDELT4'] = np.mean(np.diff(dataset.wvs[:37]))
        hdu.header['ESO ADA POSANG'] = (dataset.PAs.reshape(len(filelist),37)[:,0][0]+ 180.0)
        hdu.header['ESO ADA POSANG END'] = (dataset.PAs.reshape(len(filelist),37)[:,0][-1]+ 180.0 )
        hdul_new = fits.HDUList([hdu])
        hdul_new.writeto(data_dir + "HR8799_"+instrument + '_science_full.fits', overwrite = True)
        header_hdul.close()

        # Repeat the exercise for the PSFs
        for channel,frame in enumerate(dataset.psfs):
            if frame.shape[-1] != science.shape[-1]:
                # PSF must be the same size and shape as the science data
                # Stupid.
                padx = int((science.shape[-1] - frame.shape[0])/2.)
                pady = int((science.shape[-2] - frame.shape[1])/2.)
                if (frame.shape[0] + (2 * padx))%2 == 0:
                    padded = np.pad(frame,
                                ((padx,padx+1),
                                (pady,pady+1)),
                                'constant')
                    padded = vip.preproc.recentering.frame_shift(padded,0.5,0.5)
                else:   
                    padded = np.pad(frame,
                                    ((padx,padx),
                                    (pady,pady)),
                                    'constant')
            else:
                padded = frame
            hdu = fits.PrimaryHDU(padded)
            hdu.header = psf_hdul[0].header
            hdul_new = fits.HDUList([hdu])  
            hdul_new.writeto(data_dir + "HR8799_"+instrument+"_" + str(channel).zfill(3) + '_PSF.fits',overwrite = True)
        # Save wavelengths
        hdu = fits.PrimaryHDU(dataset.wvs[:37])
        hdul_new = fits.HDUList([hdu])
        hdul_new.writeto(data_dir + "wavelength.fits",overwrite = True)

        # pyklip does weird things with the PAs, so let's fix that.
        # Keep or remove dataset.ifs_rotation? GPI IFS is rotated 23.5 deg, 
        pas = (dataset.PAs.reshape(len(filelist),37)[:,0] + 180.0)
        hdu = fits.PrimaryHDU(pas)
        hdul_new = fits.HDUList([hdu])
        hdul_new.writeto(data_dir + "parangs.fits",overwrite = True)
        del dataset
    return data_shape

###########
# Obsolete
###########
def keep_psf_frame(frame):
    if np.any(np.isnan(frame[10:-10,10:-10])):
        return False
    xs = frame.shape[0]/2
    ys = frame.shape[1]/2
    offset = frame.shape[0]/4
    
    pos_list = [[xs,ys],
                [xs + offset, ys],
                [xs, ys + offset],
                [xs - offset, ys],
                [xs, ys - offset]]
    apertures = CircularAperture(pos_list,6)
    phot_table = aperture_photometry(frame, apertures)
    s0 = phot_table['aperture_sum'][0]
    s1 = phot_table['aperture_sum'][1]
    s2 = phot_table['aperture_sum'][2]
    s3 = phot_table['aperture_sum'][3]
    s4 = phot_table['aperture_sum'][4]

    mean_bkg = np.mean(np.array([s1,s2,s3,s4]))
    std = np.std(mean_bkg)
    #print(s0,std,mean_bkg)
    if s0 - mean_bkg < 1.5*std :
        return False
    return True

def median_combine_psf_cube(cube,output_place,output_name):
    psfs = []
    for channel,stack in enumerate(cube[:]):
        keep = []
        for frame in stack:
            if keep_psf_frame(frame):
                keep.append(frame)
        
        psf = np.nan_to_num(np.median(np.array(keep),axis = 0))
        psf = np.pad(psf,((60,60),(60,60)))
        print(psf.shape)
        psfs.append(psf)
        hdu = fits.PrimaryHDU(psf)
        hdul_new = fits.HDUList([hdu])
        hdul_new.writeto(output_place + output_name +"_" + str(channel).zfill(3) + '_PSF.fits',
                        overwrite = True)
    return psfs
    
def reshape_psf(filename):
    psfs = []
    hdul = fits.open(filename,mode='update')
    cube = hdul[0].data
    if cube.shape[0] <250:
        psf = np.pad(cube,((40,41),(40,41)))
    else:
        psf = cube
    hdul[0].data = psf
    hdul.flush()
    hdul.close()
    return psfs

# Get the PSF FWHM for each channel
def set_fwhm(channel):
    # TODO: Not really using FWHM anywhere, will probably delete all references to it
    if "sphere" in instrument.lower():
        psf_name = data_dir + "psf_satellites_calibrated.fits"
    elif "gpi" in instrument.lower():
        psf_name = glob.glob(data_dir + "*-original_PSF_cube.fits")[0]
    hdul = fits.open(psf_name)
    psfs = hdul[0].data
    global fwhm
    if len(psfs.shape) ==4 :
        fwhm_fit = vip.var.fit_2dgaussian(psfs[int(channel),0], crop=True, cropsize=11, debug=False)
    else:
        fwhm_fit = vip.var.fit_2dgaussian(psfs[int(channel)], crop=True, cropsize=11, debug=False)

    fwhm = np.mean(np.array([fwhm_fit['fwhm_y'],fwhm_fit['fwhm_x']]))*pixscale # fit for fwhm
    hdul.close()
    return

#################
# Run the script!
#################
if __name__ == '__main__':
    main(sys.argv[1:])