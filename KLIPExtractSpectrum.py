# Author: Evert Nasedkin
# email: nasedkinevert@gmail.com

import glob
import sys,os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg') # set the backend before importing pyplot
import matplotlib.pyplot as plt

from astropy.io import fits
import spectres

# Pyklip stuff
import pyklip.instruments.GPI as GPI
import pyklip.instruments.SPHERE as SPHERE
import pyklip.fm as fm
import pyklip.fakes as fakes
import pyklip.fmlib.extractSpec as es
import pyklip.fmlib.fmpsf as fmpsf
import pyklip.fitpsf as fitpsf
import pyklip.parallelized as parallelized

# My own stuff
from Astrometry import get_astrometry, read_astrometry

### KLIP Parameters ###
numbasis = np.array([3,4,5,8,10,12,15]) # "k_klip", this can be a list of any size.
maxnumbasis = 20 # Max components to be calculated
movement = 2.0 # aggressiveness for choosing reference library
stamp_size = 11 # how big of a stamp around the companion in pixels
                # stamp will be stamp_size**2 pixels
sections = 10
distance = 41.2925 #pc
numthreads = 4

# These constants are all updated later
# Data dir must contain ALL files
# This means: science cubes (wlen,nframes,x,y)
#             psf cubes
#             wavelength
#             parangs
# There is a preprocessing function to help sort everything into the correct formats (GPI and SPHERE)
data_dir = "/u/nnas/data/"
instrument = "GPI"
planet_name = "HR8799"
pxscale = 0.01422
DIT_SCIENCE = 64.0 # Set with argparse
DIT_FLUX = 4.0 # Set with argparse

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

    # Let's read in what we need
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default= "/u/nnas/data/")
    parser.add_argument("instrument", type=str, default= "GPI")
    parser.add_argument("name", type=str, default= "HR8799")   
    parser.add_argument("posn", type=float, nargs = "+") # rough guess at sep [mas] and PA [deg]
    # OBJECT/SCIENCE and OBJECT/FLUX integration times for normalisation
    parser.add_argument("-ds","--ditscience", type=float, required=False)
    parser.add_argument("-df","--ditflux", type=float, required=False)

    args = parser.parse_args(args)

    # Setup constants
    data_dir = args.path
    instrument = args.instrument
    planet_name = args.name
    guesssep, guesspa = args.posn
    guessflux = 5e-5
    if args.ditscience is not None:
        DIT_SCIENCE = args.ditscience
    if args.ditflux is not None:
        DIT_FLUX = args.ditflux
    # Setup directories
    if not data_dir.endswith("/"):
        data_dir += "/"
    if not os.path.isdir(data_dir + "pyklip"):
        os.makedirs(data_dir + "pyklip", exist_ok=True)

    # Read in stellar model - instrument name is important
    stellar_model = np.genfromtxt("/u/nnas/data/HR8799/stellar_model/hr8799_star_spec_" + instrument.upper() + "_fullfit_10pc.dat").T

    # initialize pyklip dataset
    if "gpi" in instrument.lower():
        dataset = init_gpi()
        pxscale = 0.0162
    elif "sphere" in instrument.lower():
        dataset = init_sphere()
        pxscale = 0.007462

    # Setup PSFs, return a few different psfs with different units just in case
    PSF_cube,cal_cube,spot_to_star_ratio = init_psfs(dataset)

    # Get the planet astrometry
    if not os.path.exists(data_dir + "pyklip/"+ planet_name + "_astrometry.txt"):
        posn = get_astrometry(dataset, PSF_cube, guesssep, guesspa, guessflux, data_dir, planet_name)
    else: 
        posn_dict = read_astrometry(data_dir,planet_name)
        posn = (posn_dict["Separation [mas]"][0], posn_dict["PA [deg]"][0])
    exspect, fm_matrix = KLIP_Extraction(dataset, PSF_cube, posn, numthreads)
    contrasts,flux = get_spectrum(dataset, exspect,spot_to_star_ratio, stellar_model)
    KLIP_fulframe(dataset, PSF_cube, posn, numthreads)
    combine_residuals()
    return exspect, fm_matrix, contrasts, flux

def init_sphere():
    datacube = data_dir + "science_pyklip.fits"
    psfcube = data_dir + "psf_pyklip.fits"
    fitsinfo = data_dir + "parang_removed.fits"
    wvinfo = data_dir + "wavelength.fits"
    dataset = SPHERE.Ifs(datacube, 
                         psfcube,
                         fitsinfo,
                         wvinfo, 
                         nan_mask_boxsize=9,
                         psf_cube_size = 11 )
    print("read in data")
    return dataset

def init_gpi():
    # GPI
    # Original files 131117,131118,160919
    # PynPoint structure GPIH, GPIK1, GPIK2
    psf_name = glob.glob(data_dir + "*_PSF_cube.fits")[0]

    psf = fits.open(psf_name)[0].data
    if not os.path.isdir(data_dir + "pyklip"):
        os.makedirs(data_dir + "pyklip", exist_ok=True) 

    filelist = sorted(glob.glob(data_dir +"*distorcorr.fits"))
    dataset = GPI.GPIData(filelist, highpass=False, PSF_cube = psf,recalc_centers=True)
    return dataset

def init_psfs(dataset):

    # useful constants
    N_frames = len(dataset.input)
    N_cubes = np.size(np.unique(dataset.filenums))
    nl = N_frames // N_cubes

    # The units of your model PSF are important, the return spectrum will be
    # relative to the input PSF model, see next example
    # generate_psf_cube has better background subtraction than generate_psfs
    if "sphere" in instrument.lower():
        return dataset.psfs,dataset.psfs,DIT_SCIENCE/DIT_FLUX
    
    dataset.generate_psf_cube(21)
    # NOTE: not using pretty much all of the example calibration
    PSF_cube = dataset.psfs
    model_psf_sum = np.nansum(PSF_cube, axis=(1,2))
    model_psf_peak = np.nanmax(PSF_cube, axis=(1,2))

    # Now divide the sum by the peak for each wavelength slice
    aper_over_peak_ratio = model_psf_sum/model_psf_peak

    # NOTE: THIS IS THE IMPORTANT ONE FOR FLUX CALIBRATION
    # star-to-spot calibration factor
    band = dataset.prihdrs[0]['APODIZER'].split('_')[1]
    spot_to_star_ratio = dataset.spot_ratio[band]

    # not using calibrated model
    spot_peak_spectrum = \
        np.median(dataset.spot_flux.reshape(len(dataset.spot_flux)//nl, nl), axis=0)
    calibfactor = np.array(aper_over_peak_ratio*spot_peak_spectrum / spot_to_star_ratio)
    # calibrated_PSF_model is the stellar flux in counts for each wavelength
    calibrated_PSF_model = calibfactor[:,None,None]*PSF_cube
    return PSF_cube, calibrated_PSF_model, spot_to_star_ratio

def KLIP_Extraction(dataset, PSF_cube, posn, numthreads):
    planet_sep, planet_pa = posn
    planet_sep =planet_sep/1000 / pxscale #mas to pixels

    # Print some sanity checks
    print(planet_sep,planet_pa)
    print(dataset.input.shape,numbasis,PSF_cube.shape)

    ###### The forward model class ######
    # WATCH OUT FOR MEMORY ISSUES HERE
    # If the PSF size, input size or numbasis size is too large, will cause issues on cluster
    fm_class = es.ExtractSpec(dataset.input.shape,
                        numbasis,
                        planet_sep,
                        planet_pa,
                        PSF_cube,
                        np.unique(dataset.wvs),
                        stamp_size = stamp_size,
                        datatype = 'double') #must be double

    ###### Now run KLIP! ######
    fm.klip_dataset(dataset, fm_class,
                    fileprefix="fmspect",
                    annuli=[[planet_sep-1.5*stamp_size,planet_sep+1.5*stamp_size]], # select a patch around the planet (radius)
                    subsections=[[(planet_pa-2.0*stamp_size)/180.*np.pi,\
                                  (planet_pa+2.0*stamp_size)/180.*np.pi]], # select a patch around the planet (angle)
                    movement=movement,
                    #flux_overlap = 0.1,
                    numbasis = numbasis,
                    maxnumbasis=maxnumbasis,
                    numthreads=numthreads,
                    spectrum=None,
                    #time_collapse = 'weighted-mean',
                    save_klipped=True,
                    highpass=True,
                    calibrate_flux=True,
                    outputdir=data_dir + "pyklip/",
                    mute_progression=True)

    # Save all outputs for future reference
    klipped = dataset.fmout[:,:,-1,:]
    dn_per_contrast = dataset.dn_per_contrast
    np.save(data_dir + "pyklip/klipped",klipped)
    np.save(data_dir + "pyklip/dn_per_contrast",dn_per_contrast)

    # If you want to scale your spectrum by a calibration factor:
    units = "natural"
    scaling_factor = 1.0
    exspect, fm_matrix = es.invert_spect_fmodel(dataset.fmout, dataset, units=units,
                                                scaling_factor=scaling_factor,
                                                method="leastsq")
    np.save(data_dir + "pyklip/exspect",exspect)
    np.save(data_dir + "pyklip/fm_matrix",fm_matrix)
    return exspect, fm_matrix

def get_spectrum(dataset,exspect,spot_to_star_ratio,stellar_model):
    # Convert the extracted spectrum into contrast and flux units
    # spot_to_star_ratio - different for GPI and SPHERE
     
    # Useful constants
    num_k_klip = len(numbasis) # how many k_klips running
    N_frames = len(dataset.input)
    N_cubes = np.size(np.unique(dataset.filenums))
    nl = N_frames // N_cubes
    wlen = dataset.wvs[:nl]

    # Contrast figures
    fig,ax = plt.subplots(figsize = (16,10))
    ax.set_xlabel("Wavelength [micron]")
    ax.set_ylabel("Contrast")
    ax.set_title(instrument + " " + planet_name + " Contrast KLIP")
    m_cont = np.mean(exspect[:],axis=0)
    for i in range(num_k_klip):
        ax.plot(wlen,exspect[i]*spot_to_star_ratio,label=str(numbasis[i]),alpha=0.5)
    ax.plot(wlen,m_cont*spot_to_star_ratio,label = 'Mean',linewidth=4)
    ax.set_ylim(0.0,3e-5)
    plt.legend()
    plt.savefig(data_dir + "pyklip/" + instrument + "_" + planet_name +"contrasts_KLIP.pdf")

    plt.clf()

    # Flux figure
    fig,ax = plt.subplots(figsize = (16,10))

    for i in range(num_k_klip):
        ax.plot(wlen,exspect[i]*spot_to_star_ratio*stellar_model[1]*(distance/10.)**2,label=str(numbasis[i]),alpha=0.5)
    ax.plot(wlen,m_cont*spot_to_star_ratio*stellar_model[1]*(distance/10.)**2 ,label = 'Mean',linewidth=4)
    ax.set_xlabel("Wavelength [micron]")
    ax.set_ylabel(r"Flux Density [W/m$^{2}/\mu$m")
    ax.set_title(planet_name + " " + instrument + " Flux KLIP")
    plt.legend()
    plt.savefig(data_dir + "pyklip/" + instrument + "_" + planet_name +"flux_KLIP.pdf")

    # Save the data
    np.save(data_dir +"pyklip/" + instrument + "_" + planet_name + "_contrasts",
            exspect*spot_to_star_ratio)
    np.save(data_dir +"pyklip/"+ instrument + "_" + planet_name + "_flux_10pc_7200K",
            exspect*spot_to_star_ratio*stellar_model[1]*(distance/10.)**2)
   
    # Contrast, Flux Density (W/m^2/micron)
    return exspect*spot_to_star_ratio,exspect*spot_to_star_ratio*stellar_model[1]*(distance/10.)**2


def KLIP_fulframe(dataset, PSF_cube, posn, numthreads):
    # Run KLIP again at the end so we can get the full residuals,
    # which we need for the covariance matrix later.
    planet_sep, planet_pa = posn
    planet_sep =planet_sep/1000 / pxscale
    ###### The forward model class ######
    fm_class = es.ExtractSpec(dataset.input.shape,
                        numbasis,
                        planet_sep,
                        planet_pa,
                        PSF_cube,
                        np.unique(dataset.wvs),
                        stamp_size = stamp_size,
                        datatype = 'float')

    ###### Now run KLIP! ######
    fm.klip_dataset(dataset, fm_class,
                    fileprefix="fullframe",
                    annuli=9,
                    subsections=10,
                    movement=movement,
                    #flux_overlap = 0.1,
                    numbasis = numbasis,
                    maxnumbasis=maxnumbasis,
                    numthreads=numthreads,
                    spectrum=None,
                    #time_collapse = 'weighted-mean',
                    save_klipped=True,
                    highpass=True,
                    calibrate_flux=True,
                    outputdir=data_dir + "pyklip/",
                    mute_progression=True)
    return 

def combine_residuals():
    files = sorted(glob.glob(data_dir + "pyklip/*fullframe*"))
    residuals = []
    for f in files:
        hdul = fits.open(f)
        residuals.append(hdul[0].data)
        hdul.close()
    hdr_hdul = fits.open(files[0])
    hdu = fits.PrimaryHDU(np.array(residuals))
    hdu.header = hdr_hdul[0].header
    hdu.header = {**hdu.header, **hdr_hdul[1].header}
    hdul = fits.HDUList([hdu])
    hdul.writeto(data_dir+"pyklip/" + instrument+ "_"+ planet_name + '_residuals.fits',overwrite = True)



#################
# Run the script!
#################
if __name__ == '__main__':
    main(sys.argv[1:])