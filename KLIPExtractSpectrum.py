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
from matplotlib import rcParams
from matplotlib import rc

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
#
# Matplotlib styling
#rc('font',**{'family':'serif','serif':['Computer Modern']},size = 24)
#rc('text', usetex=True)

### KLIP Parameters ###
#numbasis = np.array([2,3,4,5,6,8,10,12,15,18,20,25]) # "k_klip", this can be a list of any size.
numbasis = np.array([2,3,4,10]) # "k_klip", this can be a list of any size.

maxnumbasis = 25 # Max components to be calculated
movement = 1.0 # aggressiveness for choosing reference library
stamp_size = 9 # how big of a stamp around the companion in pixels
                # stamp will be stamp_size**2 pixels
sections = 10
distance = 41.2925 #pc
numthreads = 1

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
DIT_SCIENCE = 1.0 # Set with argparse
DIT_FLUX = 1.0 # Set with argparse
skip = []
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
    global pxscale
    # Let's read in what we need
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default= "/u/nnas/data/")
    parser.add_argument("instrument", type=str, default= "GPI")
    parser.add_argument("name", type=str, default= "HR8799")
    parser.add_argument("posn", type=float, nargs = "+") # rough guess at sep [mas] and PA [deg]
    # OBJECT/SCIENCE and OBJECT/FLUX integration times for normalisation
    parser.add_argument("-ds","--ditscience", type=float, required=False)
    parser.add_argument("-df","--ditflux", type=float, required=False)
    parser.add_argument("-c","--cont", action='store_true',required=False)

    args = parser.parse_args(args)

    # Setup constants
    data_dir = args.path
    instrument = args.instrument
    planet_name = args.name
    guesssep, guesspa = args.posn
    guessflux = 5e-5
    print(instrument,planet_name)

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
        pxscale =  0.014161
    elif "sphere" in instrument.lower():
        dataset = init_sphere()
        pxscale = 0.007462

    # Setup PSFs, return a few different psfs with different units just in case
    PSF_cube,cal_cube,spot_to_star_ratio = init_psfs(dataset)
    print(dataset.input.shape,PSF_cube.shape)
    # Get the planet astrometry
    if not os.path.exists(data_dir + "pyklip/"+ planet_name + "_astrometry.txt"):
        posn = get_astrometry(dataset, PSF_cube, guesssep, guesspa, guessflux, data_dir, planet_name)

    posn_dict = read_astrometry(data_dir,planet_name)
    posn = (posn_dict["Separation [mas]"][0], posn_dict["PA [deg]"][0])
    print(posn)
    if not args.cont:
        # Run klip
        exspect, fm_matrix = KLIP_Extraction(dataset, PSF_cube, posn, numthreads)
        # Get flux in flux and contrast units
        contrasts,flux = get_spectrum(dataset, exspect,spot_to_star_ratio, stellar_model)
        # Get full frame residuals. Not strictly correct, but should be ok
        KLIP_fulframe(dataset, PSF_cube, posn, numthreads)

    else:
        files = sorted(glob.glob(data_dir + "pyklip/"+instrument + "_" + planet_name +"_fullframe*"))
        if len(files) == 0:
            KLIP_fulframe(dataset, PSF_cube, posn, numthreads)
        exspect = np.load(data_dir + "pyklip/" + instrument + "_" + planet_name + "exspect.npy")
    # Generate nice outputs
    combine_residuals()

    # Find scaling factors for oversubtraction
    mcmc_scaling(dataset,PSF_cube,posn,exspect,spot_to_star_ratio,stellar_model)
    del dataset
    return

def init_sphere():
    datacube = data_dir + "frames_removed.fits"
    if os.path.isfile(data_dir + "psf_satellites_calibrated.fits"):
        psfcube = data_dir + "psf_satellites_calibrated.fits"
    else:
        psfcube = data_dir + "psf_cube.fits"

    # Sanity check on data shape
    # not a fan of hard coded number of channels
    fitsinfo = data_dir + "parang_removed.fits"
    wvinfo = data_dir + "wavelength.fits"
    hdul_w = fits.open(wvinfo)
    # Sanity check on wlen units
    if np.mean(hdul_w[0].data)>100.:
        hdu_wlen = fits.PrimaryHDU([hdul_w[0].data/1000])
        hdu_wlen.header = hdul_w[0].header
        hdu_wlen.header["UNITS"] = "micron"
        hdul_wlen = fits.HDUList([hdu_wlen])
        hdul_wlen.writeto(data_dir + "wavelength.fits",overwrite=True)
    hdul_w.close()
    dataset = SPHERE.Ifs(datacube,
                         psfcube,
                         fitsinfo,
                         wvinfo,
                         nan_mask_boxsize=9,
                         psf_cube_size = 15)
    print("read in data")
    dataset.input = np.nan_to_num(dataset.input)
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
    if not os.path.exists(data_dir + "wavelength.fits"):
        wlen = np.genfromtxt(data_dir + "wlens.txt")
        hdu_wlen = fits.PrimaryHDU([wlen])
        hdu_wlen.header["UNITS"] = "micron"
        hdul_wlen = fits.HDUList([hdu_wlen])
        hdul_wlen.writeto(data_dir + "wavelength.fits",overwrite=True)
    if not os.path.exists(data_dir + "parangs.fits"):
        pas = np.genfromtxt(data_dir + "parangs.txt")
        hdu_pa = fits.PrimaryHDU([pas])
        hdu_pa.header["UNITS"] = "degree"
        hdul_pa = fits.HDUList([hdu_pa])
        hdul_pa.writeto(data_dir + "parangs.fits",overwrite=True)
    dataset = GPI.GPIData(filelist, highpass=False, PSF_cube = psf,recalc_centers=False)
    return dataset

def init_psfs(dataset):
    global maxnumbasis
    # useful constants
    N_frames = len(dataset.input)
    N_cubes = np.size(np.unique(dataset.filenums))
    nl = N_frames // N_cubes
    maxnumbasis = N_cubes
    # The units of your model PSF are important, the return spectrum will be
    # relative to the input PSF model, see next example
    # generate_psf_cube has better background subtraction than generate_psfs
    if "sphere" in instrument.lower():
        return dataset.psfs,dataset.psfs,DIT_FLUX/DIT_SCIENCE
    if "K" in instrument:
        dataset.generate_psfs(11)
    else:
        dataset.generate_psf_cube(17)
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
    #print(planet_sep,planet_pa)
    #print(dataset.input.shape,numbasis,PSF_cube.shape)
    #spectrum = np.load("/u/nnas/data/HR8799/SPHERE-ZURLO/pyklip/SPHEREYJH_HR8799d_contrasts.npy")[0]
    #print(spectrum.shape)
    N_cubes =  int(dataset.input.shape[0]/np.unique(dataset.wvs).shape[0])
    #print(N_cubes)


    ###### The forward model class ######
    # WATCH OUT FOR MEMORY ISSUES HERE
    # If the PSF size, input size or numbasis size is too large, will cause issues on cluster
    dtype = "float"
    if "sphere" in instrument.lower():
        dtype = "float"
    fm_class = es.ExtractSpec(dataset.input.shape,
                        numbasis,
                        planet_sep,
                        planet_pa,
                        PSF_cube,
                        np.unique(dataset.wvs),
                        stamp_size = stamp_size,
                        datatype = dtype) #must be double?

    ###### Now run KLIP! ######
    fm.klip_dataset(dataset, fm_class,
                    fileprefix=instrument + "_" + planet_name +"_fmspect",
                    mode = "ADI+SDI",
                    annuli=[[planet_sep-1.5*stamp_size,planet_sep+1.5*stamp_size]], # select a patch around the planet (radius)
                    subsections=[[(planet_pa-2.0*stamp_size)/180.*np.pi,\
                                  (planet_pa+2.0*stamp_size)/180.*np.pi]], # select a patch around the planet (angle)
                    movement=None,
                    flux_overlap = 0.2,
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
    np.save(data_dir + "pyklip/" + instrument + "_" + planet_name + "klipped",klipped)
    np.save(data_dir + "pyklip/" + instrument + "_" + planet_name + "dn_per_contrast",dn_per_contrast)

    # If you want to scale your spectrum by a calibration factor:
    units = "natural"
    scaling_factor = 1.0
    exspect, fm_matrix = es.invert_spect_fmodel(dataset.fmout, dataset, units=units,
                                                scaling_factor=scaling_factor,
                                                method="leastsq")
    np.save(data_dir + "pyklip/" + instrument + "_" + planet_name + "exspect", exspect)
    np.save(data_dir + "pyklip/" + instrument + "_" + planet_name + "fm_matrix", fm_matrix)
    return exspect, fm_matrix

def forward_model_extraction(dataset,exspect,posn,spot_to_star_ratio,stellar_model):
    planet_sep, planet_pa = posn
    fm_class = fmpsf.FMPlanetPSF(dataset.input.shape, numbasis, guesssep, guesspa, guessflux, dataset.psfs,
                             np.unique(dataset.wvs), dn_per_contrast, star_spt='A6',
                             spectrallib=[guessspec])
    fit = fitpsf.FMAstrometry(planet_sep, planet_pa, 13, method="mcmc")
    # set kernel, no read noise
    corr_len_guess = 3.
    corr_len_label = r"$l$"
    fit.set_kernel("matern32", [corr_len_guess], [corr_len_label])

def get_spectrum(dataset,exspect,spot_to_star_ratio,stellar_model):
    # Convert the extracted spectrum into contrast and flux units
    # spot_to_star_ratio - different for GPI and SPHERE
    exspect_load = np.load(data_dir + "pyklip/" + instrument + "_" + planet_name + "exspect.npy")
    print("Echeck")
    print(exspect[0]-exspect_load[0])
    # Useful constants
    num_k_klip = len(numbasis) # how many k_klips running
    N_frames = len(dataset.input)
    N_cubes = np.size(np.unique(dataset.filenums))
    nl = N_frames // N_cubes
    wlen = dataset.wvs[:nl]

    print("Saving contrasts and flux calibrated spectrum for " + planet_name)
    # Contrast figures
    fig,ax = plt.subplots(figsize = (16,10))
    ax.set_xlabel("Wavelength [micron]")
    ax.set_ylabel("Contrast")
    ax.set_title(instrument + " " + planet_name + " Contrast KLIP")
    m_cont = np.mean(exspect[:],axis=0)
    for i in range(num_k_klip):
        ax.plot(wlen,exspect[i]*spot_to_star_ratio*(DIT_FLUX/DIT_SCIENCE),label=str(numbasis[i]),alpha=0.5)
    ax.plot(wlen,m_cont*spot_to_star_ratio,label = 'Mean',linewidth=4)
    plt.legend()
    plt.savefig(data_dir + "pyklip/" + instrument + "_" + planet_name +"contrasts_KLIP.pdf")

    plt.clf()

    # Flux figure
    fig,ax = plt.subplots(figsize = (16,10))
    print(distance, spot_to_star_ratio, stellar_model.shape,'\n')
    sm = stellar_model[1]
    """if "K1" in instrument:
        sm = np.delete(sm,np.array([0,1,36]))
    elif "K2" in instrument:
        sm = np.delete(sm,np.array([0,1,2]))"""
    for i in range(num_k_klip):
        ax.plot(wlen,exspect[i,:]*spot_to_star_ratio*sm*(distance/10.)**2,label=str(numbasis[i]),alpha=0.5)
    ax.plot(wlen,m_cont*spot_to_star_ratio*sm*(distance/10.)**2 ,label = 'Mean',linewidth=4)
    ax.set_xlabel("Wavelength [micron]")
    ax.set_ylabel("Flux Density [W/m2/micron]")
    ax.set_title(planet_name + " " + instrument + " Flux KLIP")
    plt.legend()
    plt.savefig(data_dir + "pyklip/" + instrument + "_" + planet_name +"flux_KLIP.pdf")

    # Save the data
    np.save(data_dir +"pyklip/" + instrument + "_" + planet_name + "_contrasts",
            exspect*spot_to_star_ratio*(DIT_SCIENCE/DIT_FLUX))
    np.save(data_dir +"pyklip/"+ instrument + "_" + planet_name + "_flux_10pc_7200K",
            exspect*spot_to_star_ratio*sm*((distance/10.)**2)*(DIT_SCIENCE/DIT_FLUX))
    np.save(data_dir +"pyklip/"+ instrument + "_" + planet_name + "spot_to_star_ratio",spot_to_star_ratio)

    # Contrast, Flux Density (W/m^2/micron)
    return exspect*spot_to_star_ratio*(DIT_SCIENCE/DIT_FLUX),exspect*spot_to_star_ratio*sm*(distance/10.)**2*(DIT_SCIENCE/DIT_FLUX)


def KLIP_fulframe(dataset, PSF_cube, posn, numthreads):
    print("Running full frame KLIP for residuals.\n")
    # Run KLIP again at the end so we can get the full residuals,
    # which we need for the covariance matrix later.
    planet_sep, planet_pa = posn
    planet_sep =planet_sep/1000 / pxscale
    ###### The forward model class ######
    """fm_class = es.ExtractSpec(dataset.input.shape,
                        numbasis,
                        planet_sep,
                        planet_pa,
                        PSF_cube,
                        np.unique(dataset.wvs),
                        stamp_size = stamp_size,
                        datatype = 'float')

    ###### Now run KLIP! ######
    fm.klip_dataset(dataset, fm_class,
                    fileprefix=instrument + "_" + planet_name +"_fullframe",
                    annuli=12,
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
                    mute_progression=True)"""
    parallelized.klip_dataset(dataset,
                mode='ADI',
                fileprefix=instrument + "_" + planet_name +"_fullframe",
                annuli=12,
                subsections=10,
                movement=movement,
                #flux_overlap = 0.1,
                numbasis = numbasis,
                maxnumbasis=maxnumbasis,
                numthreads=numthreads,
                spectrum=None,
                #time_collapse = 'weighted-mean',
                #save_klipped=True,
                algo='klip',
                highpass=True,
                calibrate_flux=True,
                outputdir=data_dir + "pyklip/")
    return

def recover_fake(dataset, PSF_cube, files, position, fake_flux, kklip):
    # We will need to create a new dataset each time.
    N_frames = len(dataset.input)
    N_cubes = np.size(np.unique(dataset.filenums))
    nl = N_frames // N_cubes

    # PSF model template for each cube observation, copies of the PSF model:
    inputpsfs = np.tile(dataset.psfs, (N_cubes, 1, 1))
    #bulk_contrast = 1e-2
    fake_psf = inputpsfs*fake_flux[:,None,None]*dataset.dn_per_contrast[:,None,None]

    planet_sep, pa = position
    planet_sep = planet_sep/1000 / pxscale #mas to pixels
    if "sphere" in instrument.lower():
        tmp_dataset = init_sphere()

    if "gpi" in instrument.lower():
        tmp_dataset = init_gpi()
    tmp_PSF_cube,cal_cube,spot_to_star_ratio = init_psfs(tmp_dataset)
    print(fake_psf.shape,PSF_cube.shape)
    print(np.mean(fake_psf),np.mean(PSF_cube))
    fakes.inject_planet(tmp_dataset.input, tmp_dataset.centers, fake_psf,\
                                    tmp_dataset.wcs, planet_sep, pa)

    fm_class = es.ExtractSpec(tmp_dataset.input.shape,
                               numbasis[kklip],
                               planet_sep,
                               pa,
                               PSF_cube,
                               np.unique(tmp_dataset.wvs),
                               stamp_size = stamp_size,
                               datatype = 'float')

    fm.klip_dataset(tmp_dataset, fm_class,
                        fileprefix=instrument + "_" + planet_name +"_fakespect",
                        annuli=[[planet_sep-1.5*stamp_size,planet_sep+1.5*stamp_size]],
                        subsections=[[(pa-2.0*stamp_size)/180.*np.pi,\
                                      (pa+2.0*stamp_size)/180.*np.pi]],
                        movement=movement,
                        numbasis = numbasis[kklip],
                        maxnumbasis=maxnumbasis,
                        numthreads=numthreads,
                        spectrum=None,
                        save_klipped=True,
                        highpass=True,
                        calibrate_flux=True,
                        outputdir=data_dir + "pyklip/")
    fake_spect, fakefm = es.invert_spect_fmodel(tmp_dataset.fmout,
                                           tmp_dataset, method="leastsq",
                                           units="natural", scaling_factor=1.0)
    del tmp_dataset
    return fake_spect

def mcmc_scaling(dataset,PSF_cube,posn,exspect,spot_to_star_ratio,stellar_model):
    print("Running MCMC to compute over-subtraction scaling factor.")

    files = glob.glob(data_dir + "*corr.fits")
    N_frames = len(dataset.input)
    N_cubes = np.size(np.unique(dataset.filenums))
    nl = N_frames // N_cubes

    #print(files)
    # This could take a long time to run
    # Define a set of PAs to put in fake sources
    npas = 11
    planet_sep, planet_pa = posn
    planet_sep =planet_sep
    pas = (np.linspace(planet_pa, planet_pa+360, num=npas+2)%360)[1:-1]

    # For numbasis "k"
    # repeat the spectrum over each cube in the dataset
    scaling = []
    for k in [2]:
        input_spect = np.tile(exspect[k,:]*spot_to_star_ratio, N_cubes)
        fake_spectra = np.zeros((npas, nl))
        for p, para in enumerate(pas):
            fake_spectra[p,:] = recover_fake(dataset, PSF_cube, files, (planet_sep, para), input_spect, k)
            scaling.append((exspect[k,:]*spot_to_star_ratio)/(fake_spectra[p,:]/dataset.dn_per_contrast[:nl]))
    np.save(data_dir + "pyklip/"+ instrument + "_" + planet_name + "mcmc_outputs",fake_spectra)

    scaling = np.array(scaling)
    sm = stellar_model[1]
    np.save(data_dir + "pyklip/"+ instrument + "_" + planet_name + "_mcmc_scale_factor",scaling)
    np.save(data_dir + "pyklip/"+ instrument + "_" + planet_name + "_scaled_spectrum",np.mean(scaling,axis=0)*exspect*spot_to_star_ratio*(DIT_SCIENCE/DIT_FLUX)*sm*(distance/10.)**2)

def combine_residuals():
    print("Combining residuals into fits file.")
    print(instrument, planet_name)
    files = glob.glob(data_dir + "pyklip/"+instrument + "_" + planet_name +"_fullframe-KL*")
    hduls = []
    hdu0 = fits.PrimaryHDU()
    hdul = fits.open(files[0])
    hdu0.header = hdul[0].header
    hdul.close()

    hduls.append(hdu0)
    sortlist = [0]
    if "sphere" in instrument.lower():
        dataind = 0 # For some reason the outputs for the fm are different
    elif "gpi" in instrument.lower():
        dataind = 1
    i = 0
    for f in files:
        if "KLmodes" in f:
            continue
        pca = 0
        try:
            pca = int(f.split("-KL")[-1].split("-")[0])
        except TypeError:
            continue
        if not pca in numbasis:
            continue
        sortlist.append(pca)
        hdul = fits.open(f)
        data = hdul[dataind].data
        hdu = fits.ImageHDU(data,name = str(pca)+"PC")
        hdu.header = hdul[dataind].header
        hduls.append(hdu)
        hdul.close()
        i+=1
    print(sortlist)
    sorted_inds_by_pca = np.argsort(np.array(sortlist))
    hdul = fits.HDUList([hduls[sorted_ind] for sorted_ind in sorted_inds_by_pca])
    hdul.writeto(data_dir+"pyklip/" + instrument+ "_"+ planet_name + '_residuals.fits',
                 overwrite=True, checksum=True, output_verify='fix')



#################
# Run the script!
#################
if __name__ == '__main__':
    main(sys.argv[1:])