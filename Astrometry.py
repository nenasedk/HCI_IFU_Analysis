import glob
import sys,os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pyklip.instruments.GPI as GPI
import pyklip.instruments.SPHERE as SPHERE
import pyklip.fm as fm
import pyklip.fakes as fakes
import pyklip.fmlib.extractSpec as es
import pyklip.fmlib.fmpsf as fmpsf
import pyklip.fitpsf as fitpsf

import pyklip.parallelized as parallelized
import matplotlib
matplotlib.use('Agg') # set the backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import rc

from astropy.io import fits
import spectres

# Matplotlib styling
#rc('font',**{'family':'serif','serif':['Computer Modern']},size = 24)
#rc('text', usetex=True)

instrument = "GPI"
numthreads = 4

def init_sphere(data_dir):
    global instrument
    instrument = 'SPHERE'

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

    dataset = SPHERE.Ifs(datacube,
                         psfcube,
                         fitsinfo,
                         wvinfo,
                         nan_mask_boxsize=9,
                         psf_cube_size = 13)
    print("read in data")
    return dataset

def init_gpi(data_dir):
    # GPI
    # Original files 131117,131118,160919
    # PynPoint structure GPIH, GPIK1, GPIK2
    global instrument
    instrument = "GPI"
    psf_name = glob.glob(data_dir + "*psf_cube.fits")[0]

    psf = fits.open(psf_name)[0].data
    if not os.path.isdir(data_dir + "pyklip"):
        os.makedirs(data_dir + "pyklip", exist_ok=True)

    filelist = sorted(glob.glob(data_dir +"*distorcorr.fits"))
    dataset = GPI.GPIData(filelist, highpass=False, PSF_cube = psf,recalc_centers=True)
    return dataset


def init_psfs(dataset):
    if "GPI" in dataset.__class__.__name__:
        instrument = "GPI" + dataset.band
    else:
        instrument = "SPHERE"
    # useful constants
    N_frames = len(dataset.input)
    N_cubes = np.size(np.unique(dataset.filenums))
    nl = N_frames // N_cubes

    # The units of your model PSF are important, the return spectrum will be
    # relative to the input PSF model, see next example
    # generate_psf_cube has better background subtraction than generate_psfs
    if "sphere" in instrument.lower():
        # Not letting me get this value from the dataset obj for some reason.
        # Taken from GPI SPHERE IFS class
        #psfs_wvs = np.unique(datset.wvs)
        #star_peaks = np.nanmax(dataset.psfs,axis=(1,2))
        #dn_per_contrast = np.squeeze(np.array([star_peaks[np.where(psfs_wvs==wv)[0]] for wv in dataset.wvs]))

        return dataset.psfs, dataset.psfs, dataset.dn_per_contrast

    if "K" in instrument:
        # In case we're skipping a few K band channels
        dataset.generate_psfs(11)
    else:
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


def get_astrometry(dataset, PSF_cube, guesssep, guesspa, guessflux, data_dir, planet_name):
    if not os.path.isdir(data_dir + "pyklip"):
        os.makedirs(data_dir + "pyklip", exist_ok=True)

    #### Astrometry Prep ###
    #guessspec = np.array(klipcontrast) #your_spectrum # should be 1-D array with number of elements = np.size(np.unique(dataset.wvs))
    # klipcontrast read from residuals below
    numbasis=np.array([10,15])
    if "Ifs" in dataset.__class__.__name__:
        instrument = "SPHERE"
    else:
        instrument = "GPI"
    # initialize the FM Planet PSF class
    if "sphere" in instrument.lower():
        dataind = 0 # For some reason the outputs for the fm are different
        dn_per_contrast = None
        guesssep = guesssep/1000 / dataset.platescale
    elif "gpi" in instrument.lower():
        dn_per_contrast = dataset.dn_per_contrast# your_flux_conversion # factor to scale PSF to star PSF. For GPI, this is dataset.dn_per_contrast
        dataind = 1
        guesssep = guesssep/1000 / GPI.GPIData.lenslet_scale

    fm_class = fmpsf.FMPlanetPSF(dataset.input.shape, numbasis, guesssep, guesspa, guessflux, dataset.psfs,
                                np.unique(dataset.wvs), dn_per_contrast, star_spt='A0V',
                                spectrallib=None)
    # Astrometry KLIP
    # PSF subtraction parameters
    # You should change these to be suited to your data!
    outputdir = data_dir + "pyklip/" # where to write the output files
    prefix = instrument + "_" + planet_name + "_fmpsf" # fileprefix for the output files
    stamp_size = 11
    annulus_bounds = [[guesssep-3*stamp_size, guesssep+3*stamp_size]] # one annulus centered on the planet, one for covariance

    subsections = [[(guesspa-3.0*stamp_size)/180.*np.pi,(guesspa+3.0*stamp_size)/180.*np.pi]] # we are not breaking up the annulus
    padding = 0 # we are not padding our zones
    movement = 2 # we are using an conservative exclusion criteria of 4 pixels
    numbasis = [5,10]
    # run KLIP-FM
    fm.klip_dataset(dataset, fm_class, outputdir=outputdir, fileprefix=prefix, numbasis=numbasis,
                    annuli=annulus_bounds, subsections=subsections, padding=padding, movement=movement)

    ### FIT ASTROMETRY ###
    # read in outputs
    output_prefix = os.path.join(outputdir, prefix)

    fm_hdu = fits.open(output_prefix + "-fmpsf-KLmodes-all.fits")
    data_hdu = fits.open(output_prefix + "-klipped-KLmodes-all.fits")

    # get FM frame, use KL=10
    fm_frame = fm_hdu[dataind].data[0]
    fm_centx = fm_hdu[dataind].header['PSFCENTX']
    fm_centy = fm_hdu[dataind].header['PSFCENTY']

    # get data_stamp frame, use KL=10
    data_frame = data_hdu[dataind].data[0]
    data_centx = data_hdu[dataind].header["PSFCENTX"]
    data_centy = data_hdu[dataind].header["PSFCENTY"]

    # get initial guesses
    guesssep = fm_hdu[0].header['FM_SEP']
    guesspa = fm_hdu[0].header['FM_PA']

    # create FM Astrometry object that does MCMC fitting
    fit = fitpsf.FMAstrometry(guesssep, guesspa, 13, method="mcmc")
    # alternatively, could use maximum likelihood fitting
    # fit = fitpsf.FMAstrometry(guesssep, guesspa, 13, method="maxl")

    # generate FM stamp
    # padding should be greater than 0 so we don't run into interpolation problems
    fit.generate_fm_stamp(fm_frame, [fm_centx, fm_centy], padding=5)

    # generate data_stamp stamp
    # not that dr=4 means we are using a 4 pixel wide annulus to sample the noise for each pixel
    # exclusion_radius excludes all pixels less than that distance from the estimated location of the planet
    fit.generate_data_stamp(data_frame, [data_centx, data_centy], dr=4, exclusion_radius=10)

    # set kernel, no read noise
    corr_len_guess = 3.
    corr_len_label = r"$l$"
    fit.set_kernel("matern32", [corr_len_guess], [corr_len_label])
    # set bounds

    x_range = 5 # pixels
    y_range = 5 # pixels
    flux_range = 2 # flux can vary by an order of magnitude
    corr_len_range = 3. # between 0.3 and 30
    fit.set_bounds(x_range, y_range, flux_range, [corr_len_range])

    # run MCMC fit
    fit.fit_astrometry(nwalkers=200, nburn=500, nsteps=5000, numthreads=numthreads)
    plot_astrometry(fit,data_dir,planet_name)
    if "gpi" in instrument.lower():
        platescale = GPI.GPIData.lenslet_scale*1000
        plate_err = 0.007
    elif "sphere" in instrument.lower():
        platescale = dataset.platescale*1000
        plate_err = 0.02
    # Outputs and Erro Propagation
    fit.propogate_errs(star_center_err=0.05,
                       platescale=platescale,
                       platescale_err=plate_err,
                       pa_offset=-0.1,
                       pa_uncertainty=0.13)

    write_astrometry(fit,data_dir,planet_name)
    return fit

def plot_astrometry(fit,data_dir,planet_name):
    chain = fit.sampler.chain

    ### Astrometry Plots ###
    fig = plt.figure(figsize=(10,8))
    # plot RA offset
    ax1 = fig.add_subplot(411)
    ax1.plot(chain[:,:,0].T, '-', color='k', alpha=0.3)
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("RA")

    # plot Dec offset
    ax2 = fig.add_subplot(412)
    ax2.plot(chain[:,:,1].T, '-', color='k', alpha=0.3)
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Dec")

    # plot flux scaling
    ax3 = fig.add_subplot(413)
    ax3.plot(chain[:,:,2].T, '-', color='k', alpha=0.3)
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("alpha")

    # plot hyperparameters.. we only have one for this example: the correlation length
    ax4 = fig.add_subplot(414)
    ax4.plot(chain[:,:,3].T, '-', color='k', alpha=0.3)
    ax4.set_xlabel("Steps")
    ax4.set_ylabel("l")
    plt.savefig(data_dir + "pyklip/"+planet_name+"_astrometry_walkers.pdf")

    plt.clf()
    # Corner Plots are broken for some reason
    #fig = plt.figure()
    #fig = fit.make_corner_plot(fig=fig)
    #plt.savefig(data_dir+"pyklip/"+planet_name+"_astrometry_corner.pdf")
    #plt.clf()
    # Residual Plots
    fig = plt.figure()
    fig = fit.best_fit_and_residuals(fig=fig)
    plt.savefig(data_dir+"pyklip/"+planet_name+"_astrometry_residuals.pdf")

def write_astrometry(fit,data_dir,planet_name):
    # show what the raw uncertainites are on the location of the planet
    myfile = open(data_dir + "pyklip/" + planet_name + "_astrometry.txt",'w+')
    myfile.write(planet_name + ' astrometry\n')
    myfile.write("All errors are 1 sigma uncertainties\n")
    myfile.write("Px RA offset [px]: {0:.3f} +/- {1:.3f}\n".format(fit.raw_RA_offset.bestfit, fit.raw_RA_offset.error))
    myfile.write("Px Dec offset [px]: {0:.3f} +/- {1:.3f}\n".format(fit.raw_Dec_offset.bestfit, fit.raw_Dec_offset.error))

    # Full error budget included
    myfile.write("RA offset [mas]: {0:.3f} +/- {1:.3f}\n".format(fit.RA_offset.bestfit, fit.RA_offset.error))
    myfile.write("Dec offset [mas]: {0:.3f} +/- {1:.3f}\n".format(fit.Dec_offset.bestfit, fit.Dec_offset.error))

    # Propogate errors into separation and PA space
    myfile.write("Separation [mas]: {0:.3f} +/- {1:.3f}\n".format(fit.sep.bestfit, fit.sep.error))
    myfile.write("PA [deg]: {0:.3f} +/- {1:.3f}\n".format(fit.PA.bestfit, fit.PA.error))
    myfile.close()

def read_astrometry(data_dir,planet_name):
    myfile = open(data_dir + "pyklip/" + planet_name + "_astrometry.txt")
    astro_dict = {}
    for i,line in enumerate(myfile):
        print(i,line)
        if i <= 1:
            continue
        if line =="":
            continue
        key = line.split(':')[0]
        best_fit = float(line.split(':')[1].split('+')[0].strip())
        error = float(line.split('+/-')[1].strip())
        astro_dict[key] = (best_fit,error)
    return astro_dict