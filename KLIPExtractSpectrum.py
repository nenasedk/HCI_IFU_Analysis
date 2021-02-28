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
from astropy.io import fits
from Astrometry import get_astrometry, read_astrometry
import spectres

import argparse

### KLIP Parameters ###
numbasis = np.array([2,3,4,5,8,10,12,15]) # "k_klip", this can be a list of any size.
maxnumbasis = 20 # Max components to be calculated
movement = 2.0 # aggressiveness for choosing reference library
stamp_size = 11 # how big of a stamp around the companion in pixels
                # stamp will be stamp_size**2 pixels
sections = 10
distance = 41.2925 #pc
data_dir = "/u/nnas/data/"
instrument = "GPI"
planet_name = "HR8799"
numthreads = 11
pxscale = 0.01422

def main(args):
    sys.path.append(os.getcwd())

    global data_dir 
    global instrument
    global planet_name
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default= "/u/nnas/data/")
    parser.add_argument("instrument", type=str, default= "GPI")
    parser.add_argument("name", type=str, default= "HR8799")   
    parser.add_argument("posn", type=float, nargs = "+")
    args = parser.parse_args(args)

    data_dir = args.path
    instrument = args.instrument
    planet_name = args.name
    guesssep, guesspa = args.posn
    guessflux = 5e-5
    if not data_dir.endswith("/"):
        data_dir += "/"
    if not os.path.isdir(data_dir + "pyklip"):
        os.makedirs(data_dir + "pyklip", exist_ok=True)

    stellar_model = np.genfromtxt("/u/nnas/data/HR8799/stellar_model/hr8799_star_spec_" + instrument.upper() + "_fullfit_10pc.dat").T

    if "gpi" in instrument.lower():
        dataset = init_gpi()
        pxscale = 0.0162
    elif "sphere" in instrument.lower():
        dataset = init_sphere()
        pxscale = 0.007462
    PSF_cube,cal_cube,spot_to_star_ratio = init_psfs(dataset)
    if not os.path.exists(data_dir + "pyklip/"+ planet_name + "_astrometry.txt"):
        posn = get_astrometry(dataset, PSF_cube, guesssep, guesspa, guessflux, data_dir, planet_name)
    else: 
        posn_dict = read_astrometry(data_dir,planet_name)
        posn = (posn_dict["Separation [mas]"][0], posn_dict["PA [deg]"][0])
    exspect, fm_matrix = KLIP_Extraction(dataset, PSF_cube, posn, numthreads)
    contrasts,flux = get_spectrum(dataset, exspect, stellar_model, spot_to_star_ratio)
    return

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

    filelist = glob.glob(data_dir +"*distorcorr.fits")
    dataset = GPI.GPIData(filelist, highpass=False, PSF_cube = psf)
    return dataset

def init_psfs(dataset):
    dataset.generate_psfs(10)
    N_frames = len(dataset.input)
    N_cubes = np.size(np.unique(dataset.filenums))
    nl = N_frames // N_cubes
    # in this case model_psfs has shape (N_lambda, 20, 20)
    # The units of your model PSF are important, the return spectrum will be
    # relative to the input PSF model, see next example

    PSF_cube = dataset.psfs
    model_psf_sum = np.nansum(PSF_cube, axis=(1,2))
    model_psf_peak = np.nanmax(PSF_cube, axis=(1,2))

    # Now divide the sum by the peak for each wavelength slice
    aper_over_peak_ratio = model_psf_sum/model_psf_peak

    # star-to-spot calibration factor
    band = dataset.prihdrs[0]['APODIZER'].split('_')[1]
    spot_to_star_ratio = dataset.spot_ratio[band]

    spot_peak_spectrum = \
        np.median(dataset.spot_flux.reshape(len(dataset.spot_flux)//nl, nl), axis=0)
    calibfactor = np.array(aper_over_peak_ratio*spot_peak_spectrum / spot_to_star_ratio)
    # calibrated_PSF_model is the stellar flux in counts for each wavelength
    calibrated_PSF_model = calibfactor[:,None,None]*PSF_cube
    return PSF_cube, calibrated_PSF_model, spot_to_star_ratio

def KLIP_Extraction(dataset, PSF_cube, posn, numthreads):
    planet_sep, planet_pa = posn
    planet_sep =planet_sep/1000 / pxscale
    subsections = 10#[[(planet_pa+(4.0*i*stamp_size)-2.0*stamp_size)/180.*np.pi,\
                    #(planet_pa+(4.0*i*stamp_size)+2.0*stamp_size)/180.*np.pi] for i in range(sections)]
    #num_k_klip = len(numbasis) # how many k_klips running
    N_frames = len(dataset.input)
    N_cubes = np.size(np.unique(dataset.filenums))
    #nl = N_frames // N_cubes
    spectra_template = None #np.tile(np.array(exspect[i]),N_cubes)
    print(planet_sep,planet_pa)
    print(dataset.input.shape,numbasis,PSF_cube.shape)
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
                    fileprefix=instrument + "_" + planet_name + "_fmspect",
                    annuli=[[planet_sep-1.5*stamp_size,planet_sep+1.5*stamp_size]],
                    subsections=subsections,
                    movement=movement,
                    #flux_overlap = 0.1,
                    numbasis = numbasis,
                    maxnumbasis=maxnumbasis,
                    numthreads=numthreads,
                    spectrum=spectra_template,
                    #time_collapse = 'weighted-mean',
                    save_klipped=True,
                    highpass=True,
                    calibrate_flux=True,
                    outputdir=data_dir + "pyklip/")
    #klipped = dataset.fmout[:,:,-1,:]
    # If you want to scale your spectrum by a calibration factor:
    units = "natural"
    scaling_factor = 1.0
    exspect, fm_matrix = es.invert_spect_fmodel(dataset.fmout, dataset, units=units,
                                                scaling_factor=scaling_factor,
                                                method="leastsq")
    np.save(data_dir + "pyklip/exspect",exspect)
    np.save(data_dir + "pyklip/exspect",fm_matrix)
    return exspect, fm_matrix

def get_spectrum(dataset,exspect,spot_to_star_ratio,stellar_model):
    num_k_klip = len(numbasis) # how many k_klips running
    N_frames = len(dataset.input)
    N_cubes = np.size(np.unique(dataset.filenums))
    nl = N_frames // N_cubes
    wlen = dataset.wvs[:nl]

    # Contrast
    fig,ax = plt.subplots(figsize = (16,10))
    ax.set_xlabel("Wavelength [micron]")
    ax.set_ylabel("Contrast")
    ax.set_title(planet_name + " " + instrument + " Contrast KLIP")
    m_cont = np.mean(exspect[:],axis=0)
    for i in range(len(numbasis)):
        ax.plot(wlen,exspect[i]*spot_to_star_ratio,label=str(numbasis[i]),alpha=0.5)
    ax.plot(wlen,m_cont*spot_to_star_ratio,label = 'Mean',linewidth=4)
    ax.set_ylim(0.0,3e-5)
    plt.legend()
    plt.savefig(data_dir + "pyklip/" +planet_name + "_" + instrument +"contrasts_KLIP.pdf")

    plt.clf()
    fig,ax = plt.subplots(figsize = (16,10))

    for i in range(num_k_klip):
        ax.plot(wlen,exspect[i]*spot_to_star_ratio*stellar_model[1]*(distance/10.)**2,label=str(numbasis[i]),alpha=0.5)
    ax.plot(wlen,m_cont*spot_to_star_ratio*stellar_model[1]*(distance/10.)**2 ,label = 'Mean',linewidth=4)
    ax.set_xlabel("Wavelength [micron]")
    ax.set_ylabel(r"Flux Density [W/m$^{2}/\mu$m")
    ax.set_title(planet_name + " " + instrument + " Flux KLIP")
    plt.legend()
    #ax.set_ylim(0,1e-4)
    #ax.set_xlim(1.4,2.5)
    plt.savefig(data_dir + "pyklip/" +planet_name + "_" + instrument +"flux_KLIP.pdf")
    np.save(data_dir +"pyklip/"+ instrument + planet_name + "_contrasts",
            exspect*spot_to_star_ratio)
    np.save(data_dir +"pyklip/"+ instrument + planet_name + "_flux_10pc_7200K",
            exspect*spot_to_star_ratio*stellar_model[1]*(distance/10.)**2)
        
    # Contrast, Flux Density (W/m^2/micron)
    return exspect*spot_to_star_ratio,exspect*spot_to_star_ratio*stellar_model[1]*(distance/10.)**2

if __name__ == '__main__':
    main(sys.argv[1:])