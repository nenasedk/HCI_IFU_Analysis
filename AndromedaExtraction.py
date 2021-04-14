# Author: Evert Nasedkin
# email: nasedkinevert@gmail.com

import sys,os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import glob
import shutil
import argparse

# Weird matplotlib imports for cluster use
import matplotlib
matplotlib.use('Agg') # set the backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib import rcParams
from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Computer Modern']},size = 24)
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)  

from astropy.io import fits
from scipy.ndimage import rotate
from photutils import CircularAperture, CircularAnnulus, aperture_photometry

import pyklip.instruments.GPI as GPI
import vip_hci as vip
from hciplot import plot_frames, plot_cubes
from vip_hci.preproc import cube_recenter_2dfit, cube_recenter_dft_upsampling, cube_shift, cube_crop_frames
from vip_hci.negfc import firstguess, mcmc_negfc_sampling, show_corner_plot, show_walk_plot,confidence
from vip_hci.andromeda import andromeda

from Astrometry import get_astrometry, read_astrometry, init_sphere, init_gpi, init_psfs

# There is a preprocessing function to help sort everything into the correct formats (GPI and SPHERE)
data_dir = "/u/nnas/data/HR8799/HR8799_AG_reduced/GPIK2/" #SPHERE-0101C0315A-20/channels/

# Instrument name, and optionally the band (ie GPIH, SPHEREYJ)
instrument = "GPI"
planet_name = "HR8799e" # Name to give to all outputs
distance = 41.2925 #pc

numthreads = 10
pixscale = 0.00746
fwhm = 3.5

DIT_SCIENCE = 64.0 # Set with argparse
DIT_FLUX = 4.0 # Set with argparse
NORMFACTOR = 1.0 # updated based on instrument and/or DITS
CENTER = (0,0)

def main(args):
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
    parser.add_argument("-c","--cont", action='store_true',required=False)
    args = parser.parse_args(args)

    data_dir = args.path
    instrument = args.instrument
    planet_name = args.name
    guesssep, guesspa = args.posn
    guessflux = 5e-5
    if args.ditscience is not None:
        DIT_SCIENCE = args.ditscience
    if args.ditflux is not None:
        DIT_FLUX = args.ditflux

    if not data_dir.endswith("/"):
        data_dir += "/"
    if not os.path.isdir(data_dir + "andromeda"):
        os.makedirs(data_dir + "andromeda", exist_ok=True)

    stellar_model = np.genfromtxt("/u/nnas/data/HR8799/stellar_model/hr8799_star_spec_" + instrument.upper() + "_fullfit_10pc.dat").T

    science,angles,wlen,psfs = init()

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
    posn = (-1*posn_dict["Px RA offset [px]"][0] +CENTER[0], -1*posn_dict["Px Dec offset [px]"][0]+CENTER[1])
    contrasts, snrs = run_andromeda(science,angles,wlen,psfs)
    aperture_phot_extract(contrasts,posn)
    return

def init():
    global pixscale
    global CENTER
    if "sphere" in instrument.lower():
        science_name = "frames_removed.fits"
        if os.path.isfile(data_dir + "psf_satellites_calibrated.fits"):
            psf_name= "psf_satellites_calibrated.fits"
        else:
            psf_name = "psf_cube.fits"        
        # sanity check on wlen units
        hdul_w = fits.open(data_dir + "wavelength.fits")
        if np.mean(hdul_w[0].data)>100.:
            hdu_wlen = fits.PrimaryHDU([hdul_w[0].data/1000])
            hdul_wlen = fits.HDUList([hdu_wlen])
            hdul_wlen.writeto(data_dir + "wavelength.fits",overwrite=True)        
        parang_name = "parang_removed.fits"
        wlen_name = "wavelength.fits"

        pixscale = 0.00746
        # Science Data
        hdul = fits.open(data_dir + science_name)
        science = hdul[0].data
        hdul.close()
        CENTER = (science.shape[-2]/2.0,science.shape[-1]/2.0)

        # Parangs
        ang_hdul = fits.open(data_dir + parang_name)
        angles = ang_hdul[0].data
        ang_hdul.close()
        # Wavelength
        wvs_hdul = fits.open(data_dir + wlen_name)
        wlen = wvs_hdul[0].data
        wvs_hdul.close()
        # PSF Data
        psf_hdul = fits.open(data_dir + psf_name)
        psfs = psf_hdul[0].data
        psf_hdul.close()

    elif "gpi" in instrument.lower():
        pixscale =  0.014161
        science_name = "*distorcorr.fits"
        psf_name = glob.glob(data_dir + "*-original_PSF_cube.fits")[0]
        psf_hdul = fits.open(psf_name)
        psfs = psf_hdul[0].data

        # Filelist MUST be sorted for PAs and frames to be in correct order for pynpoint
        # Assuming standard GPI naming scheme
        filelist = sorted(glob.glob(data_dir +science_name))
        dataset = GPI.GPIData(filelist, highpass=True, PSF_cube = psfs,recalc_centers=True)
        dataset.generate_psf_cube(30)
        psfs = dataset.psfs
        band = dataset.prihdrs[0]['APODIZER'].split('_')[1]
        spot_to_star_ratio = dataset.spot_ratio[band]
        NORMFACTOR = spot_to_star_ratio
        CENTER = (np.mean(dataset.centers[:,0]),np.mean(dataset.centers[:,1]))

        # Need to order the GPI data for pynpoint
        shape = dataset.input.shape
        science = dataset.input.reshape(len(filelist),37,shape[-2],shape[-1])
        science = np.swapaxes(science,0,1)
        science_pyn = []
        for channel,frame in enumerate(science[:]):
            # The PSF center isn't aligned with the image center, so let's fix that
            centx = dataset.centers.reshape(len(filelist),37,2)[:,channel,0]
            centy = dataset.centers.reshape(len(filelist),37,2)[:,channel,1]
            shiftx,shifty = (int((frame.shape[-2]/2))*np.ones_like(centx) - centx,
                             (int(frame.shape[-1]/2))*np.ones_like(centy) - centy)
            shifted = vip.preproc.recentering.cube_shift(frame,shiftx,shifty)
            # Save for a full file, not channel by channel
            science_pyn.append(shifted)
        # Save the full file (wlens,nframes,x,y)
        hdu = fits.PrimaryHDU(np.array(science_pyn))
        header_hdul = fits.open(filelist[0])
        hdu.header = header_hdul[0].header
        hdu.header.update(header_hdul[1].header)
        hdu.header['ESO ADA POSANG'] = (dataset.PAs.reshape(len(filelist),37)[:,0][0]+ 180.0)
        hdu.header['ESO ADA POSANG END'] = (dataset.PAs.reshape(len(filelist),37)[:,0][-1]+ 180.0 )
        hdul_new = fits.HDUList([hdu])
        hdul_new.writeto(data_dir + "HR8799_"+instrument + 'pyklip_frames_removed.fits', overwrite = True,
                         checksum=True,output_verify='exception')
        header_hdul.close()
        
        # Save wavelengths
        hdu = fits.PrimaryHDU(dataset.wvs[:37])
        hdul_new = fits.HDUList([hdu])
        hdul_new.writeto(data_dir + "wavelength.fits",overwrite = True)
        wlen = dataset.wvs[:37]
        # pyklip does weird things with the PAs, so let's fix that.
        # Keep or remove dataset.ifs_rotation? GPI IFS is rotated 23.5 deg, 
        angles = (dataset.PAs.reshape(len(filelist),37)[:,0] + 180.0)
        hdu = fits.PrimaryHDU(angles)
        hdul_new = fits.HDUList([hdu])
        hdul_new.writeto(data_dir + "parangs.fits",overwrite = True)
        del dataset
    set_fwhm(psfs,0)
    return science, angles, wlen, psfs

# Get the PSF FWHM for each channel
def set_fwhm(psfs,channel):
    global fwhm
    if len(psfs.shape) ==4 :
        fwhm_fit = vip.var.fit_2dgaussian(psfs[int(channel),0], crop=True, cropsize=8, debug=False)
    else:
        fwhm_fit = vip.var.fit_2dgaussian(psfs[int(channel)], crop=True, cropsize=8, debug=False)

    fwhm = np.mean(np.array([fwhm_fit['fwhm_y'],fwhm_fit['fwhm_x']]))*pixscale # fit for fwhm
    return

def run_andromeda(data,angles,wlen,psfs):
    global pixscale
    if "sphere" in instrument.lower():
        diam_tel = 8.3                                            # Telescope diameter [m]
        pixscale = 7.46                                           # Pixscale [mas/px]
    else:
        diam_tel = 10.                                             # Telescope diameter [m]
        pixscale =  0.014161 *1000                                 # Pixscale [mas/px]
    #PIXSCALE_NYQUIST = (1/2.*np.mean(wlen)*1e-6/diam_tel)/np.pi*180*3600*1e3 # Pixscale at Shannon [mas/px]
    #oversampling = PIXSCALE_NYQUIST /  pixscale                # Oversampling factor [1]
    contrasts = []
    snrs = []
    c_norms = []
    stds = []
    std_norms = []
    output_dir = data_dir + "andromeda/"
    # Iterate through wavelengths
    # data must be (wlen, time, x, y)
    #psfs = cube_crop_frames(psfs,10)
    np.save(output_dir + instrument+ "_"+ planet_name +"_psfs",psfs)

    for i,stack in enumerate(data):
        set_fwhm(psfs,i)
        if len(psfs.shape)==4:
            size = psfs.shape[2]
            psfs = np.mean(psfs,axis=1)
            psf, shy1, shx1 = cube_recenter_2dfit(psfs[i,:,int(size/2-11):int(size/2+11),int(size/2-11):int(size/2+11)], 
                                    xy=(11,11), fwhm=fwhm, nproc=1, subi_size=6, 
                                    model='gauss', negative=False, full_output=True, debug=False,plot=False)

            cube, shy1, shx1 = cube_recenter_2dfit(stack[:,:-1,:-1], 
                                               xy=(int(CENTER[0]),int(CENTER[1])), fwhm=fwhm, nproc=1, subi_size=6, 
                                               model='gauss', negative=False, full_output=True, debug=False,plot=False)                        
            psf = np.mean(psf,axis=1)

        else:
            psf = psfs[i]
            cube = cube_shift(stack[:,:-1,:-1],
                                -0.5,
                                -0.5)
                                #(stack.shape[-1]/2.0 - CENTER[1])-0.5,
                                #(stack.shape[-2]/2.0 - CENTER[0])-0.5)
        ang = angles
        if "sphere" in instrument.lower():
            ang = -1*angles

        PIXSCALE_NYQUIST = (1/2.*wlen[i]*1e-6/diam_tel)*180*3600*1e3/np.pi # Pixscale at Shannon [mas/px]
        oversampling = PIXSCALE_NYQUIST /  pixscale                # Oversampling factor [1]
        contrast,snr,snr_norm,std_contrast,std_contrast_norm,_,_ = andromeda(cube=np.nan_to_num(cube),
                                                                            oversampling_fact=oversampling,
                                                                            angles=ang, 
                                                                            psf=psf,
                                                                            filtering_fraction = 0.25,
                                                                            min_sep=0.5,
                                                                            iwa=4.0,
                                                                            annuli_width = 1.0,
                                                                            owa=40.,
                                                                            opt_method='no',
                                                                            fast=False,
                                                                            nproc=numthreads,
                                                                            homogeneous_variance=True,
                                                                            ditimg = DIT_SCIENCE,
                                                                            ditpsf = DIT_FLUX,
                                                                            verbose = True)
        contrasts.append(contrast)
        snrs.append(snr)
        stds.append(std_contrast)
        c_norms.append(snr_norm)
        std_norms.append(std_contrast_norm)
    contrasts = np.array(contrasts)
    snrs = np.array(snrs)
    stds = np.array(stds)
    c_norms = np.array(c_norms)
    std_norms = np.array(std_norms)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True) 
    np.save(output_dir + instrument+ "_"+ planet_name +"_contrastmap",contrasts)
    np.save(output_dir + instrument+ "_"+ planet_name +"_snr",snrs)
    np.save(output_dir + instrument+ "_"+ planet_name +"_stds",stds)
    np.save(output_dir + instrument+ "_"+ planet_name +"_c_norm",c_norms)
    np.save(output_dir + instrument+ "_"+ planet_name +"_std_norm",std_norms)

    hdu = fits.PrimaryHDU(contrasts)
    hdul_new = fits.HDUList([hdu])
    hdul_new.writeto(output_dir + instrument+ "_"+ planet_name + "_residuals.fits",overwrite=True)

    hdu = fits.PrimaryHDU(c_norms)
    hdul_new = fits.HDUList([hdu])
    hdul_new.writeto(output_dir + instrument+ "_"+ planet_name + "_normed.fits",overwrite=True)
    hdu = fits.PrimaryHDU(snrs)
    hdul_new = fits.HDUList([hdu])
    hdul_new.writeto(output_dir + instrument+ "_"+ planet_name + "_snrs.fits",overwrite=True)

    return contrasts, snrs

def aperture_phot_extract(contrasts,posn):
    mask = create_circular_mask(contrasts.shape[1],contrasts.shape[2],
                            center = posn, 
                            radius = 3)
    aperture = CircularAperture([posn], r=3)
    spectrum = []
    for frame in contrasts:
        phot_table = aperture_photometry(frame,aperture)
        spectrum.append(phot_table['aperture_sum'][0])
    spectrum = np.array(spectrum)
    np.save(data_dir + "andromeda/"+instrument + "_" + planet_name + "contrast",spectrum)
    return spectrum

def guess_flux(cube,posn):
    rs = [] #radius (separation)
    ts = [] #theta (position angle)
    fs = [] #flux
    output_dir = data_dir + "andromeda/"
    rs = [],
    ts = []
    fs = []
    for i,frame in enumerate(cube):
        PIXSCALE_NYQUIST = (1/2.*wlen[37]*1e-6/diam_tel)/np.pi*180*3600*1e3 # Pixscale at Shannon [mas/px]
        oversampling = PIXSCALE_NYQUIST /  pixscale                # Oversampling factor [1]
        print(oversampling)
        psf, shy1, shx1 = cube_recenter_2dfit(psfs[37,:,int(255/2-6):int(255/2+6),int(255/2-6):int(255/2+6)], 
                                            xy=(6,6), fwhm=fwhm, nproc=1, subi_size=6, 
                                            model='gauss', negative=False, 
                                            full_output=True, debug=False,plot=False)
        print(psf.shape)
        psf = np.median(psf,axis=0)
        psf = vip.metrics.normalize_psf(psf, fwhm, size=11)
        plot_frames(psf, grid=True, size_factor=4)

        r_0, theta_0, f_0 = firstguess(frame, angles, psf, ncomp=30, plsc=pxscale,
                                    planets_xy_coord=[(159, 109)], fwhm=fwhm, 
                                    f_range=None, annulus_width=3, aperture_radius=3,
                                    simplex=True, plot=True, verbose=True)
        rs.append(r_0)
        ts.append(theta_0)
        fs.append(f_0)
    rs = np.array(rs)
    ts = np.array(ts)
    fs = np.array(fs)
    np.save(output_dir + extraction_name + "_astrometry",np.array([rs,ts,fs]))

def mcmc_flux(snrs,rs,ts,fs):
    nwalkers, itermin, itermax = (100, 200, 500)
    maxs = [],
    confs = []
    for i,frame in enumerate(snrs):
        init = np.array([rs[i],ts[i],fs[i]]) #r,theta,flux
        psf = np.median(psf[i],axis=0)
        chain = mcmc_negfc_sampling(frame, angles, psf, ncomp=30, plsc=pixscale/1000,                                
                                    fwhm=fwhm, svd_mode='lapack', annulus_width=3, 
                                    aperture_radius=3, initial_state=init, nwalkers=nwalkers, 
                                    bounds=None, niteration_min=itermin, rhat_count_threshold=1,
                                    niteration_limit=itermax, check_maxgap=50, nproc=numthreads,
                                    display=False, verbosity=1, save=False)
        burnin = 0.3
        isamples_flat = chain[:, int(chain.shape[1]//(1/burnin)):, :].reshape((-1,3))

        val_max, conf = confidence(isamples_flat, cfd=68, gaussian_fit=True, 
                                verbose=True, save=False, title='fake planet')
        
        maxs.append(val_max)
        confs.append(conf)
    maxs = np.array(maxs)
    confs = np.array(conf)
    np.save(output_dir + extraction_name + "_best_fit",maxs)
    np.save(output_dir + extraction_name + "_intervals",confs)

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center >= radius
    return mask

if __name__ == '__main__':
    main(sys.argv[1:])