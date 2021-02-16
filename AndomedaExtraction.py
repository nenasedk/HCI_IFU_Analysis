import sys,os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib import rcParams
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern']},size = 24)
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)  
import pyklip.instruments.GPI as GPI

from astropy.io import fits
from scipy.ndimage import rotate
from photutils import CircularAperture, CircularAnnulus, aperture_photometry
from Astrometry import get_astrometry, read_astrometry, init_sphere, init_gpi, ser_psfs

import vip_hci as vip
from hciplot import plot_frames, plot_cubes
from vip_hci.preproc import cube_recenter_2dfit, cube_recenter_dft_upsampling
from vip_hci.negfc import firstguess, mcmc_negfc_sampling, show_corner_plot, show_walk_plot,confidence
from vip_hci.andromeda import andromeda

import argparse

instrument = "GPI"
planet_name = "HR8799"
numthreads = 35
pixscale = 0.00746
fwhm = 3.5
def main(args):
    sys.path.append(os.getcwd())

    global data_dir 
    global instrument
    global planet_name

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default= "/u/nnas/data/", required=True)
    parser.add_argument("instrument", type=str, default= "GPI", required=True)
    parser.add_argument("name", type=str, default= "HR8799", required=True)   
    parser.add_argument("posn", type=int, nargs = "+", required=True)
    args = parser.parse_args(args)

    data_dir = args.path
    instrument = args.instrument
    planet_name = args.name
    guesssep, guesspa, guessflux = args.posn
    if not data_dir.ends_with("/"):
        data_dir += "/"
    if not os.path.isdir(data_dir + "andromeda"):
        os.makedirs(data_dir + "andromeda", exist_ok=True)

    stellar_model = np.genfromtxt("/u/nnas/data/data/HR8799/hr8799_star_spec_" + instrument.lower() + "_fullfit_10pc.dat").T

    science,angles,wlen,psfs = init()

    if not os.path.exists(data_dir + "pyklip/"+ planet_name + "_astrometry.txt"):
        if "gpi" in instrument.lower():
            dataset = init_gpi(data_dir)
        elif "sphere" in instrument.lower():
            dataset = init_sphere(data_dir)
        PSF_cube,cal_cube,spot_to_star_ratio = init_psfs(dataset)
        posn = get_astrometry(dataset, PSF_cube, guesssep, guesspa, guessflux,data_dir,planet_name)
    else: 
        posn_dict = read_astrometry(data_dir,planet_name)
        posn = (posn_dict["Px RA offset [px]"], posn_dict["Px DEC offset [px]"])
    contrasts, snrs = run_andromeda(science,angles,wlen,psfs)
    aperture_phot_extract(contrasts,posn)
    return

def init():
    if "sphere" in instrument.lower():
        science_name = "frames_removed.fits"
        parang_name = "parang_removed.fits"
        psf_name = "psf_satellites_calibrated.fits"
        wlen_name = "wvs_micron.fits"

        # Science Data
        hdul = fits.open(data_dir + science_name)
        science = hdul[0].data
        hdul.close()
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
        science_name = "*distorcorr.fits"
        psf_name = "*-original_PSF_cube.fits""
        psfs = fits.open(data_dir + psf_name)[0].data

        filelist = glob.glob(data_dir +science_name)
        dataset = GPI.GPIData(filelist, highpass=False, PSF_cube = psf)

        ###### Useful values based on dataset ######
        N_frames = len(dataset.input)
        N_cubes = np.size(np.unique(dataset.filenums))
        nl = N_frames // N_cubes

        wlen = dataset.wvs[:nl]
        science = dataset.input
        angles = dataset.PAs


    global fwhm
    fwhm_fit = vip.var.fit_2dgaussian(psfs[0,0], crop=True, cropsize=11, debug=False)
    fwhm = np.mean(np.array([fwhm_fit['fwhm_y'],fwhm_fit['fwhm_x']])) # fit for fwhm

    return science,angles,wlen,psfs

def run_andromeda(data,angles,wlen,psfs):
    global pixscale
    if "sphere" in instrument.lower():
        diam_tel = 8.3                                             # Telescope diameter [m]
        pixscale = 7.46                                            # Pixscale [mas/px]
    else:
        diam_tel = 8.                                             # Telescope diameter [m]
        pixscale = 14.22                                            # Pixscale [mas/px]
    PIXSCALE_NYQUIST = (1/2.*np.mean(wlen)*1e-6/diam_tel)/np.pi*180*3600*1e3 # Pixscale at Shannon [mas/px]
    oversampling = PIXSCALE_NYQUIST /  pixscale                # Oversampling factor [1]
    fwhm = 7*oversampling
    contrasts = []
    snrs = []
    c_norms = []
    stds = []
    std_norms = []
    output_dir = data_dir + "andromeda/"
    # Iterate through wavelengths
    # data must be (wlen, time, x, y)
    for i,stack in enumerate(data):
        if len(psfs.shape)==4:
            size = psfs.shape[2]
        else:
            size = psfs.shape[1]
        
        psf, shy1, shx1 = cube_recenter_2dfit(psfs[i,:,int(psfs.shape[size/2-11):int(size/2+11),int(size/2-11):int(size/2+11)], 
                                            xy=(6,6), fwhm=fwhm, nproc=1, subi_size=6, 
                                            model='gauss', negative=False, full_output=True, debug=False,plot=False)

        cube, shy1, shx1 = cube_recenter_2dfit(stack[:,:-1,:-1], 
                                               xy=(int(stack.shape[1]/2),int(stack.shape[2]/2)), fwhm=fwhm, nproc=1, subi_size=6, 
                                               model='gauss', negative=False, full_output=True, debug=False,plot=False)
        contrast,snr,snr_norm,std_contrast,std_contrast_norm,_,_ = andromeda(cube=np.nan_to_num(cube),
                                                                            oversampling_fact=oversampling,
                                                                            angles=-1*angles, 
                                                                            psf=np.median(psf,axis=0),
                                                                            filtering_fraction = 0.2,
                                                                            min_sep=2.0,
                                                                            iwa=2.0,
                                                                            annuli_width = 1.2,
                                                                            owa=None,
                                                                            opt_method='lsq',
                                                                            fast=True,
                                                                            nproc=1,
                                                                            homogeneous_variance=False,
                                                                            ditimg = 64.,
                                                                            ditpsf = 4.,
                                                                            verbose = False)
        print(contrast)
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
    np.save(output_dir + instrument+ "_"+ planet_name +"_contrast",contrasts)
    np.save(output_dir + instrument+ "_"+ planet_name +"_snr",snrs)
    np.save(output_dir + instrument+ "_"+ planet_name +"_stds",stds)
    np.save(output_dir + instrument+ "_"+ planet_name +"_c_norm",c_norms)
    np.save(output_dir + instrument+ "_"+ planet_name +"_std_norm",std_norms)

    hdu = fits.PrimaryHDU(contrasts)
    hdul_new = fits.HDUList([hdu])
    hdul_new.writeto(output_dir + instrument+ "_"+ planet_name + "_residuals.fits",overwrite=True)
    hdul.close()
    return contrasts, snrs

def aperture_phot_extract(contrasts,posn):
    mask = create_circular_mask(contrasts.shape[1],contrasts.shape[2],
                            center = posn, 
                            radius = 3*fwhm)
    aperture = CircularAperture([posn], r=3*fwhm)
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
        chain = mcmc_negfc_sampling(frame, angles, psf, ncomp=30, plsc=/1000,                                
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