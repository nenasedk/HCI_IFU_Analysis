import sys,os
import shutil
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib import rcParams
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern']},size = 24)
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)  

from astropy.io import fits
from scipy.ndimage import rotate
from photutils import CircularAperture, CircularAnnulus, aperture_photometry
from typing import List, Optional, Tuple
from Astrometry import get_astrometry, read_astrometry, init_sphere, init_gpi, ser_psfs
from vip_hci.preproc import cube_recenter_2dfit, cube_recenter_dft_upsampling
import argparse
import pyklip.instruments.GPI as GPI

from pynpoint import Pypeline, \
                     FitsReadingModule,\
                     ParangReadingModule,\
                     Hdf5ReadingModule, \
                     PSFpreparationModule, \
                     PcaPsfSubtractionModule,\
                     AttributeReadingModule, \
                     DerotateAndStackModule, \
                     BadPixelSigmaFilterModule,\
                     ReplaceBadPixelsModule,\
                     FrameSelectionModule,\
                     WaffleCenteringModule,\
                     CropImagesModule,\
                     SimplexMinimizationModule,\
                     StarAlignmentModule,\
                     ShiftImagesModule

data_dir = "/u/nnas/data/HR8799/HR8799_AG_reduced/GPIK2/" #SPHERE-0101C0315A-20/channels/
instrument = "GPI"
planet_name = "HR8799e"

distance = 41.2925 #pc
pcas = [3,4,5,8,10,12,15,20]
fwhm = 3.5*0.01414 #0.134
pixscale = 0.00746
numthreads = 35
DIT_SCIENCE = 64.0
DIT_FLUX = 4.0

def main(args):
    sys.path.append(os.getcwd())

    global data_dir 
    global instrument
    global planet_name
    global DIT_SCIENCE
    global DIT_FLUX
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default= "/u/nnas/data/", required=True)
    parser.add_argument("instrument", type=str, default= "GPI", required=True)
    parser.add_argument("name", type=str, default= "HR8799", required=True)   
    parser.add_argument("posn", type=int, nargs = "+", required=True)
    parser.add_argument("-ds","--ditscience", type=float, required=False)
    parser.add_argument("-df","--ditflux", type=float, required=False)
    args = parser.parse_args(args)

    data_dir = args.path
    instrument = args.instrument
    planet_name = args.name
    guesssep, guesspa, guessflux = args.posn
    base_name + "HR8799_" + instrument
    if args.ditscience is not None:
        DIT_SCIENCE = args.ditscience
    if args.ditflux is not None:
        DIT_FLUX = args.ditflux
    if not data_dir.ends_with("/"):
        data_dir += "/"

    if "sphere" in instrument.lower():
        nChannels = 39
        pixscale = 0.00746
        shutil.copy("config/Pynpoint_config_SPHERE.ini",data_dir + "PynPoint_config.ini")
    elif "gpi" in instrument.lower():
        nChannels = 37
        pixscale = 0.0162
        shutil.copy("config/Pynpoint_config_GPI.ini",data_dir + "PynPoint_config.ini")

    if not os.path.isdir(data_dir + "pynpoint/"):
        os.mkdir(data_dir + "pynpoint")
    if not os.path.exjsts(data_dir + "pyklip/"+ planet_name + "_astrometry.txt"):
        if "gpi" in instrument.lower():
            dataset = init_gpi(data_dir)
        elif "sphere" in instrument.lower():
            dataset = init_sphere(data_dir)
        PSF_cube,cal_cube,spot_to_star_ratio = init_psfs(dataset)
        posn = get_astrometry(dataset, PSF_cube, guesssep, guesspa, guessflux,data_dir,planet_name)
    else: 
        posn_dict = read_astrometry(data_dir,planet_name)
        posn = (posn_dict["Px RA offset [px]"], posn_dict["Px DEC offset [px]"])

    set_fwhm()
    data_shape = preproc_files()
    run_all_channels(nChannels,
                 base_name,
                 instrument + "_" + planet_name,
                 (posn[0] + data_shape[2],posn[1]+data_shape[3]))
    contrasts = save_contrasts(nChannels,
                            base_name,
                            data_dir + "pynpoint/",
                            instrument + "_" + planet_name)
    save_flux(contrasts)



def save_residuals(residuals,name,output_place):   
    hdu = fits.PrimaryHDU(residuals)
    hdul = fits.HDUList([hdu])
    hdul.writeto(output_place + name + '.fits',overwrite = True)

def simplex_one_channel(channel,input_name,psf_name,output_name,posn,waffle = False):
    reshape_psf(data_dir + psf_name)

    pipeline = Pypeline(working_place_in=data_dir,
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
    module = ParangReadingModule(file_name="parangs.txt",
                                 input_dir=data_dir,
                                 name_in="parang",
                                 data_tag = 'science',
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
    if "gpi" in instrument.lower():
        # Not really sure why this is here...
        module = ShiftImagesModule(name_in = "center",
                                   image_in_tag = "psf",
                                   image_out_tag = "psf_centered",
                                   shift_xy = (0.5,0.5),
                                   interpolation = 'spline')
        # Let's not use this for now
        # pipeline.add_module(module)

    module = SimplexMinimizationModule(name_in = 'simplex',
                                       image_in_tag = 'science_bad',
                                       psf_in_tag = 'psf',
                                       res_out_tag = 'flux_channel_' + str(channel)+"_",
                                       flux_position_tag = 'flux_pos_channel_' + str(channel) +"_",
                                       position = posn,
                                       magnitude = 16.0, # approximate planet contrast in mag
                                       psf_scaling = -1, # deal with spectrum mormalization later
                                       merit = 'gaussian', #better than hessian
                                       aperture = 0.07, # in arcsec
                                       tolerance = 0.005, # tighter tolerance is good
                                       pca_number = pcas, #listed above
                                       cent_size = 0.15, # how much to block out 
                                       offset = 0.0) #use fixed astrometry from KLIP

    pipeline.add_module(module)
    pipeline.run()
    for pca in pcas:
        flux = pipeline.get_data('flux_pos_channel_' + str(channel) + "_" + str(pca).zfill(3))
        np.savetxt(output_place + output_name + "_ch" + str(channel) +"_flux_pos_out_pca_" +str(pca)+ ".dat",flux)
        residuals = pipeline.get_data('flux_channel_' + channel + "_" + str(pca).zfill(3))
        save_residuals(residuals[-1],output_name +"_residuals_" + channel + "_pca_" + str(pca), output_place)


def run_all_channels(nchannels, base_name, output_name, posn):
    for channel in range(nchannels):
        working_dir = data_dir + "CH" + str(channel) 
        if not os.path.isdir(working_dir):
            os.mkdir(working_dir)
        name = base_name +"_" + str(channel) + '_reduced.fits'
        psf_name = base_name +"_" + str(channel) + '_PSF.fits'
        output_place = data_dir+"pynpoint/"
        if os.path.isfile(working_dir + "/PynPoint_database.hdf5"):
            os.remove(working_dir + "/PynPoint_database.hdf5")
        simplex_one_channel(str(channel),name,psf_name,output_name,posn)

    residuals = []
    for pca in pcas:
        rpcas = []
        for channel in range(nchannels):
            hdul = fits.open(data_dir + "pynpoint/"+output_name+"_residuals_" + channel + "_pca_" + str(pca)+".fits")
            data = hdul[0].data
            rpcas.append(data)
            hdul.close()
        rpcas = np.array(rpcas)
        residuals.append(rpcas)
    hdu = fits.PrimaryHDU(residuals)
    hdul = fits.HDUList([hdu])
    hdul.writeto(data_dir+"pynpoint/" + instrument+ "_"+ planet_name + '_residuals.fits',overwrite = True)


def save_contrasts(nchannels,base_name,output_place,output_name):
    whdul = fits.open(data_dir + "wavelength.fits")
    wlen = whdul[0].data/1000
    print(wlen)
    contrasts = []
    for pca in pcas:
        contrast = []
        for channel in range(nchannels):
            samples = np.genfromtxt(data_dir + "CH"+str(channel)+"/"+base_name + "flux_pos_out_pca"+str(pca)+ ".dat")
            contrast.append(samples[-1][4])
        contrasts.append(contrast)
    cont = np.array(contrasts)
    np.save(output_place + output_name +  "contrasts",cont)
    return cont

def save_flux(contrasts):
    whdul = fits.open(data_dir + "wavelength.fits")
    wlen = whdul[0].data/1000
    stellar_model = np.genfromtxt("/Users/nasedkin/data/HR8799/hr8799_star_spec_"+ instrument +"_fullfit_10pc.dat").T
    fluxes = []
    for i in range(len(pcas)):
        fluxes.append(stellar_model[1]*10**((contrasts[i])/-2.5)* DIT_SCIENCE/DIT_FLUX)
    fluxes = np.array(fluxes)
    np.save(data_dir + "pynpoint/" + instrument + "_" + planet_name + "_flux",fluxes)

def set_fwhm():
    if "sphere" in instrument.lower():
        science_name = "frames_removed.fits"
        parang_name = "parang_removed.fits"
        psf_name = "psf_satellites_calibrated.fits"
        wlen_name = "wvs_micron.fits"
    hdul = fits.open(data_dir + psf_name)
    psfs = hdul[0].data
    global fwhm
    fwhm_fit = vip.var.fit_2dgaussian(psfs[0,0], crop=True, cropsize=11, debug=False)
    fwhm = np.mean(np.array([fwhm_fit['fwhm_y'],fwhm_fit['fwhm_x']]))*pixscale # fit for fwhm
    hdul.close()
    return

def preproc_files():
    if os.path.exists(data_dir + "HR8799_"+instrument+"_" + str(channel) + '_reduced.fits'):
        return
    data_shape = None
    if "sphere" in instrument.lower():
        science_name = "frames_removed.fits"
        psf_name = "psf_satellites_calibrated.fits"

        hdul = fits.open(data_dir + science_name)
        cube = hdul[0].data
        if data_shape is None:
            data_shape = cube.shape
        for channel,frame in enumerate(cube[:]):
            hdu = fits.PrimaryHDU(frame)
            hdul_new = fits.HDUList([hdu])
            hdul_new.writeto(data_dir + "HR8799_"+instrument+"_" + str(channel) + '_reduced.fits',overwrite=True)
        hdul.close()
        hdul = fits.open(data_dir + psf_name)
        cube = hdul[0].data
        for channel,frame in enumerate(cube[:]):
            hdu = fits.PrimaryHDU(frame)
            hdul_new = fits.HDUList([hdu])
            hdul_new.writeto(data_dir + "HR8799_"+instrument+"_" + str(channel) + '_PSF.fits',overwrite = True)
    elif "gpi" in instrument.lower():
        science_name = "*distorcorr.fits"
        psf_name = "*-original_PSF_cube.fits"
        psfs = fits.open(data_dir + psf_name)[0].data

        filelist = glob.glob(data_dir +science_name)
        dataset = GPI.GPIData(filelist, highpass=False, PSF_cube = psf)

        # Need to order the GPI data for pynpoint
        shape = dataset.input.shape
        science = dataset.input.reshape(37,len(filelist),shape[1],shape[2])
        if data_shape is None:
            data_shape = science.shape
        for channel,frame in enumerate(science[:]):
            hdu = fits.PrimaryHDU(frame)
            hdul_new = fits.HDUList([hdu])
            hdul_new.writeto(data_dir + "HR8799_"+instrument+"_" + str(channel) + '_reduced.fits',overwrite=True)
        for channel,frame in enumerate(psfs):
            hdu = fits.PrimaryHDU(frame)
            hdul_new = fits.HDUList([hdu])
            hdul_new.writeto(data_dir + "HR8799_"+instrument+"_" + str(channel) + '_PSF.fits',overwrite = True)
    hdul.close()
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
        hdul_new.writeto(output_place + output_name +"_" + str(channel) + '_PSF.fits',
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

if __name__ == '__main__':
    main(sys.argv[1:])