import sys,os
import argparse

import numpy as np
from astropy.io import fits
from photutils import CircularAperture, CircularAnnulus, aperture_photometry
data_dir = "/u/nnas/data/HR8799/HR8799_AG_reduced/GPIK2/" #SPHERE-0101C0315A-20/channels/
distance = 41.2925 #pc
instrument = "GPI"
planet_name = "HR8799e"
pxscale = 0.00746

def __main__(args):
    sys.path.append(os.getcwd())

    global data_dir 
    global instrument
    global planet_name
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default= "/u/nnas/data/", required=True)
    parser.add_argument("instrument", type=str, default= "GPI", required=True)
    parser.add_argument("name", type=str, default= "HR8799", required=True)   
    args = parser.parse_args(args)

    data_dir = args.path
    instrument = args.instrument
    planet_name = args.name
    posn_dict = read_astrometry(data_dir,planet_name)
    if os.path.exists(data_dir + instrument + "_" + planet_name + "_residuals.npy"):
        contrast = np.load(data_dir + instrument + "_" + planet_name + "_residuals.npy")
    elif os.path.exists(data_dir + instrument + "_" + planet_name + "_residuals.fits"):
        hdul = fits.load(data_dir + instrument + "_" + planet_name + "_residuals.fits")
        contrast = hdul[0].data
        hdul.close()
    else:
        print("No residual file found!")
        return
        
    if len(contrast.shape)==4:
        npca = contrast.shape[0]
        nl = contrast.shape[1]
        get_covariance(contrast,posn_dict,nl,npca)
    else:
        nl = contrast.shape[0]
        get_covariance(contrast,posn_dict,nl)
    return

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center >= radius
    return mask

def get_covariance(contrast,posn_dict,nl,npca=None):
    if npca is not None:
        cors = []
        for j in range(npca):
            fluxes = []
            for i in range(nl):
                # Stack and median subtract
                med = contrast[j,i]
                # Mask out planet (not sure if correct location)
                mask = create_circular_mask(contrast.shape[2],contrast.shape[3],
                                            center = (posn_dict["Px RA offset [px]"], posn_dict["Px DEC offset [px]"]),
                                            radius = posn_dict["Separation [mas]"]/1000*pxscale)
                
                # Get fluxes of all pixels within annulus
                annulus = annulus_c_list[i].to_mask(method='center')[0]
                flux = annulus.multiply(mask*med)[annulus.data>0]
                fluxes.append(fluxes)
            fluxes = np.array(fluxes)
            # Compute correlation, Eqn 1 Samland et al 2017, https://arxiv.org/pdf/1704.02987.pdf
            cor = np.zeros((fluxes.shape[0],fluxes.shape[0]))
            for m in range(fluxes.shape[0]):
                for n in range(fluxes.shape[0]):
                    top =    np.mean(np.dot(fluxes[m],fluxes[n]))
                    bottom = np.sqrt(np.mean(np.dot(fluxes[m],fluxes[m]))*np.mean(np.dot(fluxes[n],fluxes[n])))
                    #print(top,bottom)
                    cor[i,j] = top/bottom
            cors.append(cor)
        cors = np.array(cors)
        np.save(data_dir + instrument + "_" + planet_name + "_covariance",cors)
        return cors
    else:
        fluxes = []
        for i in range(nl):
            # Stack and median subtract
            med = contrast[i]
            # Mask out planet (not sure if correct location)
            mask = create_circular_mask(contrast.shape[1],contrast.shape[2],
                                        center = (posn_dict["Px RA offset [px]"], posn_dict["Px DEC offset [px]"]),
                                        radius = posn_dict["Separation [mas]"]/1000*pxscale)
            
            # Get fluxes of all pixels within annulus
            annulus = annulus_c_list[i].to_mask(method='center')[0]
            flux = annulus.multiply(mask*med)[annulus.data>0]
            fluxes.append(fluxes)
        fluxes = np.array(fluxes)
        cor = np.zeros((fluxes.shape[0],fluxes.shape[0]))
        for m in range(fluxes.shape[0]):
            for n in range(fluxes.shape[0]):
                top =    np.mean(np.dot(fluxes[m],fluxes[n]))
                bottom = np.sqrt(np.mean(np.dot(fluxes[m],fluxes[m]))*np.mean(np.dot(fluxes[n],fluxes[n])))
                #print(top,bottom)
                cor[i,j] = top/bottom
        np.save(data_dir + instrument + "_" + planet_name + "_covariance",cor)
        return cor

def normalize_errors(contrasts):
    # load in full residual file
    # use annulus far from center
    # get std of each aperture in contrast units
    # load in stellar spectra, convert to flux units
    # load in raw data file
    # extract photometry from sat spots, or from OBJECT/FLUX
    # extract from background, compute SNR
    # add errors in quadrature
    # save as array with dims (npca,nl)
    
    return