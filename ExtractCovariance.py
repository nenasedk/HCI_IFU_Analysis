import sys,os
import argparse
import glob
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from astropy.io import fits
from photutils import CircularAperture, CircularAnnulus, aperture_photometry

# My own files
from Astrometry import get_astrometry, read_astrometry, init_sphere, init_gpi, set_psfs

# Exoplanet stuff
import pyklip.instruments.GPI as GPI

data_dir = "/u/nnas/data/HR8799/HR8799_AG_reduced/GPIK2/" #SPHERE-0101C0315A-20/channels/
distance = 41.2925 #pc
instrument = "GPI"
planet_name = "HR8799e"
pxscale = 0.00746

def main(args):
    sys.path.append(os.getcwd())

    global data_dir 
    global instrument
    global planet_name
    global pxscale
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default= "/u/nnas/data/")
    parser.add_argument("instrument", type=str, default= "GPI")
    parser.add_argument("name", type=str, default= "HR8799")   
    args = parser.parse_args(args)

    if "gpi" in instrument.lower():
        pxscale = 0.0162
    elif "sphere" in instrument.lower():
        pxscale = 0.007462

    data_dir = args.path
    instrument = args.instrument
    planet_name = args.name
    posn_dict = read_astrometry(data_dir+"../",planet_name)

    # Spectrum - the flux calibrated spectrum
    spectrum = np.load(data_dir + instrument + "_" + planet_name + "_flux_10pc_7200K.npy")

    # Contrast unit spectrum.
    contrasts = np.load(data_dir + instrument + "_" + planet_name + "contrasts.npy")
    # Residuals - the full frame residuals from processing, in contrast units
    # Might need to be careful about naming here.
    print("Loading Data")
    if os.path.exists(data_dir + instrument + "_" + planet_name + "_residuals.npy"):
        residuals = np.load(data_dir + instrument + "_" + planet_name + "_residuals.npy")
    elif os.path.exists(data_dir + instrument + "_" + planet_name + "_residuals.fits"):
        hdul = fits.open(data_dir + instrument + "_" + planet_name + "_residuals.fits")
        residuals = hdul[0].data
        hdul.close()
    else:
        print("No residual file found!")
        return

    # Checks to see if our residuals are for a full set of PCA components
    # Spectrum must have each PCA if residuals do.
    if len(residuals.shape)==4:
        npca = residuals.shape[0]
        nl = residuals.shape[1]
        if "pynpoint" in data_dir:
            pcas = [4,5,6,8,10,12,15,20]
        elif "pyklip" in data_dir:
            pcas = [3,4,5,8,10,12,15]

    else:
        npca = None
        nl = residuals.shape[0]
        pcas = None

    # First we need to get the uncorrelated error
    # This is a comnbination of residual error far from the star, and photometric error on the stellar psf
    # Both real units and fractional error are returned
    print("Calculating Uncorrelated Errors...")
    total_err,frac_err,_,_ = uncorrelated_error(residuals,spectrum,nl,npca,flux_cal=True)
    cont_err,frac_cont_err,_,_ = uncorrelated_error(residuals,contrasts,nl,npca)

    # Now we can compute the correlation and covariance matrices
    # The covariance matrix is normalised so that sqrt(diag(cov)) = uncorrelated error
    print("Computing Covariance...")
    cor,cov = get_covariance(residuals,total_err,posn_dict,nl,npca)
    cor_cont,cov_cont = get_covariance(residuals,cont_err,posn_dict,nl,npca)

    print("Done!")
    # All of the outputs get combined and saved to a fits file
    fits_output(spectrum,cov,cor, pcas=pcas, contrast = contrasts, cont_cov = cov_cont)
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

# So much repeated code to clean up....
# Apologies to future me.
def get_covariance(residuals,total_err,posn_dict,nl,npca=None):
    center = (residuals.shape[-2]/2,residuals.shape[-1]/2.0)
    width = 7
    r_in = posn_dict['Separation [mas]'][0]/1000/pxscale - width
    r_out = posn_dict['Separation [mas]'][0]/1000/pxscale + width
    annulus_c = CircularAnnulus(center,r_in = r_in, r_out = r_out)
    if npca is not None:
        cors = []
        covs = []
        for j in range(npca):
            fluxes = []
            for i in range(nl):
                # Stack and median subtract
                med = residuals[j,i]
                # Mask out planet (not sure if correct location)
                mask = create_circular_mask(residuals.shape[-2],residuals.shape[-1],
                                            center = (posn_dict["Px RA offset [px]"][0], posn_dict["Px Dec offset [px]"][0]),
                                            radius = posn_dict["Separation [mas]"][0]/1000/pxscale)
                
                # Get fluxes of all pixels within annulus
                annulus = annulus_c.to_mask(method='center')
                flux = annulus.multiply(mask*med)[annulus.data>0]
                fluxes.append(fluxes)
            fluxes = np.array(fluxes)
            # Compute correlation, Eqn 1 Samland et al 2017, https://arxiv.org/pdf/1704.02987.pdf
            cor = np.zeros((fluxes.shape[0],fluxes.shape[0]))
            cov = np.zeros((fluxes.shape[0],fluxes.shape[0]))

            for m in range(nl):
                for n in range(nl):
                    top =    np.mean(np.dot(fluxes[m],fluxes[n]))
                    bottom = np.sqrt(np.mean(np.dot(fluxes[m],fluxes[m]))*np.mean(np.dot(fluxes[n],fluxes[n])))
                    #print(top,bottom)
                    cor[m,n] = top/bottom 
                    cov[m,n] = top/bottom * total_err[m]*total_err[n]

            cors.append(cor)
            covs.append(cov)
        cors = np.array(cors)
        covs = np.array(covs)

        np.save(data_dir + instrument + "_" + planet_name + "_correlation",cors)
        np.save(data_dir + instrument + "_" + planet_name + "_covariance",covs)
        return cors,covs

    else:
        fluxes = []
        for i in range(nl):
            # Stack and median subtract
            med = residuals[i]
            # Mask out planet (not sure if correct location)
            mask = create_circular_mask(residuals.shape[-2],residuals.shape[-1],
                                        center = (posn_dict["Px RA offset [px]"], posn_dict["Px DEC offset [px]"]),
                                        radius = posn_dict["Separation [mas]"]/1000*pxscale)
            
            # Get fluxes of all pixels within annulus
            annulus = annulus_c.to_mask(method='center')
            flux = annulus.multiply(mask*med)[annulus.data>0]
            fluxes.append(fluxes)
        fluxes = np.array(fluxes)
        cor = np.zeros((fluxes.shape[0],fluxes.shape[0]))
        cov = np.zeros((fluxes.shape[0],fluxes.shape[0]))

        for m in range(nl):
            for n in range(nl):
                top =    np.mean(np.dot(fluxes[m],fluxes[n]))
                bottom = np.sqrt(np.mean(np.dot(fluxes[m],fluxes[m]))*np.mean(np.dot(fluxes[n],fluxes[n])))
                #print(top,bottom)
                cor[m,n] = top/bottom
                cov[m,n] = top/bottom * total_err[m]*total_err[n]

        np.save(data_dir + instrument + "_" + planet_name + "_correlation",cor)
        np.save(data_dir + instrument + "_" + planet_name + "_covariance",cov)
        return cor,cov


def uncorrelated_error(residuals,spectrum,nl,npca=None,flux_cal = False):
    # load in full residual file
    # use annulus far from center
    # get std of each aperture in contrast units
    # load in stellar spectra, convert to flux units
    # load in raw data file
    # extract photometry from sat spots, or from OBJECT/FLUX
    # extract from background, compute SNR
    # add errors in quadrature
    # save as array with dims (npca,nl)
    if "gpi" in instrument.lower():
        loc = 1200.
    elif "sphere" in instrument.lower():
        loc = 750.
    center = (residuals.shape[-2]/2,residuals.shape[-1]/2.0)
    width = 7
    r_in = loc/1000/pxscale - width
    r_out = loc/1000/pxscale + width
    annulus_c = CircularAnnulus(center,r_in = r_in, r_out = r_out)
    model = np.genfromtxt("/u/nnas/data/HR8799/stellar_model/hr8799_star_spec_" + instrument.upper() + "_fullfit_10pc.dat").T

    if flux_cal:
        stellar_model = model[1]
    else:
        stellar_model = np.ones_like(model[1])
    # Get stellar PSF
    # Must be at least 30 px wide (for background error)
    if instrument.lower() == "sphereyj":
        psf_cube = fits.open(glob.glob(data_dir + psf_name)[0])
        psf_cube = np.nanmean(psf_cube,axis=1)
    elif instrument.lower() == "sphereyjh":
        psf_cube = fits.open(glob.glob(data_dir + psf_name)[0])
    elif "gpi" in instrument.lower():
        psf_name = glob.glob(data_dir + "../*_PSF_cube.fits")[0]
        psf = fits.open(psf_name)[0].data
        filelist = sorted(glob.glob(data_dir +"../*distorcorr.fits"))
        dataset = GPI.GPIData(filelist, highpass=False, PSF_cube = psf,recalc_centers=True)
        dataset.generate_psf_cube(41)
        psf_cube = dataset.psfs
    
    # Fractional error on stellar psf
    phot_err = photometric_error(psf_cube)
    fluxes = []
    uncor_err = []
    total_err = []
    real_err = []
    if npca is not None:
        for j in range(npca):
            flux_l = []
            uc_l = []
            tot_l = []
            for i in range(nl):
                # Stack and median subtract
                med = residuals[j,i]
                # Mask out planet (not sure if correct location)
                # Get fluxes of all pixels within annulus
                annulus = annulus_c.to_mask(method='center')

                # Assuming contrast units for residuals
                flux = annulus.multiply(med)[annulus.data>0]*stellar_model[i]
                frac_err = 1/np.sqrt(np.var(flux) + spectrum[j,i])
                flux_l.append(flux)
                uc_l.append(frac_err)
                tot_l.append(np.sqrt(frac_err**2 + phot_err[i]**2))
            flux_l = np.array(flux_l)
            uc_l = np.array(uc_l)
            tot_l = np.array(tot_l)

            fluxes.append(flux_l)
            uncor_err.append(uc_l)
            total_err.append(tot_l)
            real_err.append(tot_l*spectrum[j])

        fluxes = np.array(fluxes)
        uncor_err = np.array(uncor_err)
        total_err = np.array(total_err)
        real_err = np.array(real_err)
    else:
        for i in range(nl):
            # Stack and median subtract
            med = residuals[i]
            # Mask out planet (not sure if correct location)
            # Get fluxes of all pixels within annulus
            annulus = annulus_c.to_mask(method='center')

            # Assuming contrast units for residuals
            flux = annulus.multiply(med)[annulus.data>0]*stellar_model[i]
            frac_err = 1/np.sqrt(np.var(flux) + spectrum[i])

            fluxes.append(flux)
            uncor_err.append(frac_err)
            total_err.append(np.sqrt(frac_err**2 + phot_err[i]**2))
        fluxes = np.array(fluxes)
        uncor_err = np.array(uncor_err)
        total_err = np.array(total_err)
        real_err = total_err*spectrum
    return real_err, total_err, uncor_err, fluxes

def photometric_error(psf_cube):
    # psf_cube has shape (wlen,x,y)
    # Get the relative photometric error 
    # Hardcoded: photometric aperture of 4px radius
    #            noise annulus between 9-16px radii
    #            different npix is accounted for
    std = []
    flux = []
    for frame in psf_cube[:]:
        y_img, x_img = np.indices(frame.shape, dtype=float)
        r_img = np.sqrt((x_img - frame.shape[0]/2.0)**2 + (y_img - frame.shape[1]/2.0)**2)
        noise_annulus = np.where((r_img > 9) & (r_img <= 16))
        psf_mask = np.where(r_img < 4.0)
        
        background_sum = np.nansum(frame[noise_annulus])
        n_ann = frame[noise_annulus].shape[0]
        n_psf = frame[psf_mask].shape[0]
        
        background_std = np.std(frame[noise_annulus])
        std.append(np.sum(frame[psf_mask])/np.sqrt((background_sum* n_psf/n_ann) + np.sum(frame[psf_mask])) )
        flux.append(np.sum(frame[psf_mask])) 
    std = np.array(std)
    flux = np.array(flux)
    return std/flux

def fits_one_output(spectrum,covariance,correlation,pca=None,contrast = None,cont_cov = None):
    wavelength = get_wlen()

    if "gpi" in instrument.lower():
        instrum = "GPI"
    elif "sphere" in instrument.lower():
        instrum = "SPHERE"
    primary_hdu = fits.PrimaryHDU(name = 'PRIMARY')
    primary_hdu.header['INSTRUME'] = instrum
    primary_hdu.header['OBJECT'] = planet_name

    c1 = fits.Column(name = "WAVELENGTH", array = wavelength, format = 'D',unit = "micron")
    c2 = fits.Column(name = "FLUX", array = spectrum, format = 'D',unit = "W/m2/micron")
    c3 = fits.Column(name = "COVARIANCE", array = covariance, format = 'D',unit = "[W/m2/micron]^2")
    c4 = fits.Column(name = "CORRELATION", array = correlation, format = 'D',unit = " - ")
    columns = [c1,c2,c3,c4]
    if contrast is not None:
        c5 = fits.Column(name = "CONTRAST", array = wavelength, format = 'D',unit = " - ")
        c6 = fits.Column(name = "COVARIANCE_CONTRAST", array = wavelength, format = 'D',unit = " - ^2")
        columns.extend([c5,c6])
    table_hdu = fits.BinTableHDU.from_columns(columns,name = 'SPECTRUM')
    hdul = fits.HDUList([primary_hdu,table_hdu])

    outstring = data_dir + instrument + "_" + planet_name 
    if pca is not None:
        pcastr = "_"+str(pca).zfill(2)
        outstring += pcastr
    outstring += "_spectrum.fits"
    hdul.writeto(outstring)
    return

def fits_output(spectrum,covariance,correlation,pcas=None,contrast = None,cont_cov = None):
    if pcas is not None:
        for i,pca in enumerate(pcas):
            fits_one_output(spectrum[i],covariance[i],correlation[i],pca,contrast[i],cont_cov[i])
    else:
        fits_one_output(spectrum,covariance,correlation,None,contrast,cont_cov)
    return

if __name__ == '__main__':
    main(sys.argv[1:])
