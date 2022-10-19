import sys,os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
if not 'DISPLAY' in os.environ:
    import matplotlib
    matplotlib.use('Agg') # set the backend before importing pyplot
import matplotlib.pyplot as plt
# Error functions

def gauss(x, mu, sigma):
    g = (1/(sigma * np.sqrt(2 * np.pi)) *\
         np.exp(-(x - mu)**2 / (2 * sigma**2)))
    return g
def gauss_fit(array, plot = False):
    mean = np.mean(array.ravel())
    std = np.std(array.ravel())
    if plot:
        print(mean,std)
        plt.hist(array)
        plt.plot(gauss(array,mean,std))
    return mean, std

def laplacian(x, mu, width):
    pdf = np.exp(-abs(x-mu)/width)/(2.*width)
    return pdf
def laplace_fit(array, plot = False):
    mean, var = laplace.fit(array.ravel())
    if plot:
        print(mean,var)
        plt.hist(array)
        plt.plot(laplacian(array,mean,var))

    return mean, var

def snr_circles(frame,posn,fwhm, plot):
    snr = vip.metrics.snr(frame,source_xy=(posn[0],posn[1]), fwhm=fwhm, plot=plot)
    return snr
def snr_error_estimate(data,posn,fwhm, planet_spectrum = None, plot = False):
    nl = data.shape[0]
    if planet_spectrum is None:
        planet_spectrum = np.ones(nl)
    snrs = []
    errors = []
    for i in range(nl):
        snr = snr_circles(data[i],posn, fwhm, plot)
        snrs.append(snr)
        err = planet_spectrum[i]/snr
        errors.append(err)
    snrs = np.array(snrs)
    errors = np.array(errors)
    return snrs, errors

def photometric_error(psf_cube,stellar_spectrum = None, plot = True):
    # psf_cube has shape (wlen,x,y)
    # Get the relative photometric error
    # Hardcoded: photometric aperture of 4px radius
    #            noise annulus between 9-16px radii
    #            different npix is accounted for
    if stellar_spectrum is not None:
        psf_cube = normalize_psf(psf_cube)
        psf_cube *= stellar_spectrum[:,None,None]
    std = []
    bkgs = []
    flux = []
    for frame in psf_cube[:]:
        y_img, x_img = np.indices(frame.shape, dtype=float)
        r_img = np.sqrt((x_img - frame.shape[0]/2.0)**2 + (y_img - frame.shape[1]/2.0)**2)
        noise_annulus = np.where((r_img > 10) & (r_img <= 19))
        psf_mask = np.where(r_img < 5.0)

        background_sum = np.nansum(frame[noise_annulus])
        n_ann = frame[noise_annulus].shape[0]
        n_psf = frame[psf_mask].shape[0]

        background_std = np.std(frame[noise_annulus])
        if plot:
            plt.hist(frame[noise_annulus], density = True)
        std.append(np.sum(frame[psf_mask])/np.sqrt((background_sum* n_psf/n_ann) + np.sum(frame[psf_mask])))
        bkgs.append(np.std(frame[noise_annulus]))
        flux.append(np.sum(frame[psf_mask]))
    std = np.array(std)#/np.sqrt(nFrames)
    flux = np.array(flux)
    bkgs = np.array(bkgs)
    #if plot:
    #    plt.plot(std, label = "std deviation")
    #    plt.plot(bkgs, label = "backgrounds")
    #print(std, bkgs)
    return flux, std, bkgs

def sum_bkg_error(photometric_error, background_error,planet_spectrum = None,stellar_spectrum =None):
    nl = len(background_error)
    if planet_spectrum is None:
        planet_spectrum = np.ones(nl)
    if stellar_spectrum is None:
        stellar_spectrum = np.ones(nl)
    total_error = np.sqrt(background_error**2 + ((photometric_error*planet_spectrum/stellar_spectrum))**2)
    return total_error