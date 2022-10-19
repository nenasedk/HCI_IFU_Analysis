import sys,os
import glob
import warnings
import collections
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
if not 'DISPLAY' in os.environ:
    import matplotlib
    matplotlib.use('Agg') # set the backend before importing pyplot
import matplotlib.pyplot as plt

import astropy.units as u
import astropy.constants as c
from astropy.io import fits

import pyklip.instruments.GPI as GPI
import pyklip.instruments.SPHERE as SPHERE
import pyklip.fm as fm
import pyklip.fmlib.fmpsf as fmpsf
import pyklip.fitpsf as fitpsf

from IFUData import IFUProcessingObject
class IFUAstrometry(IFUProcessingObject):
    def __init__(self,
                 IFUdata : IFUData,
                 instrument : str,
                 planet_name: str,
                 estimated_position : Tuple,
                 pixel_scale: Optional[float] = 1.0,
                 input_path: Optional[str] = "",
                 output_path: Optional[str] = "",
                 verbose = 2,
                 algorithm = "klip"
                 ):
        super().__init__()
        self.data = IFUData
        self.instrument = instrument
        self.planet_name = planet_name
        self.guessep,self.guesspa = estimated_position
        self.guessep*=u.mas
        self.guesspa*=u.degree
        self.pixel_scale = pixel_scale*u.mas

        self.verbose = verbose
        self.algorithm = algorithm
        self.posn = None

        self.x_offset_px = None
        self.y_offset_px = None

    def get_astrometry(self,
                       guessflux = 1e-6,
                       stellar_type = 'A0V'):
        if os.path.exists(f"{self.output_dir}{self.planet_name}_astrometry.txt"):
            astro_dict = self.read_astrometry(self.output_dir)
            self.set_posn_from_dict(astro_dict)
            return self.x_offset_px,self.y_offset_px
        return self.estimate_astrometry(guessflux,stellar_type)

    def estimate_astrometry(self, guessflux, stellar_type):
        dataset = self.init_dataset(science_base_name = "distorcorr")
        fit = self.klip_bayesian_astrometry(dataset,
                                 guessflux = guessflux,
                                 stellar_type = stellar_type)
        self.write_astrometry(fit,self.output_dir,self.planet_name)
        astro_dict = self.read_astrometry(self.output_dir)
        self.set_posn_from_dict(astro_dict)
        return self.x_offset_px,self.y_offset_px

    def set_posn_from_dict(astro_dict):
        # read_astrometry gives offsets in x,y, need to compute absolute posns
        self.x_offset_px = -1*astro_dict["Px RA offset [px]"][0]
        self.y_offset_px = -1*astro_dict["Px Dec offset [px]"][0]
        self.posn = (astro_dict["Separation [mas]"][0]*u.mas, astro_dict["PA [deg]"][0]*u.degree)
        return self.posn

    def relative_astrometry_to_px(posn):
        x_px = (posn[0].value/self.pixel_scale)*np.sin(posn[1].to(u.rad).value)
        y_px = -(posn[0].value/self.pixel_scale)*np.cos(posn[1].to(u.rad).value)
        # negative because of image orientation on sky
        return x_px,y_px

    def pixels_to_relative_astrometry(posn_px):
        sep = np.sqrt(posn_px[0]**2 + posn_px[1]**2)*self.pixel_scale
        pa = np.arctan(posn_px[0]/posn_px[1]) #TODO check trig
        return sep,pa

    def klip_bayesian_astrometry(dataset,
                                 guessflux,
                                 stellar_type,
                                 numthreads = None):
        #### Astrometry Prep ###
        numbasis=np.array([8])
        if "Ifs" in dataset.__class__.__name__:
            instrument = "SPHERE"
        else:
            instrument = "GPI"
        # initialize the FM Planet PSF class
        if "sphere" in instrument.lower():
            dataind = 0 # For some reason the outputs for the fm are different
            dn_per_contrast = None
            guesssep = self.guesssep.to(u.arcsec).value / dataset.platescale
        elif "gpi" in instrument.lower():
            dn_per_contrast = dataset.dn_per_contrast# your_flux_conversion # factor to scale PSF to star PSF. For GPI, this is dataset.dn_per_contrast
            dataind = 1
            guesssep = self.guesssep.to(u.arcsec).value / GPI.GPIData.lenslet_scale

        fm_class = fmpsf.FMPlanetPSF(dataset.input.shape,
                                     numbasis,
                                     guesssep,
                                     self.guesspa.value,
                                     guessflux,
                                     dataset.psfs,
                                     np.unique(dataset.wvs),
                                     dn_per_contrast,
                                     star_spt=stellar_type,
                                     spectrallib=None)
        # Astrometry KLIP
        # PSF subtraction parameters
        # You should change these to be suited to your data!
        prefix = f"{instrument} _{self.planet_name}" # fileprefix for the output files
        stamp_size = 13
        annulus_bounds = [[guesssep-3*stamp_size, guesssep+3*stamp_size]] # one annulus centered on the planet
        subsections = [[(self.guesspa.value-3.0*stamp_size)/180.*np.pi,
                        (self.guesspa.value+3.0*stamp_size)/180.*np.pi]] # we are not breaking up the annulus
        padding = 2 # we are not padding our zones
        movement = 2 # we are using an conservative exclusion criteria of 2 pixels
        # run KLIP-FM
        fm.klip_dataset(dataset,
                        fm_class,
                        outputdir=self.output_dir,
                        fileprefix=prefix,
                        numbasis=numbasis,
                        annuli=annulus_bounds,
                        subsections=subsections,
                        padding=padding,
                        movement=movement)

        ### FIT ASTROMETRY ###
        # read in outputs
        output_prefix = os.path.join(outputdir, prefix)

        fm_hdu = fits.open(output_prefix + "-fmpsf-KLmodes-all.fits")
        data_hdu = fits.open(output_prefix + "-klipped-KLmodes-all.fits")

        # get FM frame, use KL=8
        fm_frame = fm_hdu[dataind].data[0]
        fm_centx = fm_hdu[dataind].header['PSFCENTX']
        fm_centy = fm_hdu[dataind].header['PSFCENTY']

        # get data_stamp frame, use KL=8
        data_frame = data_hdu[dataind].data[0]
        data_centx = data_hdu[dataind].header["PSFCENTX"]
        data_centy = data_hdu[dataind].header["PSFCENTY"]

        # get initial guesses
        guesssep = fm_hdu[0].header['FM_SEP']
        guesspa = fm_hdu[0].header['FM_PA']

        # create FM Astrometry object that does MCMC fitting
        fit = fitpsf.FMAstrometry(guesssep, guesspa, 13, method="mcmc")
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
        flux_range = 2 # flux can vary by 2 order of magnitude
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
        # Outputs and Error Propagation
        fit.propogate_errs(star_center_err=0.05,
                           platescale=platescale,
                           platescale_err=plate_err,
                           pa_offset=-0.1,
                           pa_uncertainty=0.13)
        return fit

    def write_astrometry(fit,data_dir,planet_name):
        # show what the raw uncertainites are on the location of the planet
        myfile = open(f"{data_dir}{planet_name}_astrometry.txt",'w+')
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
        plt.savefig(f"{data_dir}{planet_name}_astrometry_walkers.pdf",
                    bbox_inches = 'tight')

        plt.clf()

        # Corner Plots are broken for some reason
        try:
            fig = plt.figure()
            fig = fit.make_corner_plot(fig=fig)
            plt.savefig(f"{data_dir}{planet_name}_astrometry_corner.pdf",
                        bbox_inches = 'tight')
            plt.clf()
        except:
            pass
        # Residual Plots
        fig = plt.figure()
        fig = fit.best_fit_and_residuals(fig=fig)
        plt.savefig(f"{data_dir}{planet_name}_astrometry_residuals.pdf",
                    bbox_inches = 'tight')
