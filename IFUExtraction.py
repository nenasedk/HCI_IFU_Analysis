import glob
import sys,os
import warnings
import collections
os.environ["OMP_NUM_THREADS"] = "1"

from typing import Tuple

import numpy as np
if not 'DISPLAY' in os.environ:
    import matplotlib
    matplotlib.use('Agg') # set the backend before importing pyplot

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import rc

import astropy.units as u
import astropy.constants as c
from astropy.io import fits

# Pyklip
import pyklip.instruments.GPI as GPI
import pyklip.instruments.SPHERE as SPHERE
import pyklip.fm as fm
import pyklip.fakes as fakes
import pyklip.fmlib.extractSpec as es
import pyklip.parallelized as parallelized

# Andromeda
import vip_hci as vip
from vip_hci.invprob.andromeda import andromeda

# Pynpoint
from pynpoint import Pypeline, \
                     FitsReadingModule,\
                     ParangReadingModule,\
                     PcaPsfSubtractionModule,\
                     AttributeReadingModule, \
                     BadPixelSigmaFilterModule,\
                     SimplexMinimizationModule,\
                     WavelengthReadingModule

from IFUData import IFUData,SpectralData
from IFUData import IFUProcessingObject


class IFU1DExtraction(IFUProcessingObject):
    def __init__(self,
                 IFUdata : IFUData,
                 instrument : str,
                 planet_name: str,
                 estimated_position : Tuple,
                 science_integration_time: Optional[float] = 1.0,
                 psf_integration_time: Optional[float] = 1.0,
                 psf_scaling: Optional[float] = 1.0,
                 pixel_scale: Optional[float] = 1.0,
                 distance_normalization: Optional[float] = 10.0,
                 input_path: Optional[str] = "",
                 output_path: Optional[str] = "",
                 verbose = 2,
                 algorithm = None
                 ):
        super.__init__(instrument = instrument,
                       planet_name = planet_name,
                       IFUData = IFUData,
                       input_path = input_path,
                       output_path = output_path)

        self.guessep,self.guesspa = estimated_position
        self.guessep*=u.mas
        self.guesspa*=u.degree
        self.dit_science = science_integration_time*u.second
        self.dit_flux = psf_integration_time*u.second
        self.psf_scaling = psf_scaling
        self.pixel_scale = pixel_scale*u.mas
        self.contrast_normalization = 1.0
        self.distance_normalization = distance_normalization
        self.stellar_spectrum = None
        self.contrasts = None
        self.flux_calibrated_spectra = None
        self.verbose = verbose
        self.algorithm = algorithm

    @abstractmethod
    def define_output_dirs(self,output_dir):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    @abstractmethod
    def extract_1d_spectrum(self):
        """
        Run an HCI Algorithm
        """

    def load_stellar_model(path):
        self.stellar_model = SpectralData(f"{self.data.name}_stellar_model",
                                          path)
        return self.stellar_model.wlen, self.stellar_model.spectrum

    def normalize_contrast_spectrum(self,
                                    input_contrasts,
                                    normalization_factor = 1.0,
                                    ):
        contrasts = input_contrasts * normalization_factor
        self.contrasts = [SpectralData(name = f"{self.algorithm}-contrast-{i:03}",
                                       wavelength = self.data.wlen,
                                       spectrum = contrast,
                                       distance = self.data.distance) for i,contrast in enumerate(contrasts)]
        return contrasts

    def contrasts_to_flux(self,
                          contrasts,
                          stellar_model,
                          input_distance,
                          output_distance):
        fluxes = [contrast.spectrum * stellar_model.spectrum * (input_distance/output_distance)**2 for contrast in contrasts]
        self.flux_calibrated_spectra = [SpectralData(name = f"{self.algorithm}-flux-{i:03}",
                                                    wavelength = self.data.wlen.value,
                                                    spectrum = flux.value,
                                                    distance = output_distance) for i,flux in enumerate(fluxes)]
        return fluxes

    # Get the PSF FWHM for each channel
    @staticmethod()
    def get_fwhm(psf):
        if len(psf.shape) ==3 :
            fwhm_fit = vip.var.fit_2dgaussian(psf[0], crop=True, cropsize=8, debug=False)
        else:
            fwhm_fit = vip.var.fit_2dgaussian(psfs, crop=True, cropsize=8, debug=False)

        fwhm = np.mean(np.array([fwhm_fit['fwhm_y'],fwhm_fit['fwhm_x']]))*self.pixel_scale.value # fit for fwhm
        return fwhm

    def plot_spectra(self,
                     spectra_list,
                     mean_spectrum = None,
                     label = "",
                     units = "[]"):
        fig,ax = plt.subplots(figsize = (16,10))
        ax.set_xlabel(f"{label} {units}")
        ax.set_title(f"{self.instrument} {self.planet_name} {label} - {self.algorithm}")
        for i,spectrum in enumerate(spectra_list):
            ax.plot(spectrum.wlen,
                    spectrum.spectrum,
                    label=spectrum.name,
                    alpha=0.5)
        if mean_spectrum is not None:
            ax.plot(mean_spectrum.spectrum.wlen,
                    mean_spectrum.spectrum,
                    label = mean_spectrum.name,
                    linewidth=4)
        ax.legend()
        plt.savefig(f"{self.output_dir}{self.instrument}_{self.planet_name}_{label}_{self.algorithm}.pdf",
                    bbox_inches = "tight")
        return fig, ax

    def write_fits_bintable(self, filename, data_products, column_names, units, table_name = 'SPECTRUM'):
        primary_hdu = fits.PrimaryHDU([])
        primary_hdu.header = hdr
        primary_hdu.header['OBJECT'] = self.planet_name
        columns = []
        for i,data in enumerate(data_products):
            form = 'D'
            if len(data.shape) > 1:
                form = f"{data.shape[0]}D"
            column = fits.Column(name = column_names[i], array = data, format = form, unit = units[i])
            columns.append(column)
        table_hdu = fits.BinTableHDU.from_columns(columns,name = table_name)
        hdul = fits.HDUList([primary_hdu,table_hdu])
        hdul.writeto(f"{self.output_dir}_{self.instrument}_{self.planet_name}_{filename}.fits",overwrite=True)
        return

    def write_fits_multiextension(self,
                                  filename,
                                  data_products,
                                  header = None,
                                  ext_headers = None):
        primary_hdu = fits.PrimaryHDU([])
        primary_hdu.header = header
        primary_hdu.header['OBJECT'] = self.planet_name
        hdulist = [primary_hdu]
        for i,data in enumerate(data_products):
            ext_hdr = None
            if ext_headers is not None:
                ext_hdr = ext_headers[i]
            ext = fits.ImageHDU(data = data, header = ext_hdr)
            hdulist.extend(ext)
        hdul = fits.HDUList([hdulist])
        hdul.writeto(f"{self.output_dir}_{self.instrument}_{self.planet_name}_{filename}.fits",overwrite=True)
        return

    def write_simple_fits(self,
                          data,
                          filename):
        hdu = fits.PrimaryHDU(data)
        hdul_new = fits.HDUList([hdu])
        hdul_new.writeto(f"{self.output_dir}_{self.instrument}_{self.planet_name}_{filename}.fits",overwrite=True)
        return

class KLIP1DExtraction(IFU1DExtraction):
    def __init__(self,
                 IFUdata : IFUData,
                 instrument : str,
                 planet_name: str,
                 estimated_position : Tuple,
                 science_integration_time: Optional[float] = 1.0,
                 psf_integration_time: Optional[float] = 1.0,
                 psf_scaling: Optional[float] = 1.0,
                 pixel_scale: Optional[float] = 1.0,
                 distance_normalization: Optional[float] = 10.0,
                 input_path: Optional[str] = "",
                 output_path: Optional[str] = "",
                 verbose = 2,
                 algorithm = "klip"
                 ):
        super().__init__(IFUData,
                         instrument,
                         planet_name,
                         estimated_position,
                         science_integration_time,
                         psf_integration_time,
                         psf_scaling,
                         pixel_scale,
                         distance_normalization,
                         input_path,
                         output_path,
                         verbose)

    def extract_1d_spectrum(self,
                            posn,
                            numbasis = [2]):
        """
        Run an HCI Algorithm
        """

        dataset = self.init_dataset(science_base_name = "distorcorr")

        exspect = self.run_klip_extractspec(posn,
                                            numbasis = [2],
                                            maxnumbasis = 99,
                                            movement = None,
                                            flux_overlap = 0.2,
                                            stamp_size = 9,
                                            sections = 10,
                                            mode = "ADI+SDI",
                                            numthreads = None)

        #KLIP Normalizes to integration time AFTER processing
        contrasts = self.normalize_contrast_spectrum(exspect,
                                                     self.contrast_normalization * (self.dit_science/self.dit_flux))

        fluxes = self.contrast_to_flux(self.contrasts,
                                      self.stellar_model,
                                      self.data.distance,
                                      output_distance = self.distance_normalization)
        m_contrast = np.mean(contrasts,axis = 1)
        m_flux = np.mean(fluxes,axis = 1)

        self.plot_spectra(self.contrasts,
                          mean_spectrum = m_contrast,
                          label = "Contrast",
                          units = "[F$_{p}$/F_{*}$]")
        self.plot_spectra(self.flux_calibrated_spectra,
                    mean_spectrum = m_flux,
                    label = "Flux Density, {output_distance}pc",
                    units = "[W/m$^{2}$/micron]")
        self.run_klip_parallel(dataset,
                               posn)
        return self.contrasts, self.flux_calibrated_spectra

    def run_klip_extractspec(self,
                             dataset,
                             posn,
                             numbasis = [2],
                             maxnumbasis = 99,
                             movement = None,
                             flux_overlap = 0.2,
                             stamp_size = 9,
                             sections = 10,
                             mode = "ADI+SDI",
                             numthreads = None):
        """
        Run KLIP

        Args:
            posn : tuple(float,float)
                The position of the planet, in (separation [mas], pa [degree])
        """
        planet_sep, planet_pa = posn
        planet_sep*= u.arcsec
        planet_pa *= u.degree
        planet_sep =planet_sep / self.pixel_scale #mas to pixels

        N_cubes =  int(dataset.input.shape[0]/np.unique(dataset.wvs).shape[0])


        ###### The forward model class ######
        # WATCH OUT FOR MEMORY ISSUES HERE
        # If the PSF size, input size or numbasis size is too large, will cause issues on cluster
        dtype = "float"
        if "sphere" in self.instrument:
            dtype = "float"
        fm_class = es.ExtractSpec(dataset.input.shape,
                            numbasis,
                            planet_sep.value,
                            planet_pa.value,
                            self.data.psf,
                            np.unique(dataset.wvs),
                            stamp_size = stamp_size,
                            datatype = dtype,
                            save_outputs = True) #must be double?

        ###### Now run KLIP! ######
        fm.klip_dataset(dataset,
                        fm_class,
                        fileprefix= f"{self.instrument}_{self.planet_name}_fmspect",
                        mode = mode,
                        annuli=[[planet_sep.value-1.5*stamp_size,planet_sep.value+1.5*stamp_size]], # select a patch around the planet (radius)
                        subsections=[[(planet_pa.value-2.0*stamp_size)/180.*np.pi,\
                                    (planet_pa.value+2.0*stamp_size)/180.*np.pi]], # select a patch around the planet (angle)
                        movement=movement,
                        flux_overlap = flux_overlap,
                        numbasis = numbasis,
                        maxnumbasis=maxnumbasis,
                        numthreads=numthreads,
                        spectrum=None,
                        save_klipped=save_outputs,
                        highpass=True,
                        calibrate_flux=True,
                        outputdir=self.output_dir,
                        mute_progression=True)

        # Save all outputs for future reference
        # TODO: Change to hdf5
        klipped = dataset.fmout[:,:,-1,:]
        dn_per_contrast = dataset.dn_per_contrast
        if save_outputs:
            np.save(f"{self.output_dir}{self.instrument}_{self.planet_name}_klipped",klipped)
            np.save(f"{self.output_dir}{self.instrument}_{self.planet_name}_dn_per_contrast",dn_per_contrast)

        # If you want to scale your spectrum by a calibration factor:
        units = "natural"
        scaling_factor = 1.0
        exspect, fm_matrix = es.invert_spect_fmodel(dataset.fmout, dataset, units=units,
                                                    scaling_factor=scaling_factor,
                                                    method="leastsq")
        if save_outputs:
            np.save(f"{self.output_dir}{self.instrument}_{self.planet_name}_exspect", exspect)
            np.save(f"{self.output_dir}{self.instrument}_{self.planet_name}_fm_matrix", fm_matrix)
        return exspect

    def run_klip_parallel(self,
                          dataset,
                          posn,
                          annuli = 12,
                          subsections = 10,
                          numbasis = [2],
                          maxnumbasis = 99,
                          movement = None,
                          flux_overlap = 0.2,
                          stamp_size = 9,
                          sections = 10,
                          mode = "ADI+SDI",
                          numthreads = None):
        if self.verbose > 1:
            print("Running full frame KLIP for residuals.\n")
        # Run KLIP again at the end so we can get the full residuals,
        # which we need for the covariance matrix later.
        planet_sep, planet_pa = posn
        planet_sep*= u.arcsec
        planet_pa *= u.degree
        planet_sep =planet_sep / self.pixel_scale #mas to pixels        ###### The forward model class ######

        parallelized.klip_dataset(dataset,
                                    mode='ADI',
                                    fileprefix=f"{self.instrument}_{self.planet_name}_fullframe",
                                    annuli=annuli,
                                    subsections=subsections,
                                    movement=movement,
                                    flux_overlap = flux_overlap,
                                    numbasis = numbasis,
                                    maxnumbasis=maxnumbasis,
                                    numthreads=numthreads,
                                    spectrum=None,
                                    algo='klip',
                                    highpass=True,
                                    calibrate_flux=True,
                                    outputdir=self.output_dir)
        return

class Andromeda1DExtraction(IFU1DExtraction):
    def __init__(self,
                 IFUdata : IFUData,
                 instrument : str,
                 planet_name: str,
                 estimated_position : Tuple,
                 science_integration_time: Optional[float] = 1.0,
                 psf_integration_time: Optional[float] = 1.0,
                 psf_scaling: Optional[float] = 1.0,
                 pixel_scale: Optional[float] = 1.0,
                 distance_normalization: Optional[float] = 10.0,
                 input_path: Optional[str] = "",
                 output_path: Optional[str] = "",
                 verbose = 2
                 ):
        super().__init__(IFUData,
                         instrument,
                         planet_name,
                         estimated_position,
                         science_integration_time,
                         psf_integration_time,
                         psf_scaling,
                         pixel_scale,
                         distance_normalization,
                         input_path,
                         output_path,
                         verbose,
                         algorithm = "andromeda")
    def preproc_data(self):
        if "gpi" in self.instrument:
            dataset = self.init_dataset(science_base_name = "distorcorr")
            del dataset
        return

    def extract_1d_spectrum(self,
                            posn,
                            inner_working_angle = 2.0,
                            annuli_width = 0.8,
                            filtering_fraction = 0.2,
                            min_sep = 0.45,
                            homogeneous_variance = False,
                            numthreads = None,
                            plotting = True):
        """
        Run an HCI Algorithm
        """
        self.preproc_data()
        contrast_map,snrs,stds = self.run_andromeda(inner_working_angle = inner_working_angle,
                                                 annuli_width = annuli_width,
                                                 filtering_fraction = filtering_fraction,
                                                 min_sep = min_sep,
                                                 homogeneous_variance = homogeneous_variance,
                                                 numthreads = numthreads)
        contrasts, fluxes = self.maximum_likelihood_1d_extract(contrast_map,
                                                               stds,
                                                               posn,
                                                               tolerance = 1)
        #Plotting
        if plotting:
            self.plot_spectra([self.contrasts],
                               mean_spectrum = None,
                               label = "Contrast",
                               units = "[F$_{p}$/F_{*}$]")
            self.plot_spectra([self.flux_calibrated_spectra],
                               mean_spectrum = None,
                               label = "Contrast",
                               units = "[F$_{p}$/F_{*}$]")
        return self.contrasts, self.flux_calibrated_spectra
    def run_andromeda(self,
                      inner_working_angle = 2.0,
                      annuli_width = 0.8,
                      filtering_fraction = 0.2,
                      min_sep = 0.45,
                      homogeneous_variance = False,
                      numthreads = None):
        if "sphere" in self.instrument:
            diam_tel = 8.2*u.m          # Telescope diameter [m]
        elif "gpi" in self.instrument:
            diam_tel = 10.*u.m          # Telescope diameter [m]
        else:
            warnings.warn("Incorrect telescope diameter! Check instrument.")
            diam_tel = 1.0*u.m

        for i,stack in enumerate(self.IFUData.data):
            fwhm = self.get_fwhm(self.IFUData.psf[i])
            cube = np.nan_to_num(stack)
            psf = np.nan_to_num(self.IFUData.psf[i])
            PIXSCALE_NYQUIST = (1/2.*self.IFUData.wlen[i].to(u.m)/diam_tel)*180*3600*1e3/np.pi # Pixscale at Shannon [mas/px]
            oversampling = PIXSCALE_NYQUIST /  self.pixel_scale                # Oversampling factor [1]

            if "sphere" in instrument.lower():
                #    #ang = -1*angles
                #    iwa = 2.0
                #    min_sep = 0.45
                owa = 60./oversampling
                #    width = 0.8
                #    filtering_frac = 0.35
            else:
                #    iwa = 1.0
                #    min_sep = 0.25
                owa = 45./oversampling
                #    #if 'k2' in instrument.lower():
                #    #    owa = 38
                #    width = 1.2
                #    filtering_frac = 0.3
            if self.verbose > 1:
                verb = True
                print(PIXSCALE_NYQUIST,oversampling, psf.shape,cube.shape, ang.shape)
            else:
                verb = False
            contrast,snr,snr_norm,std_contrast,std_contrast_norm,_,_ = andromeda(cube=cube,
                                                                                oversampling_fact=oversampling,
                                                                                angles=self.IFUData.pa,
                                                                                psf=psf,
                                                                                filtering_fraction = filtering_fraction,
                                                                                min_sep=min_sep,
                                                                                iwa=inner_working_angle,
                                                                                annuli_width = annuli_width,
                                                                                owa=owa,
                                                                                opt_method='no',
                                                                                fast=False,
                                                                                nproc=numthreads,
                                                                                homogeneous_variance=homogeneous_variance,
                                                                                ditimg = 1.0,
                                                                                ditpsf = self.contrast_normalization,
                                                                                verbose = verb)
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

        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.write_simple_fits(contrasts,"residuals")
        self.write_simple_fits(c_norms,"normed")
        self.write_simple_fits(snrs,"snrs")
        self.write_simple_fits(std_norms,"stds")
        return contrasts, snrs, std_norms

    def maximum_likelihood_1d_extract(self,
                                      contrasts,
                                      stds,
                                      posn,
                                      tolerance = 1):
        #Contrast
        peak_spec = []
        for i,frame in enumerate(contrasts):
            peak_spec.append(np.nanmax(frame[int(posn[1])-tolerance:int(posn[1])+tolerance,int(posn[0])-tolerance:int(posn[0])+tolerance]))
        peak_spec = np.array(peak_spec)

        # Andromeda normalizes the contrast DURING processing
        contrasts = [SpectralData("Contrast-Andromeda",
                                 wavelength = self.data.wlen,
                                 spectrum = peak_spec,
                                 distance = None,
                                 units = None)]
        np.save(f"{self.output_dir}{self.instrument}_{self.planet_name}_contrast",peak_spec)

        # Error
        errs= []
        for i,frame in enumerate(stds):
            errs.append(frame[int(posn[1]),int(posn[0])])
        errs = np.array(errs)
        np.save(f"{self.output_dir}{self.instrument}_{self.planet_name}_conterr",errs)
        contrasts.errors = errs
        self.contrasts = contrasts

        # Flux

        fluxes = self.contrasts_to_flux(self.contrasts,
                                        self.stellar_model,
                                        self.data.distance,
                                        output_distance = self.distance_normalization)
        np.save(data_dir + "andromeda/" + instrument + "_" + planet_name + "_flux_10pc_7200K",fluxes)
        return self.contrasts, self.flux_calibrated_spectra

class Pynpoint1DExtraction(IFU1DExtraction):
    def __init__(self,
                 IFUdata : IFUData,
                 instrument : str,
                 planet_name: str,
                 estimated_position : Tuple,
                 science_integration_time: Optional[float] = 1.0,
                 psf_integration_time: Optional[float] = 1.0,
                 psf_scaling: Optional[float] = 1.0,
                 pixel_scale: Optional[float] = 1.0,
                 distance_normalization: Optional[float] = 10.0,
                 input_path: Optional[str] = "",
                 output_path: Optional[str] = "",
                 verbose = 2,
                 algorithm = "pynpoint"
                 ):
        super().__init__(IFUData,
                         instrument,
                         planet_name,
                         estimated_position,
                         science_integration_time,
                         psf_integration_time,
                         psf_scaling,
                         pixel_scale,
                         distance_normalization,
                         input_path,
                         output_path,
                         verbose)

    def extract_1d_spectrum(self,
                            posn,
                            pcas = [2],
                            skip = True):
        if "gpi" in self.instrument:
            dataset = self.init_dataset(science_base_name = "distorcorr")
            self.data.psf = self.data.pad_cube(self.data.psf,
                                                (self.data.cube.shape[-2],self.data.cube.shape[-1]))

            del dataset
        elif "sphere" in self.instrument:
            self.contrast_normalization = self.dit_flux/self.dit_science

        contrasts, residuals = self.run_pynpoint_all_channels(posn,
                                                              pcas,
                                                              output_name = f"{self.planet_name}_{self.instrument.upper()}",
                                                              skip = skip)
        # PynPoint normalizes the contrast DURING processing
        contrasts = self.normalize_contrast_spectrum(contrasts,
                                                     normalization_factor=1.0)
        fluxes = self.contrasts_to_flux(self.contrasts,
                                        self.stellar_model,
                                        self.data.distance,
                                        output_distance = self.distance_normalization)
        return self.contrasts, self.flux_calibrated_spectra

    def run_pynpoint_all_channels(self,
                                  posn,
                                  pcas,
                                  output_name,
                                  skip = True):
        if not isinstance(pcas, list):
            pcas = pcas.tolist()
        nChannels = len(self.data.wlen)
        # Loop over all channels
        for channel in range(nChannels):
            working_dir = f"{self.output_dir}/CH{channel:03}/"
            # Allow us to resume if some channels have already been calculated
            if skip:
                if os.path.exists(f"{self.output_dir}{output_name}_ch{channel:03}_flux_pos_out_pca.npy"):
                    continue
            # Have to copy the config file to the working dir
            # Using separate dirs to get unique databases for each wavelength
            # Might be unnecessary, but then I don't have a 50GB hdf5 file to deal with
            if not os.path.isdir(working_dir):
                os.makedirs(working_dir, exist_ok=True)
            if "sphere" in instrument.lower():
                shutil.copy("config/Pynpoint_config_SPHERE.ini", working_dir + "PynPoint_config.ini")
            elif "gpi" in instrument.lower():
                shutil.copy("config/Pynpoint_config_GPI.ini", working_dir + "PynPoint_config.ini")

            # Naming everything consistently
            # Channel must be zfilled for sorting
            # Better option would be using fits headers, but that's a pain.
            science_name = f"{output_name}_{channel:03}_reduced.fits"
            psf_name = f"{output_name}_{channel:03}_PSF.fits"
            if os.path.isfile(working_dir + "PynPoint_database.hdf5"):
                os.remove(working_dir + "PynPoint_database.hdf5")

            # Run Simplex Minimization module in pynpoint
            fluxes, residuals = simplex_one_channel(posn = posn,
                                                    aperture = aperture,
                                                    science_name = science_name,
                                                    psf_name = psf_name,
                                                    pcas = pcas,
                                                    tolerance = tolerance,
                                                    center_size = center_size,
                                                    allowed_offset = allowed_offset)

            np.save(f"{self.output_dir}{output_name}_ch{channel:03}_flux_pos_out",fluxes)
            self.write_fits_multiextension(f"ch{channel:03}_subtracted",
                                       residuals,
                                       header = self.data.header,
                                       ext_headers = [{"nPC":pc} for pc in pcas])

        combined_contrasts = self.load_pynpoint_contrasts("flux_pos_out")
        combined_residuals = self.load_pynpoint_residuals(f"ch{channel:03}_subtracted")
        return combined_contrasts, combined_residuals

    def load_pynpoint_residuals(self,filename):
        nChannels = len(self.data.wlen)
        residuals = []
        for channel in range(nChannels):
            hdul = fits.open(f"{self.output_dir}_{self.instrument}_{self.planet_name}_{filename}.fits")
            r_pca = []
            for hdu in hdul:
                r_pca.append(hdu.data)
            residuals.append(np.array(r_pca))
            hdul.close()
        residuals = np.array(residuals)
        return residuals

    def load_pynpoint_contrasts(self,filename):
        contrast = [] # the contrast of the planet itself
        for channel in range(nChannels):
            samples = np.load(f"{self.output_dir}{output_name}_ch{channel:03}_{filename}.npy")
            pca_contrasts = samples[:,-1,4]
            samples = 10**(samples/-2.5)
            contrast.append(samples[-1][4])
        cont = np.array(contrast)
        cont = cont.swapaxes(0, 1)
        np.save(output_place + output_name +  "_contrasts",cont) # saved in contrast units
        return cont

    def simplex_one_channel(self,
                            posn,
                            aperture,
                            science_name,
                            psf_name,
                            pcas,
                            tolerance,
                            center_size,
                            allowed_offset,
                            ):
        #set_fwhm(channel)
        pipeline = Pypeline(working_place_in=f"{self.output_dir}/CH{channel:03}/",
                            input_place_in=self.input_dir,
                            output_place_in=self.output_dir)

        module = FitsReadingModule(name_in="read_science",
                                input_dir=self.input_dir,
                                image_tag="science",
                                ifs_data = False,
                                filenames = [self.input_dir + input_name])

        pipeline.add_module(module)
        module = FitsReadingModule(name_in="read_center",
                                input_dir=self.input_dir,
                                image_tag="center",
                                ifs_data = False,
                                filenames = [self.input_dir + input_name])

        pipeline.add_module(module)

        module = FitsReadingModule(name_in="read_psf",
                                input_dir=self.input_dir,
                                image_tag="psf",
                                ifs_data = False,
                                filenames = [self.input_dir + psf_name])

        pipeline.add_module(module)
        # might need to glob parang files
        module = ParangReadingModule(file_name="parangs.fits",
                                    input_dir=self.input_dir,
                                    name_in="parang",
                                    data_tag = 'science',
                                    overwrite=True)
        pipeline.add_module(module)
        module = ParangReadingModule(file_name="parangs.fits",
                                    input_dir=self.input_dir,
                                    name_in="parang_cent",
                                    data_tag = 'center',
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

        module = SimplexMinimizationModule(name_in = 'simplex',
                                        image_in_tag = 'science',
                                        psf_in_tag = 'psf',
                                        res_out_tag = f"{self.planet_name}_flux_channel_{channel}_",
                                        flux_position_tag = f"{self.planet_name}_flux_pos_channel_{channel}_",
                                        position = posn,
                                        magnitude = 14.0, # approximate planet contrast in mag
                                        psf_scaling = -1/self.contrast_normalization,
                                        merit = 'gaussian', #better than hessian
                                        aperture = aperture*self.pixel_scale.to(u.arcsec).value,
                                        tolerance = tolerance, # tighter tolerance is good
                                        pca_number = pcas, #listed above
                                        cent_size = center_size, # how much to block out
                                        offset = allowed_offset) #use fixed astrometry from KLIP

        pipeline.add_module(module)
        pipeline.run()
        fluxes = []
        r_list = []
        for pca in pcas:
            flux = pipeline.get_data(f"{self.planet_name}_flux_pos_channel_{channel:03}_{pca:03}")
            fluxes.append(flux)
            #np.savetxt(data_dir+ "pynpoint_"+planet_name + "/" + output_name + "_ch" + str(channel).zfill(3) +"_flux_pos_out_pca_" +str(pca)+ ".dat",flux)
            residuals = pipeline.get_data(f"{self.planet_name}_flux_channel_{channel:03}_{pca:03}")
            norm = pipeline.get_data("psf")
            residuals = residuals / self.contrast_normalization  / np.nanmax(norm)
            r_list.append(residuals)
            #save_residuals(residuals[-1],output_name +"_residuals_" + str(channel).zfill(3) + "_pca_" + str(pca), data_dir+ "pynpoint_"+planet_name + "/" )
        os.remove(working_dir + "PynPoint_database.hdf5")
        return np.array(fluxes), np.array(r_list)

    def preprocess_for_pynpoint(self,
                                skip = False):

        if skip and os.path.exists(f"{self.input_dir}{self.planet_name}_{self.instrument}_000_reduced.fits"):
            return

        data_shape = None
        # Separate full cube into wavelength channels
        # SimplexMinimization doesn't work on IFU data naturally
        for channel,frame in enumerate(self.data.cube[:]):
            self.write_simple_fits(frame, f"{channel:03}_reduced")
        # Individual PSFs
        for channel,frame in enumerate(self.data.psf[:]):
            self.write_simple_fits(frame, f"{channel:03}_PSF")

         # Save wavelengths
        self.write_simple_fits(self.data.wlen, "wavelength")
        self.write_simple_fits(self.data.pa, "parangs")

