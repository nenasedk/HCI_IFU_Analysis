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

import vip_hci as vip
from vip_hci.preproc import recentering,\
                            cube_recenter_2dfit,\
                            cube_recenter_dft_upsampling,\
                            cube_shift,\
                            cube_crop_frames
from pyklip.instruments import SPHERE, GPI

from typing import Optional, Tuple
from abc import abstractmethod
from spectres import spectres
class DataObject:
    def __init__(self,
                 name = None,
                 path = None,
                 verbose = 0):
        self.name = name
        self.path = path
        self.verbose = verbose
        self.header = {}

    def read_data(self,
                  hdul_index = 0,
                  data_type = ""):
        if self.path.endswith(".fits"):
            hdul = fits.open(self.path)
            data = hdul[hdul_index].data
            hdul.close()
        elif self.path.endswith(".npy"):
            data = np.load(self.path)
        elif self.path.endswith(".txt") or self.path.endswith(".dat"):
            data = np.genfromtxt(self.path).T
        if self.verbose > 1:
            print(f"Read {data_type} data from {self.path}.")
            print(f"{data_type} has shape {data.shape}.")
        return data

    def read_data_from_file(self,
                            input_dir,
                            filename,
                            hdul_index = 0,
                            data_type = ""):
        self.path = os.path.join(input_dir,filename)
        data = self.read_data()
        return data

    def set_header_from_fits(self,
                            input_dir,
                            filename):
        if filename.endswith(".fits"):
            hdul = fits.open(os.path.join(input_dir,filename))
            self.header = hdul[0].header
            hdul.close()
        return self.header

    def add_to_header(self, key, value):
        self.header[key] = value

class SpectralData(DataObject):
    def __init__(self,
                 name,
                 path = None,
                 wavelength = None,
                 spectrum = None,
                 errors = None,
                 covariance = None,
                 correlation = None,
                 distance = None,
                 units = None,
                 verbose = 0):
        super().__init__(name,path,verbose)

        if path is not None and spectrum is None:
            input_dir, filename = os.path.split(path)
            read_in = self.read_data_from_file(input_dir,
                                               filename,
                                               name)
            if isinstance(read_in, np.ndarray):
                self.spectrum = read_in
            elif len(read_in)==2:
                self.wlen, self.spectrum = read_in
                self.error = np.zeros_like(self.wlen)
            else:
                self.wlen, self.spectrum, self.error = read_in
        else:
            self.spectrum = spectrum
            self.wlen = wavelength*u.micron
        self.distance = distance
        if distance is not None:
            self.distance *= u.parsec
        self.units = units
        if units is not None:
            self.spectrum *= units
            self.errors *= units
        self.covariance = covariance
        self.correlation = correlation

    def load_spectral_data(self):
        data = self.read_data()
        print(data.shape)
        if data.shape[0] > data.shape[-1]:
            data = data.T
        if data.shape[0] == 2:
            self.wlen = data[0]
            self.spectrum = data[1]
        elif data.shape[0] == 3:
            self.wlen = data[0]
            self.spectrum = data[1]
            self.error = data[1]
    def rebin_spectrum(self, new_wavelengths):
        if self.error is not None:
            self.spectrum,self.error = spectres(new_wavelengths,self.wlen,self.spectrum,self.error)
        else:
            self.spectrum = spectres(new_wavelengths,self.wlen,self.spectrum)
        self.wlen = new_wavelengths

    def save(self,path):
        return

class IFUData(DataObject):
    def __init__(self,
                 name,
                 input_dir : Optional[str] = None,
                 cube : Optional[np.ndarray] = None,
                 psf: Optional[np.ndarray] = None,
                 wavelengths: Optional[np.ndarray] = None,
                 parallactic_angles: Optional[np.ndarray] = None,
                 centers: Optional[np.ndarray] = None,
                 distance: Optional[float] = 1.0,
                 science_name = None,
                 psf_name = None,
                 wv_name = None,
                 pa_name = None,
                 verbose: Optional[int] = 0):
        super().__init__(name,
                         input_dir,
                         verbose)
        self.input_dir = input_dir
        if not self.input_dir.endswith("/"):
            self.input_dir += "/"

        self.cube = cube
        self.psf = psf
        self.wlen = wavelengths
        if wavelengths is not None:
            self.wlen*=u.micron
        self.pa = parallactic_angles
        self.distance = distance*u.parsec
        self.center = centers
        self.center_per_frame = None

        self.science_name = science_name
        self.psf_name = psf_name
        self.wv_name = wv_name
        self.pa_name = pa_name

        return

    def read_from_directory(self,
                            input_dir: str,
                            science_name: str,
                            psf_name: str,
                            wv_name: str,
                            pa_name: str,
                            hdul_index: int = 0):
        if not input_dir.endswith("/"):
            input_dir += "/"
        if self.verbose > 1:
            print(f"Reading IFU data from {input_dir}")

        cube = self.read_data_from_file(input_dir,
                                        science_name,
                                        hdul_index=hdul_index)
        psf = self.read_data_from_file(input_dir,
                                       psf_name)
        wlen = self.read_data_from_file(input_dir,
                                        wv_name)
        wlen = self.check_wavelength_units(wlen)

        pa = self.read_data_from_file(input_dir,
                                      pa_name)

        self.science_name = science_name
        self.psf_name = psf_name
        self.wv_name = wv_name
        self.pa_name = pa_name

        self.cube = cube
        self.psf = psf
        self.wlen = wlen
        self.pa = pa
        return cube, psf, wlen, pa

    @staticmethod
    def check_wavelength_units(wlen):
        wlen_new = wlen
        if np.mean(wlen)>100.:
            wlen_new = wlen/1000
            warnings.warn(f"Wavelengths read in were not in units of micron. ({wlen[0]},{wlen[-1]})")
        return wlen_new*u.micron

    def write_input_summary(input_dir,
                        science_name,
                        psf_name,
                        wavelength_name,
                        parallactic_name,
                        to_file = False):
        warnings.warn("Not yet implemented!")
        return

    @staticmethod
    def even_shape(data):
        if not (data.shape[-1])%2 == 0:
            if len(data.shape) == 3:
                cube = cube_shift(np.nan_to_num(data[:,:-1,:-1]),-0.5,-0.5)
                return cube
            else:
                stack = []
                for entry in data:
                    stack.append(cube_shift(np.nan_to_num(entry[:,:-1,:-1]),-0.5,-0.5))
                return np.array(stack)
        else:
            return

    def recalc_centers(self, newcenter = None):
        if newcenter is None:
            newcenter = (self.cube.shape[-2],
                         self.cube.shape[-1])
        self.cube,self.center = self.recenter_ifu_cube(self.cube,
                                                       newcenter)
        return

    @staticmethod
    def recenter_ifu_cube(data, newcenter):
        shifts = []
        if isinstance(newcenter[0],collections.abc.Iterable):
            for channel,frame in enumerate(data[:]):
                frame = IFUData.even_shape(frame)
                shiftx,shifty = (int((frame.shape[-2]//2)) - newcenter[channel,0],
                                (int(frame.shape[-1]//2)) - newcenter[channel,1])
                shifted = vip.preproc.recentering.cube_shift(frame,shifty,shiftx)
                shifts.append(shifted)
            cube = np.array(shifts)
            center = (cube.shape[-2]/2.0,cube.shape[-1]/2.0)
        else:
            for channel,frame in enumerate(data[:]):
                frame = IFUData.even_shape(frame)
                shiftx,shifty = (int((frame.shape[-2]//2)) - newcenter[0],
                                (int(frame.shape[-1]//2)) - newcenter[1])
                shifted = vip.preproc.recentering.cube_shift(frame,shifty,shiftx)
                shifts.append(shifted)
            cube = np.array(shifts)
            center = (cube.shape[-2]/2.0,cube.shape[-1]/2.0)
        return cube, center

    @staticmethod
    def crop(data, half_width, center = None):
        if center is None:
            center = data.shape[-1]//2
        cropped = data[...,
                       int(center - half_width):int(center + halfwidth),
                       int(center - half_width):int(center + half_width)]
        return cropped

    @staticmethod
    def pad_cube(data,new_shape):
        padded_cube = []
        for channel,frame in enumerate(data):
            padded_cube.append(self.pad_image(frame,new_shape))
        padded_cube = np.array(padded_cube)
        return padded_cube

    @staticmethod
    def pad_image(data, new_shape):
        padx = int((new_shape[-1] - data.shape[0])/2.)
        pady = int((new_shape[-2] - data.shape[1])/2.)
        if (data.shape[0] + (2 * padx))%2 == 0:
            padded = np.pad(data,
                        ((padx,padx+1),
                        (pady,pady+1)),
                        'constant')
            padded = recentering.frame_shift(np.nan_to_num(padded),0.5,0.5)
        else:
            padded = np.pad(data,
                            ((padx,padx),
                            (pady,pady)),
                            'constant')
        return padded

    def read_gpi_data(self,
                      science_base_name,
                      psf_base_name):
        self.psfs = self.read_data_from_file(self.input_dir, psf_base_name)

        filelist = glob.glob(f"{self.input_dir}*{science_base_name}.fits")
        dataset = GPI.GPIData(filelist,
                              highpass=False,
                              PSF_cube = self.psfs,
                              recalc_centers=True)
        nInts = len(filelist)
        nChannels = np.unique(dataset.wvs).shape[0]
        centers = dataset.centers.reshape(nInts,nChannels,2)
        self.center = centers
        # Need to order the GPI data for pynpoint
        shape = dataset.input.shape
        science = dataset.input.reshape(nInts,nChannels,shape[-2],shape[-1])
        science = np.swapaxes(science,0,1)

        cube = []
        CENTER = (shape[-2]/2.0,shape[-1]/2.0)
        self.center = (shape[-2]/2.0,shape[-1]/2.0)

        for channel,stack in enumerate(science[:]):
            # The PSF center isn't aligned with the image center, so let's fix that
            centx = centers[:,channel,0]
            centy = centers[:,channel,1]
            shiftx,shifty = (CENTER[0]*np.ones_like(centx) - centx,
                             CENTER[1]*np.ones_like(centy) - centy)
            shifted = recentering.cube_shift(np.nan_to_num(stack),shifty,shiftx,border_mode='constant')
            cube.append(shifted)
        self.cube = np.array(cube)

        # Copy the GPI header, and add some notes of our own
        header_hdul = fits.open(filelist[0])
        self.header = header_hdul[0].header
        header_hdul.close()
        self.add_to_header('ESO ADA POSANG',(dataset.PAs.reshape(len(filelist),37)[:,0][0]+ 180.0))
        self.add_to_header('ESO ADA POSANG END',(dataset.PAs.reshape(len(filelist),37)[:,0][-1]+ 180.0 ))
        # Save wavelengths
        self.wlen = np.unique(dataset.wvs)

        # pyklip does weird things with the PAs, so let's fix that.
        # Keep or remove dataset.ifs_rotation? GPI IFS is rotated 23.5 deg,
        self.pa = (dataset.PAs.reshape(len(filelist),37)[:,0] + 180.0)
        del dataset

class IFUProcessingObject:
    def __init__(self,
                 instrument : str,
                 planet_name: str,
                 IFUDataSet: Optional[IFUData] = None,
                 input_dir: Optional[str] = None,
                 output_dir: Optional[str] = ""):
        self.instrument = instrument.lower()
        self.planet_name = planet_name
        self.data = IFUDataSet
        self.output_dir = output_dir
        self.input_dir = input_dir
        if input_dir is None:
            self.input_dir = IFUData.input_dir
        self.contrast_normalization = 1.0
        return

    @staticmethod
    def init_sphere_dataset(data):
        dataset = SPHERE.Ifs(data.input_dir + data.science_name,
                            data.input_dir + data.psf_name,
                            data.input_dir + data.pa_name,
                            data.input_dir + data.wv_name,
                            nan_mask_boxsize=9,
                            psf_cube_size = 15)
        return dataset

    @staticmethod
    def init_gpi_dataset(data,
                         input_dir,
                         science_base_name):
        filelist = sorted(glob.glob(f"{input_dir}*{science_base_name}.fits"))
        dataset = GPI.GPIData(filelist,
                              highpass=False,
                              PSF_cube = data.psf,
                              recalc_centers=False)
        band = dataset.prihdrs[0]['APODIZER'].split('_')[1]
        normalization = dataset.spot_ratio[band]
        return dataset, normalization

    def init_dataset(self,
                     science_base_name = "distorcorr"):
        if "sphere" in self.instrument:
            dataset = self.init_sphere_dataset(self.data)
        elif "gpi" in self.instrument:
            dataset,norm = self.init_gpi_dataset(self.data,
                                            self.input_dir,
                                            science_base_name = science_base_name)
            self.contrast_normalization = norm
        else:
            raise NotImplementedError("Only SPHERE and GPI have been implemented")
        return dataset
