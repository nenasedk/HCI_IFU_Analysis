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
import pyklip.instruments.GPI as GPI

class DataObject:
    def __init__(self,
                 name = None,
                 location = None):
        self.name = name
        self.location = location
        self.header = {}

    def read_data_from_file(self,
                            input_path,
                            filename,
                            data_type = ""):
        if filename.endswith(".fits"):
            hdul = fits.open(os.path.join(input_path,filename))
            data = hdul[0].data
            hdul.close()
        elif filename.endswith(".npy"):
            data = np.load(os.path.join(input_path,filename))
        elif filename.endswith(".txt") or filename.endswith(".dat"):
            data = np.genfromtxt(os.path.join(input_path,filename)).T
        if self.verbose > 1:
            print(f"Read {data_type} data from {filename}.")
            print(f"{data_type} has shape {data.shape}.")
        return data

    def set_header_from_fits(self,
                            input_path,
                            filename):
        if filename.endswith(".fits"):
            hdul = fits.open(os.path.join(input_path,filename))
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
                 units = None):
        super().__init__(name,path)

        if path is not None and spectrum is None:
            input_path, filenameos.path.split(path)
            read_in = self.read_data_from_file(input_path,
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
        self.distance = distance*u.parsec
        self.units = units
        if units is not None:
            self.spectrum *= units
            self.errors *= units
        self.covariance = covariance
        self.correlation = correlation

    def save(self,path):
        return

class IFUData(DataObject):
    def __init__(self,
                 name,
                 input_path : Optional[str] = None,
                 cube : Optional[np.ndarray] = None,
                 psf: Optional[np.ndarray] = None,
                 wavelengths: Optional[np.ndarray] = None,
                 parallactic_angles: Optional[np.ndarray] = None,
                 distance: Optional[float] = 1.0,
                 verbose: Optional[int] = 0):
        super().__init__(name,
                         input_path)

        if not self.input_path.endswith("/"):
            self.input_path += "/"

        self.cube = cube
        self.psf = psf
        self.wlen = wavelengths
        if wavelengths is not None:
            self.wlen*=u.micron
        self.pa = parallactic_angles
        self.distance = distance*u.parsec
        self.verbose = verbose
        self.centers = None
        return

    def read_from_directory(self,
                            input_path: str,
                            science_name: str,
                            psf_name: str,
                            wavelength_name: str,
                            parallactic_name: str):
        if not input_path.endswith("/"):
            input_path += "/"
        if self.verbose > 1:
            print(f"Reading IFU data from {input_path}")

        cube = self.read_data_from_file(input_path,
                                        science_name)
        psf = self.read_data_from_file(input_path,
                                       psf_name)
        wlen = self.read_data_from_file(input_path,
                                        wavelength_name)
        wlen = self.check_wavelength_units(wlen)

        pa = self.read_data_from_file(input_path,
                                      parallactic_name)
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

    def write_input_summary(input_path,
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
        self.psfs = self.read_data_from_file(self.input_path, psf_base_name)

        filelist = glob.glob(f"{self.input_path}*{science_base_name}.fits")
        dataset = GPI.GPIData(filelist,
                              highpass=False,
                              PSF_cube = self.psfs,
                              recalc_centers=True)
        nInts = len(filelist)
        nChannels = np.unique(dataset.wvs).shape[0]
        centers = dataset.centers.reshape(nInts,nChannels,2)

        # Need to order the GPI data for pynpoint
        shape = dataset.input.shape
        science = dataset.input.reshape(nInts,nChannels,shape[-2],shape[-1])
        science = np.swapaxes(science,0,1)

        cube = []
        CENTER = (shape[-2]/2.0,shape[-1]/2.0)
        self.centers = (shape[-2]/2.0,shape[-1]/2.0)

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
                 IFUData: Optional[IFUData] = None,
                 input_path: Optional[str] = "",
                 output_path: Optional[str] = "",):
        self.instrument = instrument
        self.planet_name = planet_name
        self.data = IFUData
        self.output_path = output_path
        self.input_path = input_path
        self.contrast_normalization = 1.0
        return

    @staticmethod
    def init_sphere_dataset(data):
        dataset = SPHERE.Ifs(data.cube,
                            data.psf,
                            data.pa,
                            data.wlen,
                            nan_mask_boxsize=9,
                            psf_cube_size = 15)
        return dataset
    @staticmethod
    def init_gpi_dataset(input_path,
                         science_base_name,
                         data):
        filelist = glob.glob(f"{input_path}*{science_base_name}.fits")
        dataset = GPI.GPIData(filelist,
                                highpass=False,
                                PSF_cube = data.psf,
                                recalc_centers=False)
        band = dataset.prihdrs[0]['APODIZER'].split('_')[1]
        normalization = dataset.spot_ratio[band]
        return dataset, normalization

    def init_dataset(self,
                     data,
                     science_base_name = "distorcorr"):
        if "sphere" in instrument:
            dataset = init_sphere_dataset(data)
        if "gpi" in instrument:
            dataset,norm = init_gpi_dataset(data,
                                            self.input_path,
                                            science_base_name = science_base_name)
            self.contrast_normalization = norm
        return dataset
