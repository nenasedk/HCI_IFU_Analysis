import species
import numpy as np
import urllib

input_dir = "/Users/nasedkin/data/HR8799/.../"
filename = "hr8799e.dat"
gravity_wavel, gravity_contrast = np.loadtxt(input_dir + filename, unpack=True)
species.SpeciesInit()

distance = ( 41.2925, 0.1502)  # [pc]

magnitudes = {'TYCHO2/TYCHO2.B': (6.21,0.01),
              'TYCHO2/TYCHO2.V': (5.953,0.010),
              '2MASS/2MASS.J': (5.383, 0.027),
              '2MASS/2MASS.H': (5.280, 0.018),
              '2MASS/2MASS.K': (5.240, 0.018)}

filters = list(magnitudes.keys())

database = species.Database()

database.add_model(model='bt-nextgen',
                    wavel_range=(0.5, 5.),
                    teff_range=(7200., 7600.),
                    spec_res=1000.)


database.add_object(object_name='HD 218396',
                    distance=distance,
                    app_mag=magnitudes)

#database.add_calibration(filename='../data/star_spec_approx.dat',
#                         tag='btnextgen')
urllib.request.urlretrieve('http://irtfweb.ifa.hawaii.edu/~spex/IRTF_Spectral_Library/Data/F0V(n)_HD108519.txt',
database.add_calibration(filename='data/F0V_HD218396.txt',
                         tag='F0V_HD218396')

fit = species.FitSpectrum(object_name='HD 218396',
                          filters=filters,
                          spectrum='btnextgen',
                          bounds={'scaling': (0., 1e1)})

fit.run_mcmc(nwalkers=200,
             nsteps=1000,
             guess={'scaling': 1.},
             tag='hd218396')

species.plot_walkers(tag='hd218396',
                     nsteps=None,
                     offset=(-0.2, -0.08),
                     output='plot_spectrum/walkers.pdf')
species.plot_posterior(tag='hd218396',
                       burnin=500,
                       offset=(-0.3, -0.10),
                       output='plot_spectrum/posteriors.pdf')
objectbox = database.get_object(object_name='HD 218396', 
                                filters=filters)

samples = database.get_mcmc_spectra(tag='hd218396',
                                    burnin=500,
                                    random=30,
                                    wavel_range=(0.2, 30.0),
                                    spec_res=None)

median = database.get_median_sample(tag='hd218396', burnin=500)
print(median)
readcalib = species.ReadCalibration(tag='btnextgen')

specbox = readcalib.get_spectrum(model_param=median)

specbox_gravity = readcalib.resample_spectrum(gravity_wavel, model_param=median)

np.savetxt(output_dir + 'hr8799_star_spec_fit.dat',
           np.column_stack([specbox.wavelength, specbox.flux]),
           header='Wavelength (micron) - Flux (W m-2 micron-1)')

np.savetxt(output_dir + 'hr8799_star_spec_gravity.dat',
           np.column_stack([specbox_gravity.wavelength, specbox_gravity.flux]),
           header='Wavelength (micron) - Flux (W m-2 micron-1)')

synphot = species.multi_photometry(datatype='calibration',
                                   spectrum='btnextgen',
                                   filters=filters,
                                   parameters=median)

residuals = species.get_residuals(datatype='calibration',
                                  spectrum='btnextgen',
                                  parameters=median,
                                  filters=filters,
                                  objectbox=objectbox,
                                  inc_phot=True,
                                  inc_spec=False)
species.plot_spectrum(boxes=[specbox, objectbox, synphot, specbox_gravity],
                      filters=filters,
                      residuals=residuals,
                      # colors=('black', ('tomato', None), 'tomato', 'darkblue'),
                      xlim=(0.2, 30.),
                      ylim=(2e-16, 7e-10),
                      scale=('log', 'log'),
                      title='HD 218396 - BT-NextGen',
                      offset=(-0.3, -0.08),
                      legend='upper right',
                      object_type='star',
                      output='plot_spectrum/hr8799_stellar_spectrum.pdf')
                      
filter_phot = ['Paranal/SPHERE.IRDIS_D_H23_2', 'Paranal/SPHERE.IRDIS_D_H23_3',
               'Paranal/SPHERE.IRDIS_D_K12_1', 'Paranal/SPHERE.IRDIS_D_K12_2']

for i, item in enumerate(filter_phot):
    synphot = species.SyntheticPhotometry(item)

    app_mag, _ = synphot.spectrum_to_magnitude(specbox.wavelength, specbox.flux)
    flux = synphot.spectrum_to_flux(specbox.wavelength, specbox.flux, error=specbox.error)

    print(f'{item} (mag) = {app_mag[0]:.2f}')
    print(f'{item} (W m-2 micron-1) = {flux[0]:.2e}')
