import sphere.IFS as IFS

data_dir = "/u/nnas/data/HR8799/SPHERE-0101C0315A-18/"
#%% init reduction
reduction = IFS.Reduction(data_dir, log_level='info')

#%% configuration
reduction.config['preproc_collapse_science'] = False
reduction.show_config()
reduction._files_info

####################################################
# manual reduction
#
####################################################

#%% init reduction
reduction = IFS.Reduction(data_dir, log_level='info')

#%% sorting
reduction.sort_files()
reduction.sort_frames()
reduction.check_files_association()

#%% static calibrations
reduction.sph_ifs_cal_dark(silent=False)
reduction.sph_ifs_cal_detector_flat(silent=False)
reduction.sph_ifs_cal_specpos(silent=True)
reduction.sph_ifs_cal_wave(silent=True)
reduction.sph_ifs_cal_ifu_flat(silent=False)

#%% science pre-processing
reduction.sph_ifs_preprocess_science(subtract_background=True, fix_badpix=True, correct_xtalk=True,
                                     collapse_science=False, collapse_type='mean', coadd_value=2,
                                     collapse_psf=False, collapse_center=False)
reduction.sph_ifs_preprocess_wave()
reduction.sph_ifs_science_cubes(silent=True)

#%% high-level science processing
reduction.sph_ifs_wavelength_recalibration(high_pass=True, offset=(-3, 0), plot=False)
reduction.sph_ifs_star_center(high_pass=True, offset=(-3, 0), plot=False)
reduction.sph_ifs_combine_data(cpix=True, psf_dim=80, science_dim=200, correct_anamorphism=True,
                               shift_method='fft', manual_center=None, coarse_centering=False,
                               save_scaled=False)

#%% cleaning
#reduction.sph_ifs_clean(delete_raw=False, delete_products=False)