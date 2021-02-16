# HCI IFU Analysis
A structured format for analysing high contrast imaging IFU data with a variety of algorithms. The goal is to generate consistent outputs for different algorithms (PCA, KLIP and ANDROMEDA), producing for each a spectrum with an associated covariance matrix.

All algorithms require the same parameters to run, in order:

    data_dir : The path to the directory where the science, psf, parallactic angle and wavelength data is stored.
    instrument : the name of the instrument. Must include the instrument name (SPHERE,GPI), but should include the band (Y,J,H,K) etc.
    planet_name : The name of the planet, eg HR8799e
    posn : an initial guess for the separation [mas], parallactic angle [deg] and flux [contrast]. 


