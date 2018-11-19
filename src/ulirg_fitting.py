
# coding: utf-8

# # Minimization using scipy.minimize
# ### Using dowhill simplex method defined by python funciton
# ```
#     >>> scipy.optimize.minimize(method = 'Nelder-mead')
# ```
# ## spectral form 
# $$ f(\lambda) = m \left(\frac{\lambda}{1407}\right)^{b} + A*\frac{1}{\sqrt{2\pi\sigma^2}}exp[-(\lambda -\lambda_0)/(2\sigma^2)] $$
# $$\sigma =1, \lambda_0 = 1215.67\times(1.158), $$

# In[8]:


### imports ....


from astropy.io import ascii
from astropy.io import fits
import pysynphot as S
import warnings
import scipy
warnings.simplefilter('ignore')
from collections import OrderedDict
import time
import numpy as np


class grid_search:
    def __init__(self, x, x0, sigma, bands, wave):
        self.x = x
        self.x0 = x0
        self.sig = sigma
        self.bands = bands
        self.wave = wave

    def gauss(self, x, x0, sig):
        return 1 / np.sqrt(2 * np.pi * sig**2) * np.exp(-0.5 * (x - x0)**2 / sig**2)

    def spec_calc(self, m, b, A, wave):
        lya = self.x0
        sigma = self.sig
        spec = m * (wave / 1407 ) ** b + A * self.gauss(wave, lya, sigma)
        return spec

    ### function to get counts give a spectrum
    
    def counts_syn(self, m, b, A, wave, bands):

        spec = self.spec_calc(m, b, A, wave)
        sp = S.ArraySpectrum(wave, spec, waveunits='angstrom', fluxunits='flam', name='MySource', keepneg=True)
        obs = [S.Observation(sp, band) for band in bands]
        try:
            
            counts = np.array([ob.effstim('counts') for ob in obs])
        except ValueError :
            print ('total flux negative', m,b,A)
            counts = np.array([0.0,0.0, 0.0, 0.0])
        return counts,spec


    #### residiuls
    def residual(self, params, y, yerr):

        m, b, A = params['m'], params['b'], params['A']
        model = self.counts_syn(m, b, A, self.wave, self.bands)
        res = [(model[i] -y[i])/yerr[i] for i in range (4)]
        return  abs(np.array(res))#((model -y)/y)





def counts_out(m,b,A):
    x = np.array([1438.19187645, 1527.99574111, 1612.22929754, 1762.54619064])
    red_shift = 0.158
    #### input params###
    sigma = 4
    #### bandpass calculations###
    band_names = ['f125lp', 'f140lp', 'f150lp', 'f165lp']

    bands = [S.ObsBandpass('acs,sbc,%s' % band) for band in band_names]
    waves = [band.wave for band in bands]
    lya = 1215.67 * (1 + red_shift)
    wave1 = np.arange(1000, 10000, 1.0)

    grid_new = grid_search(x, lya, sigma, bands, wave1)
    

    m1 = 100
    cent_x = 550
    cent_y = 550
    counts,spec = grid_new.counts_syn(m, b, A, wave1, bands)

    return counts,spec