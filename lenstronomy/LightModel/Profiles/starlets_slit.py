# starlet transform from SLIT

import numpy as np
import scipy.signal as scs
import scipy.ndimage.filters as scf


def _transform(self, img, n_scales, Filter='Bspline', convol2d=0, second_gen=False):

    mode = 'nearest'
    
    lvl = n_scales-1
    sh = np.shape(img)
    if np.size(sh) == 3:
        mn = np.min(sh)
        wave = np.zeros([lvl+1, sh[1], sh[1], mn])
        for h in range(mn):
            if mn == sh[0]:
                wave[:, :, :, h] = wave_transform(img[h,:,:], lvl+1, Filter=Filter)
            else:
                wave[:, :, :, h] = wave_transform(img[:,:,h], lvl+1, Filter=Filter)
        return wave

    n1 = sh[1]
    n2 = sh[1]
    
    if Filter == 'Bspline':
        h = [1./16, 1./4, 3./8, 1./4, 1./16]
    else:
        h = [1./4, 1./2, 1./4]
    n = np.size(h)
    h = np.array(h)
    
    max_lvl = np.min( (lvl, int(np.log2(n2))) )
    if lvl > max_lvl:
        lvl = max_lvl
        print("Warning : lvl set to {}".format(lvl))

    c = img
    ## wavelet set of coefficients.
    wave = np.zeros((lvl+1, n1, n2))

    for i in range(lvl):
        newh = np.zeros((1, n+(n-1)*(2**i-1)))
        newh[0, np.linspace(0, np.size(newh)-1, len(h), dtype=int)] = h

        H = np.dot(newh.T, newh)

        ######Calculates c(j+1)
        ###### Line convolution
        if convol2d == 1:
            cnew = scs.convolve2d(c, H, mode='same', boundary='symm')
        else:
            cnew = scf.convolve1d(c, newh[0, :], axis=0, mode=mode)

            ###### Column convolution
            cnew = scf.convolve1d(cnew, newh[0,:],axis=1, mode=mode)

 
      
        if second_gen:
            ###### hoh for g; Column convolution
            if convol2d == 1:
                hc = scs.convolve2d(cnew, H, mode='same', boundary='symm')
            else:
                hc = scf.convolve1d(cnew, newh[0, :],axis=0, mode=mode)
 
                ###### hoh for g; Line convolution
                hc = scf.convolve1d(hc, newh[0, :],axis=1, mode=mode)
            
            ###### wj+1 = cj - hcj+1
            wave[i, :, :] = c - hc
            
        else:
            ###### wj+1 = cj - cj+1
            wave[i, :, :] = c - cnew


        c = cnew
     
    wave[i+1, :, :] = c

    return wave


def _inverse_transform(self, wave, convol2d=0, fast=True, second_gen=False):
    if fast and not second_gen:
        # simply sum all scales, including the coarsest one
        return np.sum(wave, axis=0)

    mode = 'nearest'
    
    lvl, n1, n2 = np.shape(wave)
    h = np.array([1./16, 1./4, 3./8, 1./4, 1./16])
    n = np.size(h)

    cJ = np.copy(wave[lvl-1, :, :])
    

    for i in range(1, lvl):
        
        newh = np.zeros((1, n+(n-1)*(2**(lvl-1-i)-1)))
        newh[0, np.linspace(0, np.size(newh)-1, len(h), dtype=int)] = h
        H = np.dot(newh.T, newh)

        ###### Line convolution
        if convol2d == 1:
            cnew = scs.convolve2d(cJ, H, mode='same', boundary='symm')
        else:
            cnew = scf.convolve1d(cJ, newh[0, :], axis=0, mode=mode)
            ###### Column convolution
            cnew = scf.convolve1d(cnew, newh[0, :], axis=1, mode=mode)

        cJ = cnew + wave[lvl-1-i, :, :]

    return np.reshape(cJ, (n1, n2))
