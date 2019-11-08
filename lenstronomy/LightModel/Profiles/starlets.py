__author__ = 'aymgal'

import numpy as np
# import pysap
import scipy.signal as scs
import scipy.ndimage.filters as scf


class Starlets(object):
    """

    """
    param_names = ['amp', 'n_scales']
    lower_limit_default = {'amp': 0, 'n_scales': 2}
    upper_limit_default = {'amp': 1e6, 'n_scales': 20}

    def __init__(self, thread_count=1, fast_inverse=True, second_gen=True):
        # self._transf_class = pysap.load_transform('BsplineWaveletTransformATrousAlgorithm')
        self._thread_count = thread_count
        self._fast_inverse = fast_inverse
        self._second_gen = second_gen

    def function(self, x, y, amp, n_scales):
        """return inverse starlet transform from starlet coefficients stored in amp"""
        # return self._inverse_transform(amp, n_scales)
        return self._iuwt_slit(amp, n_scales)

    def decomposition(self, image, x, y, n_scales):
        """
        decomposes an image into starlet coefficients
        :return:
        """
        # return self._transform(image, n_scales)
        return self._uwt_slit(image, n_scales)

    def _inverse_transform_pysap(self, amp, n_scales):
        """performs inverse starlet transform"""
        self._check_transform_pysap(n_scales)
        if self._fast_inverse:
            # for 1st gen starlet the reconstruction can be performed by summing all scales 
            image = np.sum(amp, axis=0)
        else:
            coeffs = self._amp2pysap(amp)
            self._transf.analysis_data = coeffs
            result = transform.synthesis()
            result.show()
            image = result.data
        return image

    def _transform_pysap(self, image, n_scales):
        """
        decomposes an image into starlets coefficients
        """
        self._check_transform_pysap(n_scales)
        self._transf.data = image
        self._transf.analysis()
        self._transf.show()
        coeffs = self._transf.analysis_data
        amp = self._pysap2amp(coeffs)
        return amp

    def _check_transform_pysap(self, n_scales):
        if not hasattr(self, '_transf') or n_scales != self._n_scales:
            self._transf = self._transf_class(nb_scale=n_scales, verbose=False, 
                                              nb_procs=self._thread_count)
            self._n_scales = n_scales

    def _pysap2amp(self, coeffs):
        return np.asarray(coeffs)

    def _amp2pysap(self, amp):
        coeffs = []
        for i in range(amp.shape[0]):
            coeffs.append(amp[i, :, :])
        return coeffs

    def _uwt_slit(self, img, n_scales, Filter='Bspline', convol2d=0):

        mode = 'nearest'
        
        lvl = n_scales-1
        sh = np.shape(img)
        if np.size(sh) ==3:
            mn = np.min(sh)
            wave = np.zeros([lvl+1,sh[1], sh[1],mn])
            for h in range(mn):
                if mn == sh[0]:
                    wave[:,:,:,h] = wave_transform(img[h,:,:],lvl+1, Filter = Filter)
                else:
                    wave[:,:,:,h] = wave_transform(img[:,:,h],lvl+1, Filter = Filter)
            return wave

        n1 = sh[1]
        n2 = sh[1]
        
        if Filter == 'Bspline':
            h = [1./16, 1./4, 3./8, 1./4, 1./16]
        else:
            h = [1./4,1./2,1./4]
        n = np.size(h)
        h = np.array(h)
        
        if lvl > np.min( (lvl, int(np.log2(n2))) ):
            lvl = np.min( (lvl, int(np.log2(n2))) )
            print("Warning : lvl set to {}".format(lvl))

        c = img
        ## wavelet set of coefficients.
        wave = np.zeros((lvl+1, n1, n2))

        for i in range(lvl):
            newh = np.zeros((1, n+(n-1)*(2**i-1)))
            newh[0, np.linspace(0,np.size(newh)-1, len(h), dtype=int)] = h

            H = np.dot(newh.T,newh)

            ######Calculates c(j+1)
            ###### Line convolution
            if convol2d == 1:
                cnew = scs.convolve2d(c, H, mode='same', boundary='symm')
            else:
                cnew = scf.convolve1d(c,newh[0,:],axis = 0, mode =mode)

                ###### Column convolution
                cnew = scf.convolve1d(cnew,newh[0,:],axis = 1, mode =mode)

     
          
            if self._second_gen:
                ###### hoh for g; Column convolution
                if convol2d == 1:
                    hc = scs.convolve2d(cnew, H, mode='same', boundary='symm')
                else:
                    hc = scf.convolve1d(cnew,newh[0,:],axis = 0, mode = mode)
     
                    ###### hoh for g; Line convolution
                    hc = scf.convolve1d(hc,newh[0,:],axis = 1, mode = mode)
                
                ###### wj+1 = cj-hcj+1
                wave[i,:,:] = c-hc
                
            else:
                ###### wj+1 = cj-cj+1
                wave[i,:,:] = c-cnew


            c = cnew
         
        wave[i+1,:,:] = c

        return wave

    def _iuwt_slit(self, wave, convol2d=0, fast=True):
        if not self._second_gen and self._fast_inverse:
            # simply sum all scales, including the coarsest one
            return np.sum(wave, axis=0)

        mode = 'nearest'
        
        lvl,n1,n2 = np.shape(wave)
        h = np.array([1./16, 1./4, 3./8, 1./4, 1./16])
        n = np.size(h)

        cJ = np.copy(wave[lvl-1,:,:])
        

        for i in range(1, lvl):
            
            newh = np.zeros((1,n+(n-1)*(2**(lvl-1-i)-1)))
            newh[0, np.linspace(0,np.size(newh)-1,len(h), dtype=int)] = h
            H = np.dot(newh.T,newh)

            ###### Line convolution
            if convol2d == 1:
                cnew = scs.convolve2d(cJ, H, mode='same', boundary='symm')
            else:
              cnew = scf.convolve1d(cJ,newh[0,:],axis = 0, mode = mode)
                ###### Column convolution
              cnew = scf.convolve1d(cnew,newh[0,:],axis = 1, mode = mode)

            cJ = cnew+wave[lvl-1-i,:,:]

        return np.reshape(cJ,(n1,n2))
