import numpy as np
import astrofunc.util as util
import lenstronomy.util as lenstronomy_util


class ImagePosition(object):
    """
    class to solve for image positions given lens model and source position
    """
    def __init__(self, lensModel):
        """

        :param imsim: imsim class
        """
        self.LensModel = lensModel

    def image_position(self, sourcePos_x, sourcePos_y, deltapix, numPix, kwargs_lens, kwargs_else=None):
        """
        finds image position and magnification given source position and lense model

        :param sourcePos: source position in units of angel
        :type sourcePos: numpy array
        :param args: contains all the lens model parameters
        :type args: variable length depending on lense model
        :returns:  (exact) angular position of (multiple) images [[posAngel,delta,mag]] (in pixel image , including outside)
        :raises: AttributeError, KeyError
        """
        x_grid, y_grid = util.make_grid(numPix, deltapix)
        x_mapped, y_mapped = self.LensModel.ray_shooting(x_grid, y_grid, kwargs_lens, kwargs_else)
        absmapped = util.displaceAbs(x_mapped, y_mapped, sourcePos_x, sourcePos_y)
        x_mins, y_mins, values = util.neighborSelect(absmapped, x_grid, y_grid)
        if x_mins == []:
            return None, None
        num_iter = 1000
        x_mins, y_mins, values = self._findIterative(x_mins, y_mins, sourcePos_x, sourcePos_y, deltapix, num_iter, kwargs_lens, kwargs_else)
        x_mins, y_mins, values = lenstronomy_util.findOverlap(x_mins, y_mins, values, deltapix)
        x_mins, y_mins = lenstronomy_util.coordInImage(x_mins, y_mins, numPix, deltapix)
        if x_mins == []:
            return None, None
        return x_mins, y_mins

    def _findIterative(self, x_min, y_min, sourcePos_x, sourcePos_y, deltapix, num_iter, kwargs_lens, kwargs_else=None):
        """
        find iterative solution to the demanded level of precision for the pre-selected regions given a lense model and source position

        :param mins: indices of local minimas found with def neighborSelect and def valueSelect
        :type mins: 1d numpy array
        :returns:  (n,3) numpy array with exact position, displacement and magnification [posAngel,delta,mag]
        :raises: AttributeError, KeyError
        """
        num_candidates = len(x_min)
        x_mins = np.zeros(num_candidates)
        y_mins = np.zeros(num_candidates)
        values = np.zeros(num_candidates)
        for i in range(len(x_min)):
            l = 0
            x_mapped, y_mapped = self.LensModel.ray_shooting(x_min[i], y_min[i], kwargs_lens, kwargs_else)
            delta = np.sqrt((x_mapped - sourcePos_x)**2+(y_mapped - sourcePos_y)**2)
            potential, alpha1, alpha2, kappa, gamma1, gamma2, mag = self.LensModel.all(x_min[i], y_min[i], kwargs_lens, kwargs_else)
            DistMatrix = np.array([[1-kappa+gamma1, gamma2], [gamma2, 1-kappa-gamma1]])
            det = 1./mag
            posAngel = np.array([x_min[i], y_min[i]])
            while(delta > deltapix/100000 and l<num_iter):
                deltaVec = np.array([x_mapped - sourcePos_x, y_mapped - sourcePos_y])
                posAngel = posAngel - DistMatrix.dot(deltaVec)/det
                x_mapped, y_mapped = self.LensModel.ray_shooting(posAngel[0], posAngel[1], kwargs_lens, kwargs_else)
                delta = np.sqrt((x_mapped - sourcePos_x)**2+(y_mapped - sourcePos_y)**2)
                potential, alpha1, alpha2, kappa, gamma1, gamma2, mag = self.LensModel.all(posAngel[0], posAngel[1], kwargs_lens, kwargs_else)
                DistMatrix=np.array([[1-kappa+gamma1, gamma2], [gamma2, 1-kappa-gamma1]])
                det=1./mag
                l+=1
            x_mins[i] = posAngel[0]
            y_mins[i] = posAngel[1]
            values[i] = delta
        return x_mins, y_mins, values

    def findBrightImage(self, sourcePos_x, sourcePos_y, kwargs_lens, deltapix, numPix, magThresh=1., numImage=4, kwargs_else=None):
        """

        :param sourcePos_x:
        :param sourcePos_y:
        :param deltapix:
        :param numPix:
        :param magThresh: magnification threshold for images to be selected
        :param numImage: number of selected images (will select the highest magnified ones)
        :param kwargs_lens:
        :return:
        """
        x_mins, y_mins = self.image_position(sourcePos_x, sourcePos_y, deltapix, numPix, kwargs_lens, kwargs_else)
        mag_list = []
        for i in range(len(x_mins)):
            potential, alpha1, alpha2, kappa, gamma1, gamma2, mag = self.LensModel.all(x_mins[i], y_mins[i], kwargs_lens, kwargs_else)
            mag_list.append(abs(mag))
        mag_list = np.array(mag_list)
        x_mins_sorted = util.selectBest(x_mins, mag_list, numImage)
        y_mins_sorted = util.selectBest(y_mins, mag_list, numImage)
        return x_mins_sorted, y_mins_sorted