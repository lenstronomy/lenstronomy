__author__ = 'sibirrer'

import numpy as np


def findOverlap(x_mins, y_mins, values, deltapix):
    """
    finds overlapping solutions, deletes multiples and deletes non-solutions and if it is not a solution, deleted as well
    """
    n = len(x_mins)
    idex = []
    for i in range(n):
        if i==0:
            if values[0] > deltapix/100.:
                idex.append(i)
        else:
            for j in range(0,i):
                if ((abs(x_mins[i]-x_mins[j])<deltapix and abs(y_mins[i]-y_mins[j])<deltapix) or values[i]>deltapix/100.):
                    idex.append(i)
                    break
    x_mins = np.delete(x_mins, idex, axis=0)
    y_mins = np.delete(y_mins, idex, axis=0)
    values = np.delete(values, idex, axis=0)
    return x_mins, y_mins, values


def coordInImage(x_coord, y_coord, numPix, deltapix):
    """
    checks whether image positions are within the pixel image in units of arcsec
    if not: remove it

    :param imcoord: image coordinate (in units of angels)  [[x,y,delta,magnification][...]]
    :type imcoord: (n,4) numpy array
    :returns: image positions within the pixel image
    """
    idex=[]
    min = -deltapix*numPix/2
    max = deltapix*numPix/2
    for i in range(len(x_coord)): #sum over image positions
        if (x_coord[i] < min or x_coord[i] > max or y_coord[i] < min or y_coord[i] > max):
            idex.append(i)
    x_coord = np.delete(x_coord, idex, axis=0)
    y_coord = np.delete(y_coord, idex, axis=0)
    return x_coord, y_coord