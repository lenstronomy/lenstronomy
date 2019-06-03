__author__ = 'dgilman'

import numpy as np
from copy import deepcopy

class PhysicalLocation(object):

    def __call__(self, kwargs_lens):

        return kwargs_lens

class LensedLocation(object):

    def __init__(self, multiplane_instance, index):

        self._multiplane = multiplane_instance

        if len(index) == 1:
            self._inds = index
        else:
            inds = np.array(index)
            z = []

            for ind in inds:
                z.append(multiplane_instance._redshift_list[ind])

            sort = np.argsort(z)

            self._inds = inds[sort]

    def __call__(self, kwargs_lens):

        new_kwargs = deepcopy(kwargs_lens)

        for ind in self._inds:

            theta_x = kwargs_lens[ind]['center_x']
            theta_y = kwargs_lens[ind]['center_y']
            zstop = self._multiplane._redshift_list[ind]
            x, y, _, _ = self._multiplane.ray_shooting_partial(0, 0, theta_x,
                            theta_y, 0, zstop, kwargs_lens, check_convention = False)

            D = self._multiplane._T_z_list[ind]
            new_kwargs[ind]['center_x'] = x / D
            new_kwargs[ind]['center_y'] = y / D

        return new_kwargs