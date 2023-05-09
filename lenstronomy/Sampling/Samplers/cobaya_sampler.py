__author__ = 'nataliehogg'

# man with one sampling method always knows his posterior distribution; man with two never certain.

from cobaya.run import run as crun

class CobayaSampler(object):
    """
    Pure Metropolis--Hastings MCMC sampling with Cobaya.

    If you use this sampler, you must cite the following works:

    Lewis & Bridle, https://arxiv.org/abs/astro-ph/0205436

    Lewis, https://arxiv.org/abs/1304.4473

    Torrado & Lewis, https://arxiv.org/abs/2005.05290 and https://ascl.net/1910.019

    For more information about Cobaya, see https://cobaya.readthedocs.io/en/latest/index.html

    """

    # to do:
    # - put all the relevant arguments in the init definition
    # - write the relevant function definitions: run() and log_l
    # - think about MPI functionality?
    # - unit testing!

    def __init__(self, likelihood_module):

        #  get the logL from lenstronomy Likelihood class
        self._ll = likelihood_module

    def run(self, **kwargs):

        logL = self.likelihood

        info = {'likelihood': {'lenstronomy_likelihood': logL}}

        # get all the parameter info from lenstronomy
        self.n_dims, self.param_names = self._ll.param.num_param()
        self._lower_limit, self._upper_limit = self._ll.param.param_limits()

        # going to be so annoying to translate the lenstronomy kwargs to fit here
        # we can do some kind of loop to populate the dictionary
        info['params'] =  {
                           "x": {"prior": {"min": 0, "max": 2}, "ref": 0.5, "proposal": 0.01},
                           "y": {"prior": {"min": 0, "max": 2}, "ref": 0.5, "proposal": 0.01}
                           }

        GR = kwargs['GR']

        info['sampler'] = {'mcmc': {'Rminus1_stop': GR}} # consider what should be hardcoded here or not

        updated_info, sampler = crun(info)

        output = [updated_info, sampler]

        return output

    def likelihood():

        return self._ll.likelihood()
