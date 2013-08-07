""" Spline model used for age-specific rates"""

import pylab as pl
import pymc as mc


def spline(name, ages, knots, smoothing, interpolation_method='linear'):
    """ Generate PyMC objects for a spline model of age-specific rate

    Parameters
    ----------
    name : str
    knots : array
    ages : array, points to interpolate to
    smoothing : pymc.Node, smoothness parameter for smoothing spline
    interpolation_method : str, optional, one of 'linear', 'nearest', 'zero', 'slinear', 'quadratic, 'cubic'

    Results
    -------
    Returns dict of PyMC objects, including 'gamma' (log of rate at
    knots) and 'mu_age' (age-specific rate interpolated at all age
    points)
    """
    assert pl.all(pl.diff(knots) > 0), 'Spline knots must be strictly increasing'

    # TODO: consider changing this prior distribution to be something more familiar in linear space
    gamma = [mc.Normal('gamma_%s_%d'%(name,k), 0., 10.**-2, value=-10.) for k in knots]
    #gamma = [mc.Uniform('gamma_%s_%d'%(name,k), -20., 20., value=-10.) for k in knots]

    # TODO: fix AdaptiveMetropolis so that this is not necessary
    flat_gamma = mc.Lambda('flat_gamma_%s'%name, lambda gamma=gamma: pl.array([x for x in pl.flatten(gamma)]))


    import scipy.interpolate
    @mc.deterministic(name='mu_age_%s'%name)
    def mu_age(gamma=flat_gamma, knots=knots, ages=ages):
        mu = scipy.interpolate.interp1d(knots, pl.exp(gamma), kind=interpolation_method, bounds_error=False, fill_value=0.)
        return mu(ages)

    vars = dict(gamma=gamma, mu_age=mu_age, ages=ages, knots=knots)

    if (smoothing > 0) and (not pl.isinf(smoothing)):
        #print 'adding smoothing of', smoothing
        @mc.potential(name='smooth_mu_%s'%name)
        def smooth_gamma(gamma=flat_gamma, knots=knots, tau=smoothing**-2):
            # the following is to include a "noise floor" so that level value
            # zero prior does not exert undue influence on age pattern
            # smoothing
            # TODO: consider changing this to an offset log normal
            gamma = gamma.clip(pl.log(pl.exp(gamma).mean()/10.), pl.inf)  # only include smoothing on values within 10x of mean

            return mc.normal_like(pl.sqrt(pl.sum(pl.diff(gamma)**2 / pl.diff(knots))), 0, tau)
        vars['smooth_gamma'] = smooth_gamma

    return vars
