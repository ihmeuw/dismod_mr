# Copyright 2008-2019 University of Washington
#
# This file is part of DisMod-MR.
#
# DisMod-MR is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DisMod-MR is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with DisMod-MR.  If not, see <http://www.gnu.org/licenses/>.
import pandas as pd
import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d


# @numba.jit(nopython=True)
def odefun(a, y, age, incidence, remission, excess, all_cause):
    i = interp1d(age, incidence, kind='linear')
    r = interp1d(age, remission, kind='linear')
    e = interp1d(age, excess, kind='linear')
    m = interp1d(age, all_cause, kind='linear')

    s, c = y

    ds_da = - (i(a) + (m(a) - e(a) * s / (s + c))) * s + r(a) * c
    dc_da = i(a) * s - (r(a) + (m(a) - e(a) * s / (s + c)) + e(a)) * c

    return [ds_da, dc_da]


def f(a, susceptible_condition,
        incidence, remission, excess, all_cause):
    s = susceptible_condition[0]
    c = susceptible_condition[1]
    i = incidence[int(a)]
    r = remission[int(a)]
    e = excess[int(a)]
    m = all_cause[int(a)]
    other = m - e * s / (s + c)
    ds_da = - (i + other) * s + r * c
    dc_da = +           i * s - (r + other + e) * c
    return np.array([ds_da, dc_da])


def ode_function(susceptible, condition, num_step, age_local, all_cause, incidence, remission, excess, s0, c0, scipy=False):
    if not isinstance(scipy, bool):
        raise Exception('scipy flag in ode_function must be of type bool')

    t0 = time.time()
    age = age_local
    N = len(age)
    susceptible[0] = s0
    condition[0] = c0
    sc = np.array([s0, c0])

    if scipy == False:
        for j in range(N-1):
            a_step = (age[j+1] - age[j]) / num_step
            a_tmp = age[j]

            dt = a_step
            ti = a_tmp
            yi = sc

            for step in range(num_step):
                # copied from http://www.seanet.com/~bradbell/pycppad/runge_kutta_4.xml
                k1 = dt * f(ti, yi, incidence, remission, excess, all_cause)
                k2 = dt * f(ti + .5*dt, yi + .5*k1, incidence, remission, excess, all_cause)
                k3 = dt * f(ti + .5*dt, yi + .5*k2, incidence, remission, excess, all_cause)
                k4 = dt * f(ti + dt, yi + k3, incidence, remission, excess, all_cause)
                yf = yi + (1./6.) * (k1 + 2.*k2 + 2.*k3 + k4)
                sc = yf
                a_tmp = a_tmp + a_step

            susceptible[j+1] = sc[0]
            condition[j+1] = sc[1]

    if scipy == True:
        res = integrate.solve_ivp(lambda a, y: odefun(a, y, age, incidence, remission, excess, all_cause), t_span=(
            age[0], age[-1]), y0=[s0, c0], method='RK23', t_eval=age)

        susceptible = res.y[0, :]
        condition = res.y[1, :]
