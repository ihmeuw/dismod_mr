# $begin dismod_ode$$ $newlinech #$$
# $spell
#	subvector
#	adfun
#	numpy
#	Dismod
#	pycppad
#	num
# $$
# $latex \newcommand{\R}[1]{{\rm #1}}$$
# $latex \newcommand{\B}[1]{{\bf #1}}$$
#
# $section Create a Pycppad Function Object That Integrates The Dismod ODE$$
#
# $head Syntax$$
# $codei%from dismod_ode import ode_function
# %fun% = ode_function(%num_step%, %age%, %all_cause%)%$$
#
# $head Purpose$$
# Creates a
# $href%http://www.seanet.com/~bradbell/pycppad/adfun.xml%pycppad.adfun%$$
# object that can be used for fast solutions of the Dismod ODE
# (and derivatives of solutions of the Dismod ODE).
#
# $head Dismod ODE$$
# The Dismod ODE is defined by
# $latex \[
# \begin{array}{rcrr}
# S'(a) & = & - [ \iota (a) + \omega (a) ] S(a) & + \rho(a) C (a) 
# \\
# C'(a) & = & + \iota(a)  S(a) & - [ \omega (a) + \chi (a) ] C(a)
# \end{array}
# \] $$
#
# $head num_step$$
# is an integer scalar that specifies the number of integration steps
# to take for each interval 
# $codei%
#	%[ age%[%j%+1] - %age%[%j%]  %]
# %$$
#
# $head age$$
# is $code numpy.array$$ of $code float$$ values.
# This specifies the age values for which a solution of the ODE is computed.
# The constant $icode N$$ below is the length of $icode age$$.
#
# $head all_cause$$
# is $code numpy.array$$ of $code float$$ values with length $icode N$$.
# This specifies all cause mortality on the grid of age values; i.e.,
# $icode%all_cause[%j%]%$$ is the value of
# $latex \[
#	[ \iota( a_j ) + \omega( a_j ) ] S( a_j ) 
#	+ 
#	[ \iota( a_j ) + \omega( a_j ) + \chi( a_j ) ] C( a_j )
# \] $$
# where $latex a_j$$ denotes the value $icode%age[%j%]%$$.
#
# $head fun$$
# is a $code pycppad.adfun$$ object that maps $latex y = f(x)$$
# where $latex x \in \B{R}^{3 N + 2}$$, $latex y \in \B{R}^{2 N}$$ are
# defined below:
#
# $subhead x$$
# The vector $latex x \in \B{R}^{3 N + 2}$$ is defined by
# $latex \[
# \begin{array}{rcl}
#	i & = & [ \iota ( a_0 ) , \ldots , \iota ( a_{N-1} ) ] \\
#	r & = & [ \rho ( a_0 ) , \ldots , \rho ( a_{N-1} ) ] \\
#	e & = & [ \chi ( a_0 ) , \ldots , \chi ( a_{N-1} ) ] \\
#	x & = & [ i , r , e , S( a_0 ) , C( a_0 ) ]
# \end{array}
# \] $$
# where $icode i$$ is the subvector of incidence values,
# $icode r$$ is the subvector of remission values,
# and $icode e$$ is the subvector of excess mortality values,
#
# $subhead y$$
# The vector $latex y \in \B{R}^{2 N}$$ is defined by
# $latex \[
# \begin{array}{rcl}
#	s & = & [ S ( a_0 ) , \ldots , S ( a_{N-1} ) ] \\
#	c & = & [ C ( a_0 ) , \ldots , C ( a_{N-1} ) ] \\
#	y & = & [ s , c ]
# \end{array}
# \] $$
# where $icode s$$ is the subvector of susceptible values,
# and $icode c$$ is the subvector of with condition values.
#
# $head Approximation$$
# The functions $latex \iota(a)$$, $latex \rho (a)$$,
# $latex \chi(a)$$ and $latex \omega(a)$$ are approximated by
# piecewise constants using the initial value for each interval.
# This may change in future implementations; e.g., perhaps a piecewise
# linear approximation will be used in the future.
#
# $children%
#	dismod_ode_test.py
# %$$
# $head Example$$
# The routine $cref dismod_ode_test$$ is an example and test of 
# $code dismod_ode$$.
#
# $end
import numpy
import pycppad

N = 0

# copied from http://www.seanet.com/~bradbell/pycppad/runge_kutta_4.xml
def runge_kutta_4(f, ti, yi, dt) :
	k1 = dt * f(ti         , yi)
	k2 = dt * f(ti + .5*dt , yi + .5*k1) 
	k3 = dt * f(ti + .5*dt , yi + .5*k2) 
	k4 = dt * f(ti + dt    , yi + k3)
	yf = yi + (1./6.) * ( k1 + 2.*k2 + 2.*k3 + k4 )
	return yf 

#

def ode_fun(a, susceptible_condition) :
	global age, incidence, remission, excess, all_cause
	s      = susceptible_condition[0]
	c      = susceptible_condition[1]
	i      = incidence[a]
	r      = remission[a]
	e      = excess[a]
	m      = all_cause[a];
	other  = m - e * s / (s + c)
	ds_da  = - (i + other) * s +              r  * c
	dc_da  = +           i * s - (r + other + e) * c
	return numpy.array( [ ds_da , dc_da ] )

def ode_integrate(N, num_step, s0, c0) :
	global age, incidence, remission, excess, all_cause
	global susceptible, condition
	susceptible[0] = s0
	condition[0]   = c0
	sc             = numpy.array( [s0, c0] )
	N              = len( all_cause )
	for j in range(N-1) :
		a_step = (age[j+1] - age[j]) / num_step
		a_tmp  = age[j]
		for step in range(num_step) :
			sc    = runge_kutta_4(ode_fun, a_tmp, sc, a_step)
			a_tmp = a_tmp + a_step
		susceptible[j+1] = sc[0]
		condition[j+1]   = sc[1]

def ode_function(num_step, age_local, all_local) :
	global age, incidence, remission, excess, all_cause
	global susceptible, condition
	N            = len( age_local )
	age          = age_local
	all_cause    = all_local
	susceptible  = pycppad.ad( numpy.zeros(N) )
	condition    = pycppad.ad( numpy.zeros(N) )
	incidence    = .00 * numpy.ones(N)
	remission    = .00 * numpy.ones(N)
	excess       = .00 * numpy.ones(N)
	s0           = 0.
	c0           = 0.
	x            = numpy.hstack( (incidence, remission, excess, s0, c0) )
	x            = pycppad.independent( x )
	incidence    = x[(0*N):(1*N)]
	remission    = x[(1*N):(2*N)]
	excess       = x[(2*N):(3*N)]
	s0           = x[3*N]
	c0           = x[3*N+1]
	ode_integrate(N, num_step, s0, c0)
	y            = numpy.hstack( (susceptible, condition) )
	fun          = pycppad.adfun(x, y)
	return fun
