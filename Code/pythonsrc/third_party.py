# Copyright 2018 Constantinos Papayiannis
#
# This file is part of Reverberation Learning Toolbox for Python.
#
# Reverberation Learning Toolbox for Python is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Reverberation Learning Toolbox for Python is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Reverberation Learning Toolbox for Python.  If not, see <http://www.gnu.org/licenses/>.

"""

This file contains third party code used for the project.

This file was original distributed in the repository at:
{repo}

"""

from __future__ import print_function, division

import sys


#### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### ####   Part 1  #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### ####
# From https://github.com/awesomebytes/parametric_modeling/blob/master/src, Sam Pfeiffer,
# @awesomebytes.github, Unknown license

def convm(x, p):
    """

    ---

    This code is taken from:

    Notes from the distribution:
     This file is a Python translation of the MATLAB file convm.m
     Python version by RDL 10 Jan 2012
     Copyright notice from convm.m:
     copyright 1996, by M.H. Hayes.  For use with the book
     "Statistical Digital Signal Processing and Modeling"
     (John Wiley & Sons, 1996).

     ---

    Generates a convolution matrix

    Usage: X = convm(x,p)
    Given a vector x of length N, an N+p-1 by p convolution matrix is
    generated of the following form:
              |  x(0)  0      0     ...      0    |
              |  x(1) x(0)    0     ...      0    |
              |  x(2) x(1)   x(0)   ...      0    |
         X =  |   .    .      .              .    |
              |   .    .      .              .    |
              |   .    .      .              .    |
              |  x(N) x(N-1) x(N-2) ...  x(N-p+1) |
              |   0   x(N)   x(N-1) ...  x(N-p+2) |
              |   .    .      .              .    |
              |   .    .      .              .    |
              |   0    0      0     ...    x(N)   |

    That is, x is assumed to be causal, and zero-valued after N.
    """
    N = len(x) + 2 * p - 2
    xpad = np.concatenate([np.zeros(p - 1), x[:], np.zeros(p - 1)])
    X = np.zeros((len(x) + p - 1, p))
    # Construct X column by column
    for i in xrange(p):
        X[:, i] = xpad[p - i - 1:N - i]

    return X


def prony(x, p, q):
    """

    ---

    This code is taken from :

    Notes from the distribution:
    This file is a Python translation of the MATLAB file prony.m

     Python version by RDL 12 Jan 2012
     Copyright notice from prony.m:
     copyright 1996, by M.H. Hayes.  For use with the book
     "Statistical Digital Signal Processing and Modeling"
     (John Wiley & Sons, 1996).

     ---

    Model a signal using Prony's method

    Usage: [b,a,err] = prony(x,p,q)

    The input sequence x is modeled as the unit sample response of
    a filter having a system function of the form
        H(z) = B(z)/A(z)
    The polynomials B(z) and A(z) are formed from the vectors
        b=[b(0), b(1), ... b(q)]
        a=[1   , a(1), ... a(p)]
    The input q defines the number of zeros in the model
    and p defines the number of poles. The modeling error is
    returned in err.

    This comes from Hayes, p. 149, 153, etc

    """
    x = x[:]
    N = len(x)
    if p + q >= len(x):
        print('ERROR: model order too large')
        print("p q len(x) " + str(p) + " " + str(q) + " " + str(len(x)))
        sys.exit(1)

    # This formulation uses eq. 4.50, p. 153
    # Set up the convolution matrices
    X = convm(x, p + 1)
    Xq = X[q:N + p - 1, 0:p]
    xq1 = -X[q + 1:N + p, 0]

    # Solve for denominator coefficients
    if p > 0:
        a = np.linalg.lstsq(Xq, xq1)[0]
        a = np.insert(a, 0, 1)  # a(0) is 1
    else:
        # all-zero model
        a = np.array(1)

    # Solve for the model error
    err = np.dot(x[q + 1:N].conj().T, X[q + 1:N, 0:p + 1])
    err = np.dot(err, a)

    # Solve for numerator coefficients
    if q > 0:
        # (This is the same as for Pad?)
        b = np.dot(X[0:q + 1, 0:p + 1], a)
    else:
        # all-pole model
        # b(0) is x(0), but a better solution is to match energy
        b = np.sqrt(err)

    return (b, a)


#### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### ####   Part 2  #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### ####
# From https://gist.github.com/endolith/4625838, Copyright (c) 2011 Christopher Felton,
# GNU Lesser General Public License


"""
From https://gist.github.com/endolith/4625838 :
"Combination of 
http://scipy-central.org/item/52/1/zplane-function
and
http://www.dsprelated.com/showcode/244.php
with my own modifications"
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.pyplot import axvline, axhline
from collections import defaultdict
from scipy.signal import tf2zpk


def zplane(b, a, ax=None, color='b', ignore_extremes=False, zeros_marker='o', poles_marker='x'):
    """Plot the complex z-plane given zeros and poles.
    """

    z, p, _ = tf2zpk(b, a)

    if ignore_extremes:
        z = z[np.logical_and(~(z.real == 0), ~(z.imag == 0))]
        p = p[np.logical_and(~(p.real == 0), ~(p.imag == 0))]

    # get a figure/plot
    if ax is None:
        # plt.figure()
        ax = plt.gca()

    # Add unit circle and zero axes
    unit_circle = patches.Circle((0, 0), radius=1, fill=False,
                                 color='black', ls=':', alpha=0.8)
    ax.add_patch(unit_circle)
    axvline(0, color='k', ls='-', linewidth=.5, alpha=.8)
    axhline(0, color='k', ls='-', linewidth=.5, alpha=.8)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1)

    ax.set_xticks([-1, -.5, 0, .5, 1])
    ax.set_xlabel('Real Part')
    ax.set_yticks([-1, -.5, 0, .5, 1])
    ax.set_ylabel('Imaginary Part')
    ax.grid(True, linestyle=':', linewidth=.5)

    # Plot the poles and set marker properties
    plot_args = {'markersize': 5, 'markeredgecolor': color, 'alpha': 0.5, 'color': 'none'}
    poles = ax.plot(p.real, p.imag, poles_marker, **plot_args)
    # Plot the zeros and set marker properties
    zeros = ax.plot(z.real, z.imag, zeros_marker, **plot_args)

    # Scale axes to fit
    r = 1.5 * np.amax(np.concatenate((abs(z), abs(p), [1])))
    ax.axis('scaled')
    ax.axis([-r, r, -r, r])
    #    ticks = [-1, -.5, .5, 1]
    #    plt.xticks(ticks)
    #    plt.yticks(ticks)

    """
    If there are multiple poles or zeros at the same point, put a 
    superscript next to them.
    TODO: can this be made to self-update when zoomed?
    """
    # Finding duplicates by same pixel coordinates (hacky for now):
    poles_xy = ax.transData.transform(np.vstack(poles[0].get_data()).T)
    zeros_xy = ax.transData.transform(np.vstack(zeros[0].get_data()).T)

    # dict keys should be ints for matching, but coords should be floats for
    # keeping location of text accurate while zooming

    # TODO make less hacky, reduce duplication of code
    d = defaultdict(int)
    coords = defaultdict(tuple)
    for xy in poles_xy:
        key = tuple(np.rint(xy).astype('int'))
        d[key] += 1
        coords[key] = xy
    for key, value in d.iteritems():
        if value > 1:
            x, y = ax.transData.inverted().transform(coords[key])
            ax.text(x, y,
                    r' ${}^{' + str(value) + '}$',
                    fontsize=13,
                    )

    d = defaultdict(int)
    coords = defaultdict(tuple)
    for xy in zeros_xy:
        key = tuple(np.rint(xy).astype('int'))
        d[key] += 1
        coords[key] = xy
    for key, value in d.iteritems():
        if value > 1:
            x, y = ax.transData.inverted().transform(coords[key])
            ax.text(x, y,
                    r' ${}^{' + str(value) + '}$',
                    fontsize=13,
                    )
