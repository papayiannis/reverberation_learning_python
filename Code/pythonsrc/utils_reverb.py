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

This is a collection of functions relevant to room acoustics and reverberation

This file was original distributed in the repository at:
{repo}

If you use this code in your work, then cite:
{placeholder}

"""
import numpy as np

from utils_base import matmax, getfname, column_vector
from utils_spaudio import get_array_energy


def npm(h, hhat):
    """

    Estimates the Normalized Projection Misalignment (NPM) from [1]

    Args:
        h: AIR to compare
        hhat: AIR to compare

    Returns: The NPM

    [1] Morgan, D.R., Benesty, J. and Sondhi, M.M., 1998. On the evaluation of estimated impulse
    responses. IEEE Signal processing letters, 5(7), pp.174-176.

    """

    hhat = np.array(hhat)
    h = np.array(h)

    if hhat.ndim > 1 or h.ndim > 1:
        raise AssertionError('Expecting single channel responses')

    h = h.flatten()
    hhat = hhat.flatten()

    epsilon = np.sum(h * hhat) / (
        np.sqrt(np.sum(hhat * hhat) * np.sum(h * h))
    )
    npm_val = 1 - epsilon ** 2

    return npm_val


def scale_with_absorption_coefs(x, fs, freqs, abs_coef, framesize=0.020, times=(1,)):
    """

    Filters a given input, based on the sound energy abosrption coefficients provided,
    in a frame-based mode

    Args:
        x: Audio signal
        fs: Sampling frequency
        freqs: Frequency points
        abs_coef: Sound energy absorption coefficient at given frquency point
        framesize: Framesize in samples
        times: Number of times to pass the signal through the filering

    Returns:
        The filtered signal

    """
    from utils_spaudio import enframe, overlapadd

    times = np.array(times).round().astype(int)
    abs_coef = np.atleast_2d(abs_coef)

    if np.sum(times) == 0:
        return x

    framelength = int(np.ceil(framesize * fs))
    window = np.hanning(framelength)
    original_length = x.size
    if x.size < framelength:
        missing = framelength - x.size
        x = np.concatenate((x.flatten(), np.zeros((missing,)).astype(x.dtype)))
        x = np.atleast_2d(x)
    else:
        x = enframe(x, framelength, int(np.ceil(framelength / 2)))
        if x.shape[0] > 1:
            for i in range(x.shape[0]):
                x[i, :] = x[i, :] * window
    xft = np.fft.rfft(x, axis=1)
    dreqs = np.arange(0, xft.shape[1], 1) / float(xft.shape[1]) * fs / 2.

    if not times.size == abs_coef.shape[0]:
        raise AssertionError('times for application should match filters')

    def get_scale(f, freqs_local, abs_coef_local):
        previous = np.where(f > freqs_local)[-1]
        if previous.size == 0:
            previous = 0
        else:
            previous = previous[-1]
        scale = 1
        for i, this_time in enumerate(times):
            if this_time > 0:
                if previous == freqs_local.size - 1:
                    this_absorption = abs_coef_local[i, -1]
                else:
                    this_absorption = abs_coef_local[i, previous] * (f - freqs_local[previous]) / (
                            freqs_local[previous + 1] - freqs_local[previous]) + \
                                      abs_coef_local[i, previous + 1] * (
                                              -f + freqs_local[previous + 1]) / (
                                              freqs_local[previous + 1] - freqs_local[previous])

                scale *= np.sqrt(1 - this_absorption) ** this_time
        return scale

    freqs = np.array(freqs)
    scale = []
    for i in range(dreqs.size):
        scale.append(get_scale(dreqs[i], freqs, abs_coef))
        xft[:, i] = xft[:, i] * scale[-1]

    y = np.fft.irfft(xft, axis=1)
    if y.shape[0] > 1:
        y = overlapadd(y, inc=int(np.ceil(framelength / 2)))[0]
    else:
        y = y.flatten()

    y = y[0:original_length]
    return y


def get_drr_linscale(air_fir_taps, sampling_freq, direct_window_length_secs=0.0008,
                     ignore_reflections_up_to=None):
    """
    Estimates the Direct to Reverberant ration given an Acoustic Impulse Response (AIR)

    Args:
        air_fir_taps: The taps of the AIR
        sampling_freq: The sampling frequency
        direct_window_length_secs: The length of the window that is estimated ot contain the
        direct sound
        ignore_reflections_up_to: The reflections up to this point (in seconds) are ignored and
        not considered to be part of either the early or the late part.

    Returns: The DRR in linear scale

    """
    air_fir_taps = np.array(air_fir_taps)
    if air_fir_taps.ndim > 1:
        raise NameError(getfname() + "AIR_Not1D")
    nsidesamples = int(np.ceil(direct_window_length_secs / 2. * sampling_freq))
    dpathcenter = abs(air_fir_taps).argmax()
    dstartsample = max(dpathcenter, dpathcenter - nsidesamples)
    dendsample = min(dpathcenter + nsidesamples, air_fir_taps.size)
    if ignore_reflections_up_to is not None:
        ignore_until_sample = int(np.ceil(ignore_reflections_up_to * sampling_freq))
        if ignore_until_sample > dendsample:
            air_fir_taps[dendsample:ignore_until_sample] = 0
        else:
            print('You gave an ignore range for DRR calculation but it was invalid')
    return get_array_energy(air_fir_taps[dstartsample:dendsample]) \
           / get_array_energy(air_fir_taps[dendsample:])


def get_t60_decaymodel(air_fir_taps, sampling_freq):
    """ Estimates the Reverberation Time, given an Acoustic Impulse Response.

    The mode of operation is defined by the reference below.
    This is a wrapper of a python translation of the code made available by
    the authors of: Karjalainen, Antsalo, and Peltonen,
    Estimation of Modal Decay Parameters from Noisy Response Measurements.
    at : http://www.acoustics.hut.fi/software/decay

    Examples:
        When involving an AIR that is sampled at 48 kHz for example.

        reverb_time = get_t60_decaymodel(air, 48000)

    Args:
        air_fir_taps    :   Acoustic Impulse Response to process
        sampling_freq   :   The sampling frequency at which air was recorded at

    Returns:
        An estimate of the reverbration time in seconds

    """
    from scipy.optimize import curve_fit

    def decay_model(x_points, param0, param1, param2):
        """ The function used bo the non-linear least squares fitting method to
        estimate the decay parameters"""
        expf = 0.4
        y1_dm = np.multiply(param0, np.exp(np.multiply(param1, x_points)))
        y2_dm = param2
        fit_res = np.multiply(weights, np.power(
            np.add(np.power(y1_dm, 2), np.power(y2_dm, 2)), 0.5 * expf))
        return fit_res

    # air is a 1D list
    # Set up things. Move to dB domain and scale
    leny = len(air_fir_taps)
    air = np.multiply(20, np.log10(abs(air_fir_taps) + np.finfo(float).eps))
    _, ymaxi = matmax(air)
    air = air - air[ymaxi]
    weights = [1] * leny
    weights[0:max(1, ymaxi)] = [0] * max(1, ymaxi)
    time_points = np.linspace(0, leny / float(sampling_freq), leny)
    # Lin fit
    leny2 = leny // 2
    leny10 = leny // 10
    ydata = np.power(np.power(10, air / 20.), 0.4)
    start_of_range = np.nonzero(weights)[0][0]
    meanval1 = np.mean(ydata[start_of_range:leny10 + start_of_range + 1])
    meanvaln = np.mean(ydata[leny - leny10:leny])
    tmat = np.concatenate((np.ones((leny2, 1)),
                           column_vector(time_points[start_of_range:leny2 + start_of_range])),
                          axis=1)
    tau0 = np.linalg.lstsq(tmat, air[start_of_range:leny2 + start_of_range], rcond=None)
    tau0 = tau0[0][1] / 8.7
    ydata = np.multiply(weights, np.array(ydata))
    fit_bounds = ([0, -2000, 0], [200., -0.1, 200.])
    if tau0 > -0.1:  # to satisfy the bounds
        tau0 = -0.1

    sol_final = curve_fit(decay_model, time_points, ydata, p0=(meanval1, tau0, meanvaln),
                          bounds=fit_bounds)

    reverb_time = np.log(1 / 1000.) / float(sol_final[0][1])
    if reverb_time <= 0:
        raise NameError(getfname() + ':NegativeRT')

    return reverb_time


def get_edc(air_fir_taps):
    """
    Calculates and returns the Energy Decay Curve (EDC) (Schroeder Integral) of the supplied
    AIR.

    Args:
        air_fir_taps: Taps of FIR filter representation of AIR

    Returns:
        The EDC of the supplied AIR as a numpy.array

    Examples:
        edcurve = get_edc(air_fir_taps)

    """
    edcurve = np.flip(np.cumsum(np.array(air_fir_taps[::-1]) ** 2).flatten(), 0)
    return edcurve


def air_up_to_db(air_fir_taps, up_to_db):
    """

    Args:
        air_fir_taps: Input AIR
        up_to_db: The energy cutoff point in dB

    Returns: The AIR truncated from tap 0 to tap N, the point at which the energy remaining is
    less than 'up_to_db' of the energy of the entire AIR
    than up_to_db of the

    """
    up_to = 10 ** (up_to_db / 10.)
    air_edc = get_edc(air_fir_taps)
    cutoff_sample = np.where(air_edc < up_to)[0]
    if cutoff_sample.size > 0:
        cutoff_sample = min(air_fir_taps.size, np.where(air_edc < up_to)[0][0])
    else:
        cutoff_sample = air_fir_taps.size

    return air_fir_taps[0:cutoff_sample]
