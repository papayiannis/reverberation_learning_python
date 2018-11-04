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

This is a collection of functions relevant to speech and audio processing.

This file was original distributed in the repository at:
{repo}

If you use this code in your work, then cite:
{placeholder}

"""
import numpy as np
from scipy.signal import lfilter

from utils_base import getfname, column_vector, row_vector

resample_eng = None


def my_resample(x, fs_old, fs_new, matlab_eng=None, verbose=False,
                close_after=False):
    """

    Uses the Matlab engine to resample audio files. Produces much better results than other
    alternatives.

    If you do not have matlab, then you can :
        bash conda install -c conda-forge resampy
    then replace this function with
        def my_resample(x, fs_old, fs_new, matlab_eng=None, verbose=False,
                close_after=False):
            from resampy import resample
            return resample(x, fs_old, fs_new)
    You can use any other library other than resampy of course.

    Args:
        x: Signal to resample
        fs_old: Old sampling rate
        fs_new: New sampling rate
        matlab_eng: Matlab engine object if pre-initialized
        verbose: Verbose reporting
        close_after: Close the matlab engine when done

    Returns: The resampled audio signal

    """
    from utils_base import flatten_array_list, repack_array_list
    global resample_eng
    was_int16 = False

    if fs_old == fs_new:
        return x
    if x.dtype == 'int16':
        was_int16 = True
        x = (x / np.iinfo('int16').max).astype('float')
        if verbose:
            print('Your input for resampling is int16, will go to float for '
                  'calculations then put it back to int16')

    if resample_eng is not None:
        eng = resample_eng
        print('Using static engine')
        import matlab
    elif matlab_eng is not None:
        print('Using provided Matlab engine')
        import matlab
        eng = matlab_eng
        my_resample.ext_eng = eng
    else:
        import matlab.engine
        print('Creating Matlab engine')
        eng = matlab.engine.start_matlab()
    resample_eng = eng

    x, shapes = flatten_array_list(x, orientation='landscape')

    print('Resampling ' + str(x.shape) + ' from ' + str(fs_old) + ' to ' + str(fs_new))

    (up, down) = (float(fs_new) / fs_old).as_integer_ratio()
    x_tmp = matlab.double(x.tolist())
    x_out = eng.resample(x_tmp, matlab.double([up]), matlab.double([down]), nargout=1)

    if close_after:
        eng.exit()
        resample_eng = None

    x_out = np.array(x_out)
    print('Got ' + str(x_out.shape) + ' output samples')
    x_out = repack_array_list(x_out, shapes=shapes, orientation='landscape')
    if len(x_out) == 1:
        x_out = x_out[0]

    if was_int16:
        if type(x_out) is list:
            for i in range(len(x_out)):
                x_out[i] = (x_out[i].astype('float64') * np.iinfo('int16').max).astype('int16')
        else:
            x_out = (x_out.astype('float64') * np.iinfo('int16').max).astype('int16')

    return x_out


def get_psd(x, fs, window_seconds, ):
    """

    Estimates the Power Spectral Density and Power spectrum of a given signal, using :
    P. Welch, "The use of the fast Fourier transform for the
           estimation of power spectra: A method based on time averaging
           over short, modified periodograms", IEEE Trans. Audio
           Electroacoust. vol. 15, pp. 70-73, 1967.

    Args:
        x: Signal
        fs: Sampling frequency
        window_seconds: Window length in seconds

    Returns: frequency_points, PSD, Power Spectrum

    """

    from scipy.signal import welch

    nperseg = int(np.ceil(fs * window_seconds))
    _, psd = welch(x, fs=fs, window='hamming', nperseg=nperseg, noverlap=None, nfft=None,
                   detrend='constant', return_onesided=True, scaling='density', axis=-1)
    f, pspec = welch(x, fs=fs, window='hamming', nperseg=nperseg, noverlap=None, nfft=None,
                     detrend='constant', return_onesided=True, scaling='spectrum', axis=-1)

    return f, psd, pspec


def distitpf(pf1, pf2, mode='0'):
    """

    Adaptation of the Itakura distance estimation method from Voicebox[1]

    Args:
        pf1: Power spectrum to compare
        pf2: Power spectrum to compare
        mode: Character string selecting the following options:
              'x'  Calculate the full distance matrix from every row of PF1 to every row of PF2
              'd'  Calculate only the distance between corresponding rows of PF1 and PF2
                   The default is 'd' if PF1 and PF2 have the same number of rows otherwise 'x'.

    Returns: Itakura distance

    [1] http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/doc/voicebox/distitpf.html

    """
    #
    pf1 = np.atleast_2d(pf1)
    pf2 = np.atleast_2d(pf2)
    (nf1, p2) = pf1.shape
    p1 = p2 - 1
    nf2 = pf2.shape[0]

    if mode == 'd' or (not mode == 'x' and nf1 == nf2):
        nx = min(nf1, nf2);
        r = pf1[0:nx, :] / pf2[0:nx, :]
        q = np.log(r);
        d = (np.log((np.sum(r[:, 1:p1], 1) + 0.5 * (r[:, 0] + r[:, p2 - 1])) / p1) -
             (np.sum(q[:, 1:p1], 1) + 0.5 * (q[:, 0] + q[:, p2 - 1])) / p1)[0]
    else:
        r = np.transpose(pf1[:, :, np.ones((nf2,), dtype=int)], axes=[0, 2, 1]) / \
            np.transpose(pf2[:, :, np.ones((nf1,), dtype=int)], axes=[2, 0, 1])
        q = np.log(r)
        d = np.log((np.sum(r[:, :, 1:p1], 2) + 0.5 * (r[:, :, 0] + r[:, :, p2 - 1])) / p1) - \
            (np.sum(q[:, :, 1:p1], 2) + 0.5 * (q[:, :, 0] + q[:, :, p2 - 1])) / p1
    return d


def write_wav(filename, fs, ss):
    """

    Writes audio samples as wav files to disk.

    Args:
        filename: Name to save file as
        fs: Sampling frequency
        ss: Audio samples as a numpy array

    Returns: Nothing

    """
    from scipy.io import wavfile
    wavfile.write(filename, fs, (ss.astype('float64') / abs(ss).max() * np.iinfo('int16').max
                                 ).astype('int16'))
    print('Wrote : ' + filename)


def ar_to_cepstrum(ar_coef, cep_order=None):
    """

    Converts Autoregressive (AR) coeficients to cepstral coefficients using the method discussed
    in [1]

    Args:
        ar_coef: AR coefficients
        cep_order: Order up to which to estimate cepstral coefficients

    Returns:
        The cepstral coefficients

    [1] K. Kalpakis, D. Gada, and V. Puttagunta, 'Distance measures for effective clustering of
    ARIMA time-series,' in ICDM, San Jose, California, USA, 2001, pp. 273-280.

    """
    back_to_flat = False
    if ar_coef.ndim == 1:
        back_to_flat = True
    ar_coef = np.atleast_2d(ar_coef)
    cep_order = ar_coef.shape[1] if cep_order is None else cep_order
    cep_coef = np.zeros((ar_coef.shape[0], cep_order), dtype=ar_coef.dtype)
    if not np.all(cep_coef[:, 0] == 1):
        raise AssertionError('Expected first AR coef ot be 1')
    for i in range(cep_coef.shape[0]):
        for k in range(cep_order):
            if k == 1:
                cep_coef[i, k] = -ar_coef[i, k]
            elif k <= cep_coef.shape[1]:
                cep_coef[i, k] = -ar_coef[i, k]
                for m in range(k):
                    cep_coef[i, k] = cep_coef[i, k] - (1 - (m + 1) / k) * ar_coef[i, m] * cep_coef[
                        i, k - m]
            else:
                cep_coef[i, k] = 0
                for m in range(k):
                    cep_coef[i, k] = cep_coef[i, k] - (1 - (m + 1) / k) * ar_coef[i, m] * cep_coef[
                        i, k - m]
    if back_to_flat:
        cep_coef.flatten()

    return cep_coef


def align_max_samples(yin, scan_range=None):
    """

    Aligns input signals so that the maximum energy samples are at the same index.

    Args:
        yin: List of input singals
        scan_range: Range of indicesto scan for in order to find the maximum energy sample

    Returns:
        List of aligned signals
        List of delay introduces in each signal during alignment

    """
    from utils_base import flatten_array_list

    yin = flatten_array_list(yin)[0]
    if yin.ndim == 1:
        raise ValueError('Expected 2D array as input')

    if yin.shape[0] == 1:
        yout = yin
        delays = np.array([0]).astype(float)
        return yout, delays

    if scan_range is None:
        scan_range = range(yin.shape[1])
    delays = abs(yin[:, scan_range]).argmax(axis=1)

    delays = abs(delays - delays.max())
    padding = delays.max()

    yout = np.concatenate(
        (np.zeros_like(yin),
         np.zeros((yin.shape[0], padding), dtype=yin.dtype)),
        axis=1).astype(yin.dtype)

    for idx, this_delay in enumerate(delays):
        tmp = np.zeros_like(yout[idx, :])
        tmp[this_delay:this_delay + yin.shape[1]] = yin[idx, :]
        yout[idx, :] = tmp
    if padding > 0:
        yout = yout[:, 0:-padding]

    return yout, delays


def scale_x_to_y(x, y):
    """

    Estimates a scaling for signal X, in order to Least Squares match the amplitude of samples in
    X and Y

    Args:
        x: X
        y: Y

    Returns:
        Scale (scalar)

    """
    scale = np.linalg.lstsq(np.atleast_2d(x).T, np.atleast_2d(y).T, rcond=None)[0][0][0]
    return scale


def fractional_alignment(yin, resolution=0.01, ls_scale=False, take_base_as=0):
    """

    Fractionally aligns signals (between (-1,+1) sample shifts) based on a least squares method
    of the mismatch of the samples, operating on a fixed resolution grid.

    Args:
        yin: List of input signals
        resolution: Resolution of search grid for fractional alignment
        ls_scale: Enable the scaling of the signals to a least squares mach of their samples in
        addition to the delaying
        take_base_as: Signal to take as the reference signal in the matching process. The default
        is to take the first signal in the list. This signal will remain unchanged and the rest
        will be matched to it.

    Returns:
        List of aligned signals
        List of delay introduces in each signal during alignment
        List of scale introduces in each signal

    """

    def get_mse(frac_kernels, idx, x, valid_range, y):
        x = np.convolve(x, frac_kernels[:, idx])
        x = x[valid_range]
        if ls_scale:
            scale = scale_x_to_y(x, y)
        else:
            scale = 1.
        mse = np.sum((y - x * scale) ** 2)
        return mse, scale

    if yin.ndim == 1:
        raise ValueError('Expected 2D array as input')

    resolution = float(resolution)
    npoints = int(np.ceil(1. / resolution))
    yin = np.array(yin)
    if yin.shape[0] == 1:
        yout = yin
        delays = np.array([0]).astype(float)
        return yout, delays

    context = 4
    valid_range = np.arange(context, context + yin.shape[1], 1).astype(int)
    _, frac_kernels = gm_frac_delayed_copies(np.ones((npoints,)),
                                             np.arange(context, context + 1, resolution) - .5,
                                             2 * context + 1)
    yout = np.zeros_like(yin)
    yout[take_base_as, :] = yin[take_base_as, :]
    delays = np.zeros((yin.shape[0],))
    scales = np.zeros((yin.shape[0],))
    scales[take_base_as] = 1.
    all_delays = np.arange(0, 1, resolution) - .5
    for i in range(yin.shape[0]):
        if take_base_as == i:
            continue
        mse_scales = np.array(
            [get_mse(frac_kernels, idx, yin[i, :], valid_range, yin[take_base_as, :])
             for idx in range(0, npoints)])
        min_point = mse_scales[:, 0].argmin()
        delays[i] = all_delays[min_point]
        scales[i] = mse_scales[min_point, 1]
        thisyout = scales[i] * np.convolve(yin[i, :], frac_kernels[:, min_point])
        yout[i, :] = thisyout[valid_range]

    return yout, delays, scales


def gm_frac_delayed_copies(amplitudes, delays, tot_length, excitation_signal=np.array([]),
                           center_filter_peaks=True):
    """

    The function when called implements the model defined as:
    (1) h(n) = \sum_{i=1}^{D}\left[{\beta_i}h_e(n){\ast}\frac{\sin~\pi(n-k_i)}{\pi(n-k_i)}\right]
    This is the model proposed in [1]

    This is effectively the summation on D copies of the signal excitation_signal.
    This copies are placed at sample locations 'delays' (which are not bound to integers) and
    their scaling is defined by  'amplitudes'. The length of the signal is defined as 'tot_length'.

    Args:
        amplitudes: Vector containing the scaling of each copy
        delays: Sample index of occurence each copy
        tot_length: Total length of the output vector
        excitation_signal: The signal to be copied at each location
        center_filter_peaks: Center the signal in 'excitation_signal', so that the samples of
        maximum energy occur at samples 'delays'

    Returns:
        y   : h(n) from the equation (1)
        Y   : A matrix containing as vectors the components of the summation in equation (1)

    [1] Papayiannis, C., Evers, C. and Naylor, P.A., 2017, August. Sparse parametric modeling of
    the early part of acoustic impulse responses. In Signal Processing Conference (EUSIPCO),
    2017 25th European (pp. 678-682). IEEE.

    """
    excitation_signal = np.atleast_2d(excitation_signal)
    if excitation_signal.shape[1] == 1:
        excitation_signal = excitation_signal.T

    if excitation_signal.size > 0:
        nfilters = excitation_signal.shape[0]
    else:
        nfilters = 0
    sincspan = 9

    amplitudes = np.atleast_2d(amplitudes)
    if amplitudes.shape[1] == 1:
        amplitudes = amplitudes.T
    delays = np.atleast_2d(delays)
    if delays.shape[1] == 1:
        delays = delays.T

    tot_components = amplitudes.size
    if delays.size != tot_components:
        raise NameError(getfname() + ':InputsMissmatch tot components are ' + str(
            tot_components) + ' and got delays ' + str(delays.size))
    if nfilters > 1:
        raise NameError(getfname() + ':InputsMissmatch')
    if tot_components < 1:
        yindiv = np.array([])
        ysignal = np.zeros((tot_length,))
        return [ysignal, yindiv]
    sample_indices = np.repeat(column_vector(np.arange(0, tot_length, 1, dtype=np.float64)),
                               tot_components, axis=1)
    sample_indices_offsetting = np.repeat(row_vector(delays), tot_length, axis=0)
    sample_indices -= sample_indices_offsetting
    yindiv = np.sinc(sample_indices) * np.repeat(row_vector(amplitudes), tot_length, axis=0)
    if ~np.isinf(sincspan):
        spanscale_numer = sincspan * np.sin(np.pi * sample_indices / float(sincspan))
        spanscale_denom = (np.pi * sample_indices)  # Lanczos kernel
        limiting_case_idx = np.where(spanscale_denom == 0)
        spanscale_denom[limiting_case_idx] = 1
        spanscale = spanscale_numer / spanscale_denom
        spanscale[limiting_case_idx] = 1
        yindiv *= spanscale
        yindiv[np.where(abs(sample_indices > sincspan))] = 0

    excitation_signal = excitation_signal.flatten()
    if nfilters > 0:
        if not center_filter_peaks:
            yindiv = lfilter(excitation_signal, [1], yindiv, axis=0)
        else:
            fcenter = np.argmax(abs(excitation_signal), axis=0)
            if fcenter == 0:
                yindiv = lfilter(excitation_signal, [1], yindiv, axis=0)
            else:
                futuresamples = fcenter
                tmpconc = np.concatenate((yindiv, np.zeros((futuresamples, yindiv.shape[1]))),
                                         axis=0)
                tmpconc = lfilter(excitation_signal, [1], tmpconc, axis=0)
                yindiv = tmpconc[futuresamples:, :]

    ysignal = np.sum(yindiv, axis=1)
    return [ysignal, yindiv]


def enframe(alike, flength, fincr, hamming_window=False):
    """
    Breaks the input into frames of length flength, with an increment of fincr samples per frame
    Args:
        alike: Input array like description of a vector
        flength: Frame Length in samples
        fincr:  Frame Increment in Samples
        hamming_window: Apply hamming window on frames

    Returns: The signal broken into frames

    """
    npa = np.array(alike)
    flength = int(flength)
    fincr = int(fincr)
    if npa.ndim > 1:
        raise NameError(getfname() + ':Non1DInput')
    if not (flength % fincr) == 0:
        raise NameError(getfname() + ':SizeOfFrameNotMultipleOfIncrement')
    nshifts = int(flength / float(fincr))
    totnframes = int(np.ceil(npa.size / float(fincr)))
    noframes_per_shift = int(np.ceil(totnframes / float(nshifts)))
    discardlastframes = (totnframes % nshifts) > 0
    xout = np.zeros((noframes_per_shift * nshifts, flength))
    tnsamples = int(noframes_per_shift * flength)
    fidxs = np.arange(0, noframes_per_shift, dtype=np.int) * nshifts
    npadding = (flength - npa.size % flength) % flength + flength
    npa = np.append(npa, np.zeros(npadding))
    for i in range(nshifts):
        fidxs += i
        xout[fidxs, :] = np.array(npa[i * fincr:i * fincr + tnsamples]).reshape(fidxs.size, flength)
        if hamming_window:
            xout[fidxs, :] = xout[fidxs, :] * np.hamming(flength)
    if discardlastframes:
        xout = xout[0:-1, :]
    return xout


def overlapadd(input_frames, window_samples=None, inc=None,
               previous_partial_output=None):
    """
    Performs overlap-add using the frames in input_samples. This is a Python implementation of
    the MATLAB code available in the VOICEBOX toolbox.
    (http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/doc/voicebox/overlapadd.html)

    Args:
        input_frames: The array of input frames of size M X window_samples
        window_samples: The window to be used for the frames
        inc: The increment in samples between frames
        previous_partial_output: Provide the partial output returned from a previous call to
        this function

    Returns:
        The overlap-add result and the partial output at the end

    """

    if window_samples is None:
        window_samples = np.ones((input_frames.shape[1],))
    elif window_samples.size != input_frames.shape[1]:
        raise NameError(getfname() + ":WindowSizeDoesNotMatchFrameSize")
    if inc is None:
        inc = input_frames.shape[1]
    elif inc > input_frames.shape[1]:
        raise NameError(getfname() + ":SampleIncrementTooLarge")
    nr = input_frames.shape[0]
    nf = input_frames.shape[1]

    nb = int(np.ceil(nf / float(inc)))
    no = int(nf + (nr - 1) * inc)
    overlapped_output_shape = (no, nb)

    z = np.zeros((int(no * nb),))
    # input_frames = np.asfortranarray(input_frames)

    zidx = (
            np.repeat(row_vector(np.arange(0, nf, dtype=np.int)), nr, axis=0) +
            np.repeat(column_vector(np.arange(0, nr, dtype=np.int) * inc +
                                    (np.arange(0, nr, dtype=np.int) % nb) * no), nf, axis=1))
    # input_frames_windowed = input_frames * np.repeat(row_vector(window_samples), n_frames, axis=0)
    input_frames *= np.repeat(row_vector(window_samples), nr, axis=0)
    z[zidx.flatten(order='F').astype(np.int32)] = input_frames.flatten(order='F')
    z = z.reshape(overlapped_output_shape, order='F')
    if z.ndim > 1:
        z = np.sum(z, axis=1)
    if previous_partial_output is not None:
        if previous_partial_output.ndim > 1:
            raise NameError(getfname() + "PrevPartialOutDimError")
        else:
            z[0:previous_partial_output.size] += previous_partial_output
    out_samples = int(inc * nr)
    if no < out_samples:
        z[out_samples] = 0
        current_partial_output = np.array([])
    else:
        current_partial_output = z[out_samples:]
        z = z[0:out_samples]

    return z, current_partial_output


def playaudio(samples, sampling_freq, wait_to_finish=True):
    """

    Plays audio using pygame

    Args:
        samples: Audio samples
        sampling_freq: Sampling frequency
        wait_to_finish: Wait for the player to finish before returning

    Returns:
        The samples passed to the player
    """
    try:
        import pygame as pyg
    except ImportError:
        raise
    pyg.mixer.init(frequency=sampling_freq, channels=2)

    samples = np.array(samples, dtype='int16')
    if samples.ndim == 1:
        samples = np.concatenate((column_vector(samples), column_vector(samples)), axis=1)
    elif samples.ndim > 2:
        samples = samples[:, 0:2]

    pyg.mixer.Sound(array=samples).play()

    if wait_to_finish:
        nsecs = int(np.ceil(samples.shape[0] / float(sampling_freq)))
        print(getfname() + " : Playing " + repr(nsecs) + " seconds of audio")
        while pyg.mixer.get_busy():
            pass

    return samples


def get_array_energy(alike):
    """
    Getthe total energy of the elements in an array

    Args:
        alike: The input array

    Returns:
        The total energy

    """
    return np.sum(np.array(alike, dtype='float128') ** 2)


def plotnorm(x=None, y=None, title=None, interactive=False, clf=False, savelocation=None,
             no_rescaling=False, **mplotargs):
    """

    A useful and flexible plotting tool for signal processing.
    It allows you to plot a number of signals on the same normalised scale. It can plot signals
    in the time domain when provided with a sampling frequency.

    Args:
        x: The x axis points or the sampling frequency as a scalar
        y: The list of vectros or the array to plot. the vectors can have a different number of
        elements each
        title: The string to use as the plot title
        interactive: Wait for the user to close the plot before continuing
        clf: Clear the plot before plotting
        savelocation: Save hte plot as this file
        no_rescaling: Do not normalize the scale of the signals
        **mplotargs: Arguments to be passed to matplotlib.pyplot.plot

    Returns:
        The plot

    """

    import matplotlib.pyplot as mplot

    hasfs = False
    if (x is None) & (y is not None):
        x = np.arange(y.size)
    elif (not isinstance(x, np.ndarray)) & (y is not None):
        sampling_freq = float(x)
        x = np.arange(y.size) / sampling_freq
        hasfs = True
    elif x.size != y.size:
        raise NameError(getfname() + 'XYSizeMismatch')
    elif x.size == 0:
        return
    if not no_rescaling:
        y = y / abs(y).max()

    if clf:
        mplot.clf()
    res = mplot.plot(x, y, linewidth=0.5, **mplotargs)
    gen_title = 'Normalised Amplitude'
    if hasfs:
        gen_title += ' at Fs=' + repr(sampling_freq) + 'Hz'
        mplot.xlabel('Time (s)')
    else:
        mplot.xlabel('Sample')
    if title is None:
        mplot.title(gen_title)
    elif not title == '':
        mplot.title(title)
    mplot.ylabel('Normalised Amplitude')
    mplot.grid(True)
    if savelocation is not None:
        mplot.savefig(savelocation)
        print('Saved: ' + savelocation)
    if interactive:
        mplot.show()

    return res