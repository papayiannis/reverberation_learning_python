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

This file contains a set of routines which make feature extraction and data handling easy for
deep learning tasks around reverberation. It offeres some dedicated routines for the ACE
challenge database (http://www.ee.ic.ac.uk/naylor/ACEweb/index.html)

This file was original distributed in the repository at:
{repo}

If you use this code in your work, then cite:
{placeholder}

"""


def print_split_report(y, split_idxs=(), split_names=()):
    """
    
    Prints the distribution of the labels of data across splits. If no splits exist,
    then a report on the global distribution is given. Splits are indicated with the set of
    indices in the provided tuple. Split names can be provided fro clarity.
    
    Args:
        y: The classes labels as a vector
        split_idxs: A list of lists of indices, indicating members of the vector which belong to
        each split.
        split_names: The name of each split as a list. Should have the same length as the
        index list.

    Returns:
        Nothing

    """
    import numpy as np
    from tabulate import tabulate
    if len(split_idxs) == 0:
        split_idxs = (np.arange(0, y.shape[0]),)
    if not len(split_idxs) == len(split_names):
        if len(split_names) > 0:
            raise AssertionError('Invalid Inputs')
        else:
            split_names = []
            for i in range(len(split_idxs)):
                split_names.append('Set ' + str(i))

    distributions = np.zeros((y.shape[1], len(split_idxs)), dtype=int)
    for i in range(len(split_idxs)):
        idxs = split_idxs[i]
        for j in range(y.shape[1]):
            distributions[j, i] = np.sum(y[idxs, j])
    print('Data Distributions:')
    print(tabulate(distributions, headers=split_names, showindex=True))


def data_post_proc(x, fs, start_at_max, framesize, get_pow_spec, max_len, wavform_logpow):
    """

    Processes audio files, before they are passed to DNNs for training.

    Args:
        x: Audio signal samples a numpy array [N_signals X N_samples]
        fs: Sampling frequency
        start_at_max: Shift each signal so that they start at the maximum sample
        framesize: Set a framesize (in samples) for the signals to be enframed at. Turns the 
        array of signals into a 3D array.
        get_pow_spec: Transform the signals into the log-power spectrum of the signals
        max_len: Maximum singal length in seconds (truncate or pad to this)
        wavform_logpow: Get the signals in the log-power time domain

    Returns:
        The processed signals

    """
    import numpy as np

    match_training_spectrum = False
    if get_pow_spec and wavform_logpow:
        raise AssertionError('Unexpected scenario')

    print('For this feature extraction i will:')
    if start_at_max:
        print('Make all inputs start at the maximum energy sample')
    if framesize is not None:
        print('Enframe the inputs')
    if get_pow_spec:
        print('Get the logpow spectrum')
    if max_len is not None:
        print('Truncate the maximum length of the input')
    if wavform_logpow:
        print('Convert the time domain samples to the logpow')
    print('And return the results for ' + str(x.shape[0]) + ' inputs')

    def make_start_at_max(x, fs=None, max_air_len=None, just_crop=False, leeway=0.0):
        max_samples = None
        if max_air_len is not None or leeway is not None:
            if fs is None:
                raise AssertionError('Max length and leeway should be given with fs')
        if max_air_len is not None:
            max_samples = int(np.ceil(max_air_len * fs))

        if leeway is None:
            leeway_samples = 0
        else:
            leeway_samples = int(np.ceil(leeway * fs))
        min_max = np.inf
        max_max = 0
        if not just_crop:
            for i in range(x.shape[0]):
                maxp = max(0, abs(x[i, :]).argmax() - leeway_samples)
                min_max = min(min_max, maxp)
                max_max = max(max_max, maxp)
                if maxp > 0:
                    x[i, 0:-maxp] = np.array(x[i, maxp:])
                    x[i, -maxp:] = 0
            x = x[:, np.any(x > 0, axis=0)]
            print('Max shift : ' + str(max_max) + ' min max : ' + str(min_max))
        if max_air_len is not None:
            x = x[:, 0:max_samples]
            if x.shape[1] < max_samples:
                padding = np.zeros((x.shape[0], max_samples - x.shape[1]))
                x = np.concatenate((x, padding), axis=1)
        return x

    def to_enframed(x, framesize=None, window=True):
        from utils_spaudio import enframe

        return np.stack(
            [enframe(x[i, :], framesize, int(np.ceil(framesize / 2)), hamming_window=window)
             for i in range(x.shape[0])])

    def to_pow_spec(x, framesize=None):
        from numpy.fft import rfft
        print('Getting FFTs')
        oshape = x.shape
        if framesize is not None:
            if x.ndim < 3:
                x = to_enframed(x, framesize=framesize)
                oshape = x.shape
            x = np.concatenate([x[i, :, :] for i in range(x.shape[0])], axis=0)
        x = (abs(rfft(x, axis=1)))
        if match_training_spectrum:
            av_spec = np.genfromtxt('../results_dir/surface_models/average_response_abs_fft.csv',
                                    delimiter=',')
            x *= np.atleast_2d(av_spec)
        x[x == 0] = np.min(x[x > 0]) * 0.01
        x = np.log(x)
        if framesize is not None:
            x = x.reshape(tuple(list(oshape)[0:2] + [-1]))
        return x

    if start_at_max or (max_len is not None):
        from time import sleep
        retry = True
        give_up_at = 1
        retries = 0
        while retry:
            try:
                x = make_start_at_max(x, fs, max_len, just_crop=(not start_at_max), leeway=0.0)
                retry = False
            except ValueError as ME:
                print('Failed to fex because ' + ME.message)
                retries += 1
                sleep(.5)
                if retries + 1 > give_up_at:
                    print('Trying to make it writable')
                    x.setflags(write=1)
                if retries > give_up_at:
                    raise ME

    if framesize is not None:
        x = to_enframed(x, framesize=framesize, window=get_pow_spec)

    if get_pow_spec:
        x = to_pow_spec(x, framesize=framesize)

    if wavform_logpow:
        x = x ** 2
        xmax = np.max(x)
        x[x < (xmax / 200)] = xmax / 200
        x = np.nan_to_num(np.log(x))

    return x


def read_airs_from_wavs(wav_files, framesize=None, get_pow_spec=True,
                        max_air_len=None, fs=None, forced_fs=None,
                        keep_ids=None, cacheloc='/tmp/',
                        start_at_max=True, read_cached_latest=False,
                        wavform_logpow=False,
                        write_cached_latest=True, max_speech_read=None,
                        max_air_read=None, utt_per_env=1,
                        parse_as_dirctories=True,
                        speech_files=None, save_speech_associations=True,
                        save_speech_examples=10, drop_speech=False, as_hdf5_ds=True,
                        choose_channel=None, no_fex=False, scratchpad='/tmp/',
                        copy_associations_to=None, given_associations=None):
    """

    Given a set of AIR files and additional inforamtion, data for the training of DNNs for
    environment classification are prepared.

    Args:
        wav_files: Location of AIR wav files
        framesize: The framesize to ues
        get_pow_spec: Convert audio to log-power spectrum domain
        max_air_len: The maximum length of the signals (truncate to or pad to)
        fs: The sampling frequency of the wav fiels to expect
        forced_fs: The sampling frequency to convert the data to
        keep_ids: None (not used)
        cacheloc: Location to use for cache reading and saving
        start_at_max: Modify the signals so that the maximum energy sample is at the begiing. (
        can be used to align AIRs)
        read_cached_latest: Read the data from the last saved cache (if nay)
        wavform_logpow: Get the signals in the log-power time domain
        write_cached_latest: Write the collected data in a cache for fast reuse
        max_speech_read: Maximum length of speech signal to read
        max_air_read: maximum aIR length to read up to
        utt_per_env: Number of utternaces to convolve with each AIR
        parse_as_dirctories: Parse the inputs as directiries and not as individual fiels
        speech_files: Speec files of locations
        save_speech_associations: Save the speech associations with the corresponding AIRs
        save_speech_examples: Enable the saving of examples of the reverberant speech created
        drop_speech: Do not include the speech samples in the saving of the cache or in the RAM.
        Keep only the training data arrays
        as_hdf5_ds: Keep the data as HDF5 datasets on disk. (Reduces RAM usage a lot)
        choose_channel: Channels to use for each AIR
        no_fex: Skip the data processign phase and just return the raw singals
        scratchpad: Location to use for temporary saving
        copy_associations_to: Save a copy of the speech-aIR associations here
        given_associations: Provided associatiosn between speech files and AIRs. This can be used
        in the case where you want to use specific speech samples for specific AIRs

    Returns:
        (X, None), Sample_names, None,
        (AIRs, Speech, Reverberant_speech),
        (Group_name, Groups), Number_of_utternaces_convolved_with_each_AIR

    """
    try:
        from os.path import isfile, basename
    except ImportError:
        raise
    from scipy.signal import fftconvolve
    import numpy as np
    from h5py import File
    from scipy.io import wavfile
    from utils_spaudio import my_resample, write_wav
    from utils_base import find_all_ft, run_command
    from random import sample
    import pandas as pd
    from random import randint
    from time import time

    run_command('mkdir -p ' + cacheloc)
    latest_file = cacheloc + '/training_test_data_wav.h5'
    timestamp = str(time())
    filename_associations = scratchpad + '/air_speech_associations_' + timestamp + '.csv'
    base_examples_dir = scratchpad + '/feature_extraction_examples/'
    if keep_ids is not None:
        raise AssertionError('No ids exist in this context')
    if speech_files is None:
        utt_per_env = 1
        if save_speech_associations:
            print('There is no speech to save in associations, setting to false')
            save_speech_associations = False
        if save_speech_examples:
            print('There is no speech to save audio for, setting to 0 examples')
            save_speech_examples = 0

    try:
        hf = None
        if isfile(latest_file) and read_cached_latest:
            print('Reading :' + latest_file)
            hf = File(latest_file, 'r')
            if as_hdf5_ds:
                x = hf['x']
                ids = hf['ids']
                airs = hf['airs']
                utt_per_env = np.array(hf['utts'])
                rev_speech = hf['rev_names']
                clean_speech = hf['clean_speech']
                print('Done creating handles to : ' + latest_file)
            else:
                utt_per_env = np.array(hf['utts'])
                x = np.array(hf.get('x'))
                ids = np.array(hf.get('ids'))
                airs = np.array(hf.get('airs'))
                rev_speech = np.array(hf.get('rev_names'))
                clean_speech = np.array(hf.get('clean_speech'))
                print('Done reading : ' + latest_file)
            if given_associations is not None:
                print('! I read the cache so the given associations were not used')
            if copy_associations_to is not None:
                print('! I read the cache so the associations could not be saved at ' +
                      copy_associations_to)
            return (x, None), ids, None, (airs, clean_speech, rev_speech), utt_per_env
    except (ValueError, KeyError) as ME:
        print('Tried to read ' + latest_file + ' but failed with ' + ME.message)
        if hf is not None:
            hf.close()

    if given_associations is not None:
        print('You gave me speech associations, Speech: ' + str(len(given_associations['speech'])) +
              ' entries and Offsets: ' + str(len(given_associations['speech'])) + ' entries')

    ids = None
    x = None
    x_speech = None
    x_rev_speech = None

    if forced_fs is None:
        forced_fs = fs
    resample_op = lambda x: x
    if not forced_fs == fs:
        resample_op = lambda x: np.array(
            my_resample(np.array(x.T, dtype=float), fs, forced_fs)
        ).T

    if max_air_read is not None:
        if fs is None:
            raise AssertionError('Cannot work with max_air_read without fs')
        max_air_read_samples = int(np.ceil(fs * max_air_read))
    if max_speech_read is not None:
        if fs is None:
            raise AssertionError('Cannot work with max_speech_read without fs')
        max_speech_read_samples = int(np.ceil(fs * max_speech_read))
    else:
        max_speech_read_samples = None

    if parse_as_dirctories:
        if not type(wav_files) is list:
            wav_files = [wav_files]
        wav_files = find_all_ft(wav_files, ft='.wav', find_iname=True)
    if speech_files is not None:
        if not type(speech_files) is list:
            speech_files = [speech_files]
        speech_files = find_all_ft(speech_files, ft='.wav', find_iname=True)

    if save_speech_examples:
        run_command('rm -r ' + base_examples_dir)
        run_command('mkdir -p ' + base_examples_dir)

    associations = []
    save_counter = 0
    all_names = [basename(i).replace('.wav', '') + '_' + str(j) for i in wav_files for j in
                 range(utt_per_env)]
    if type(choose_channel) is list:
        choose_channel = [i for i in choose_channel for _ in range(utt_per_env)]
    wav_files = [i for i in wav_files for _ in range(utt_per_env)]
    offsets = []
    for i, this_wav_file in enumerate(wav_files):
        if False and speech_files is not None:
            print "Reading: " + this_wav_file + " @ " + str(i + 1) + " of " + str(len(wav_files)),
        names = [all_names[i]]
        this_fs, airs = wavfile.read(this_wav_file)
        airs = airs.astype(float)
        if airs.ndim > 1:
            if choose_channel is not None:
                if type(choose_channel) is list:
                    airs = airs[:, choose_channel[i]]
                    names[0] += '_ch' + str(choose_channel[i])
                else:
                    airs = airs[:, choose_channel]
                    names[0] += '_ch' + str(choose_channel)
            else:
                names = [names[0] + '_' + str(ch_id) for ch_id in range(airs.shape[1])]
            airs = airs.T
        airs = np.atleast_2d(airs)
        airs /= np.repeat(np.atleast_2d(abs(airs).max()).T, airs.shape[1], 1).astype(float)
        if airs.shape[0] > 1 and given_associations is not None:
            raise AssertionError('Cannot work out given associations for multichannel airs')
        this_speech_all = []
        this_rev_speech_all = []
        if speech_files is not None:
            for ch_id in range(airs.shape[0]):
                if given_associations is None:
                    chosen_file = sample(range(len(speech_files)), 1)[0]
                    this_speech_file = speech_files[chosen_file]
                else:
                    chosen_file = given_associations['speech'][i]
                    this_speech_file = chosen_file
                associations.append(chosen_file)
                this_speech_fs, this_speech = wavfile.read(this_speech_file)
                if this_speech.ndim > 1:
                    raise AssertionError('Can\'t deal with multichannel speech in this context')
                if not this_speech_fs == this_fs:
                    this_speech = my_resample(this_speech, this_speech_fs, this_fs)
                max_offset_for_check = None
                if max_speech_read_samples is not None:
                    max_offset_for_check = this_speech.size - max_speech_read_samples
                    offset = randint(0, this_speech.size - max_speech_read_samples)
                    this_speech = this_speech[offset:offset + max_speech_read_samples]
                else:
                    offset = 0
                if given_associations is not None:
                    offset = given_associations['offsets'][i]
                    if max_speech_read_samples is not None:
                        if offset >= max_offset_for_check:
                            raise AssertionError(
                                'Invalid offset from given associations, got ' + str(
                                    offset) + ' expected max is ' + str(
                                    this_speech.size - max_speech_read_samples))

                conv_air = np.array(airs[ch_id, :])
                conv_air = conv_air[
                           np.where(~(conv_air == 0))[-1][0]:np.where(~(conv_air == 0))[-1][-1]]

                # Making convolution
                this_rev_speech = fftconvolve(this_speech, conv_air, 'same')
                #

                dp_arival = np.argmax(abs(conv_air))
                this_rev_speech = this_rev_speech[dp_arival:]
                if dp_arival > 0:
                    this_rev_speech = np.concatenate(
                        (this_rev_speech, np.zeros(dp_arival, dtype=this_rev_speech.dtype)))

                this_speech = np.atleast_2d(this_speech)
                this_rev_speech = np.atleast_2d(this_rev_speech)
                this_speech_all.append(this_speech)
                this_rev_speech_all.append(this_rev_speech)

                offsets.append(offset)
                if save_speech_examples >= save_counter:
                    save_names = [
                        basename(this_wav_file).replace('.wav', '') + '_air_' + str(
                            offset) + '.wav',
                        basename(this_wav_file).replace('.wav', '') + '_clean_speech_' + str(
                            offset) + '.wav',
                        basename(this_wav_file).replace('.wav', '') + '_rev_speech_' + str(
                            offset) + '.wav'
                    ]
                    for examples in range(len(save_names)):
                        save_names[examples] = base_examples_dir + save_names[examples]
                    write_wav(save_names[0], this_fs, airs[ch_id, :])
                    write_wav(save_names[1], this_fs, this_speech.flatten())
                    write_wav(save_names[2], this_fs, this_rev_speech.flatten())
                    save_counter += 1
            this_speech = np.concatenate(this_speech_all, axis=0)
            this_rev_speech = np.concatenate(this_rev_speech_all, axis=0)

        if not this_fs == fs:
            raise AssertionError('Your sampling rates are not consistent')
        if i > 0:
            ids = np.concatenate((ids, names))
        else:
            ids = names

        if max_air_read is not None:
            airs = airs[:, 0:max_air_read_samples]
        if False and speech_files is not None:
            print("Got " + str(airs.shape))
        airs = resample_op(airs)
        if airs.ndim < 2:
            airs = np.atleast_2d(airs)
        # print('Done resampling')
        if i > 0:
            if x.shape[1] < airs.shape[1]:
                npads = -x.shape[1] + airs.shape[1]
                x = np.concatenate((x, np.zeros((x.shape[0], npads)).astype(x.dtype)), axis=1)
                x = np.concatenate((x, airs), axis=0)
            else:
                if x.shape[1] > airs.shape[1]:
                    npads = x.shape[1] - airs.shape[1]
                    airs = np.concatenate((airs, np.zeros((airs.shape[0], npads)).astype(
                        airs.dtype)), axis=1)
                x.resize((x.shape[0] + airs.shape[0], x.shape[1]), refcheck=False)
                x[-airs.shape[0]:, :] = np.array(airs)

            if speech_files is not None:
                if x_speech.shape[1] < this_speech.shape[1]:
                    npads = -x_speech.shape[1] + this_speech.shape[1]
                    x_speech = np.concatenate(
                        (x_speech, np.zeros((x_speech.shape[0], npads)).astype(x_speech.dtype)),
                        axis=1)
                    x_speech = np.concatenate((x_speech, this_speech), axis=0)
                else:
                    if x_speech.shape[1] > this_speech.shape[1]:
                        npads = x_speech.shape[1] - this_speech.shape[1]
                        this_speech = np.concatenate(
                            (this_speech, np.zeros((this_speech.shape[0],
                                                    npads)).astype(
                                this_speech.dtype)), axis=1)
                    x_speech.resize(
                        (x_speech.shape[0] + this_speech.shape[0], x_speech.shape[1]),
                        refcheck=False)
                    x_speech[-this_speech.shape[0]:, :] = this_speech

                if x_rev_speech.shape[1] < this_rev_speech.shape[1]:
                    npads = -x_rev_speech.shape[1] + this_rev_speech.shape[1]
                    x_rev_speech = np.concatenate(
                        (x_rev_speech, np.zeros((x_rev_speech.shape[0], npads)
                                                ).astype(x_rev_speech.dtype)),
                        axis=1)
                    x_rev_speech = np.concatenate((x_rev_speech, this_rev_speech), axis=0)
                else:
                    if x_rev_speech.shape[1] > this_rev_speech.shape[1]:
                        npads = x_rev_speech.shape[1] - this_rev_speech.shape[1]
                        this_rev_speech = np.concatenate(
                            (this_rev_speech, np.zeros((this_rev_speech.shape[0], npads)
                                                       ).astype(
                                this_rev_speech.dtype)), axis=1)
                    x_rev_speech.resize(
                        (x_rev_speech.shape[0] + this_rev_speech.shape[0],
                         x_rev_speech.shape[1]), refcheck=False)
                    x_rev_speech[-this_rev_speech.shape[0]:, :] = this_rev_speech
        else:
            x = np.array(airs)
            if speech_files is not None:
                x_speech = np.array(this_speech)
                x_rev_speech = np.array(this_rev_speech)

    if save_speech_associations:
        from utils_base import run_command
        df = pd.DataFrame(
            {'air': wav_files,
             'speech': np.array(speech_files)[associations]
             if given_associations is None else
             given_associations['speech'],
             'offsets': offsets
             if given_associations is None else
             given_associations['offsets']})

        df.to_csv(filename_associations, index=False)
        print('Saved: ' + filename_associations + ('' if given_associations is None else
                                                   ' which was created from the given associations'
                                                   ))
        if copy_associations_to is not None:
            run_command('cp ' + filename_associations + ' ' + copy_associations_to)
            print('Saved: ' + copy_associations_to)

    if fs is not None:
        print('Got ' + str(x.shape[0]) + ' AIRs of duration ' + str(x.shape[1] / float(fs)))
    else:
        print('Got ' + str(x.shape[0]) + ' AIRs of length ' + str(x.shape[1]))

    if speech_files is not None:
        proc_data = x_rev_speech
    else:
        proc_data = x

    if drop_speech:
        x_rev_speech = []
        x_speech = []
        x = []

    if no_fex:
        x_out = None
        print('Skipping feature extraction')
    else:
        x_out = data_post_proc(np.array(proc_data), forced_fs, start_at_max, framesize,
                               get_pow_spec, max_air_len, wavform_logpow)

        print('Left with ' + str(x_out.shape) + ' AIR features data ')

    ids = ids.astype(str)

    wrote_h5 = False
    if write_cached_latest:
        try:
            hf = File(latest_file, 'w')
            if no_fex:
                hf.create_dataset('x', data=[])
            else:
                hf.create_dataset('x', data=x_out)
            hf.create_dataset('y', data=[])
            hf.create_dataset('ids', data=ids)
            hf.create_dataset('class_names', data=[])
            hf.create_dataset('airs', data=x)
            hf.create_dataset('utts', data=utt_per_env)
            if speech_files is not None:
                hf.create_dataset('clean_speech', data=x_speech)
                hf.create_dataset('rev_names', data=x_rev_speech)
            else:
                hf.create_dataset('clean_speech', data=[])
                hf.create_dataset('rev_names', data=[])
            hf.close()
            wrote_h5 = True
            print('Wrote: ' + str(latest_file))
        except IOError as ME:
            print('Cache writing failed with ' + str(ME.message))

        if (not wrote_h5) and as_hdf5_ds:
            raise AssertionError('Could not provide data in correct format')
        if as_hdf5_ds:
            hf = File(latest_file, 'r')
            x_out = hf['x']
            ids = hf['ids']
            x = hf['airs']
            x_speech = hf['clean_speech']
            x_rev_speech = hf['rev_names']
            # hf.close()

    return (x_out, None), ids, None, (x, x_speech, x_rev_speech), utt_per_env


def compile_ace_h5(wav_loc, saveloc, ft='.wav', all_single_channel=False):
    """

    Create an HDF5 dataset which contains information about a set of files which describe AIRs of
    acoustic environments. This file can be used to train DNNs using ace_discriminative_nets.py

    Args:
        wav_loc: The location of the wav files as a list
        saveloc: The location to save to the HDF5 file
        ft: The file type to look for
        all_single_channel: Assume that all responses are single channel (faster and does not
        require soxi)

    Returns:
        Nothing

    """
    from utils_base import find_all_ft, run_command
    try:
        from os.path import abspath
    except ImportError:
        raise
    from h5py import File

    all_wavs = find_all_ft(wav_loc, ft=ft, use_find=True)
    channels = []
    for i in range(len(all_wavs)):
        print('Reading : ' + all_wavs[i])
        all_wavs[i] = abspath(all_wavs[i])
        if all_single_channel:
            channels.append('1')
        else:
            try:
                channels.append(run_command('soxi -c ' + all_wavs[i])[0])
            except OSError as ME:
                print('I think that soxi is not installed because when i tried to use it to get '
                      'the number of channels, i got this ' + ME.message)
                raise

    hf = File(saveloc, 'w')
    hf.create_dataset('filenames', data=all_wavs)
    hf.create_dataset('chan', data=channels)
    hf.close()
    print('Done with : ' + saveloc)


def get_ace_xy(h5_file='../results_dir/ace_h5_info.h5', ace_base='../Local_Databases/AIR/ACE/',
               y_type='room', group_by=None, utt_per_env=1, speech_files=None,
               print_distributions=False,
               parse_as_dirctories=False,
               choose_channel=None,
               **kwargs):
    """

    Collects training data and labels for traiing of DNNs using ace_discriminative_nets,
    based on the ACE Challenge data[1].

    Args:
        h5_file: Location of HFD5 dataset file for the ACE database, which is provided with this
        repository at Code/results_dir/ace_h5_info.h5. Contains information about the filenames,
        number of channels and also ground truth acoustic parameter values. If you want to create a
        new one, then use fe_utils.compile_ace_h5
        ace_base: The location of the ACE database data
        y_type: Creating labels from the data using specific information. This
        can be either of:
             'room', 'recording', 'array', 'recording', 'position', 'air'
        group_by: Creating grouping information from the data using specific information. This
        can be either of:
             'room', 'recording', 'array', 'recording', 'position', 'air'
        utt_per_env: Number of speech utterances to convolve with each AIR
        speech_files: Speech directory to pick up speech from and convolve it with the AIRs
        print_distributions: Print data information with regards to class distributions
        parse_as_dirctories: (ignored)
        choose_channel: (ignored)
        **kwargs: Additional arguments to be passed to read_airs_from_wavs

    Returns:
        (X, Y), Sample_names, Class_names,
        (AIRs, Speech, Reverberant_speech),
        (Group_name, Groups)

    """
    from h5py import File
    import numpy as np
    try:
        from os.path import basename
    except ImportError:
        raise
    from utils_base import flatten_list
    parse_as_dirctories = False

    hf = File(h5_file, 'r')
    wav_files = (np.array(hf.get('filenames')).astype(str)).tolist()
    chan = (np.array(hf.get('chan')).astype(int) - 1).tolist()

    type_dict = {'502': 'Office', '803': 'Office', '503': 'Meeting_Room', '611': 'Meeting_Room',
                 '403a': 'Lecture_Room', '508': 'Lecture_Room', 'EE-lobby': 'Building_Lobby'}
    basenames = [thename.split('/')[-1].replace('EE_lobby', 'EE-lobby') for thename in wav_files]
    room = [thename.split('_')[1] for thename in basenames]
    array = [thename.split('_')[0] for thename in basenames]
    room_type = [type_dict[thename.split('_')[1]] for thename in basenames]
    recording = basenames

    if ace_base is None:
        x_out = None
        x = None
        x_speech = None
        x_rev_speech = None
        ids = flatten_list([[basename(this_file).replace('.wav', '') + '_' + str(j) + '_ch' + str(k)
                             for k in range(chan[i])]
                            for i, this_file in enumerate(wav_files)
                            for j in range(utt_per_env)])
    else:
        for i in range(len(wav_files)):
            wav_files[i] = ace_base + '/' + wav_files[i]
        (x_out, _), \
        ids, _, \
        (x, x_speech, x_rev_speech), \
        utt_per_env = read_airs_from_wavs(
            wav_files, utt_per_env=utt_per_env, speech_files=speech_files,
            parse_as_dirctories=parse_as_dirctories,
            choose_channel=chan,
            **kwargs)
    if 'ch' not in ids[0]:
        if np.sum(['ch' in ids[i] for i in range(len(ids))]) > 0:
            raise AssertionError('Unexpected condition')
        ch = [0 for _ in range(len(ids))]
    else:
        ch = [int(i.split('ch')[1]) for i in ids]

    y = []
    class_names = []

    flat_back_y = False
    if not (isinstance(y_type, list) or isinstance(y_type, tuple)):
        flat_back_y = True
        y_type = (y_type,)

    for this_y_type in y_type:
        if this_y_type == 'room':
            new_y, new_class_names, _ = categorical_to_mat(room)
            def_group_by = 'recording'
        elif this_y_type == 'type':
            new_y, new_class_names, _ = categorical_to_mat(room_type)
            def_group_by = 'room'
        elif this_y_type == 'array':
            new_y, new_class_names, _ = categorical_to_mat(array)
            def_group_by = 'recording'
        elif this_y_type == 'position' or y_type == 'position':
            new_y, new_class_names, _ = categorical_to_mat(recording)
            def_group_by = 'air'
        elif this_y_type == 'channel':
            new_y, new_class_names, _ = categorical_to_mat(ch)
            def_group_by = 'position'
        else:
            raise AssertionError('Invalid y_type')
        y.append(new_y)
        class_names.append(new_class_names)

    flat_back_groups = False
    if group_by is None:
        group_by = (def_group_by,)
    elif not (isinstance(group_by, list) or isinstance(group_by, tuple)):
        flat_back_groups = True
        group_by = (group_by,)

    group_name, groups = ([], [])
    for this_group_by in group_by:
        if this_group_by == 'recording' or this_group_by == 'position':
            _, new_group_name, new_groups = categorical_to_mat(recording)
        elif this_group_by == 'room':
            _, new_group_name, new_groups = categorical_to_mat(room)
        elif this_group_by == 'array':
            _, new_group_name, new_groups = categorical_to_mat(array)
        elif this_group_by == 'air':
            new_groups = np.atleast_2d(np.arange(y.shape[0])).T
            new_group_name = np.array(ids)
        elif this_group_by == 'channel':
            max_ch = max(ch) + 1
            ch_array = np.zeros((len(ch), max_ch), dtype=bool)
            for i in range(len(ch)):
                ch_array[i, ch[i]] = True
            new_groups = np.array(ch_array)
            new_group_name = np.array(['ch_' + str(i) for i in range(max_ch)])
        else:
            raise AssertionError('Invalid group_by ' + this_group_by)
        group_name.append(new_group_name)
        groups.append(new_groups)

    for i in range(len(y)):
        if print_distributions:
            print_split_report(y[i])
        if np.any(~(np.sum(y[i], axis=1) == 1)):
            raise AssertionError('Invalid y outputs')
        y[i] = np.concatenate([y[i][ii:ii + 1, :] for ii in range(y[i].shape[0])
                               for _ in range(utt_per_env)],
                              axis=0)
    for ii in range(len(groups)):
        groups[ii] = [np.concatenate([list(range(i * utt_per_env, (i + 1) * utt_per_env))
                                      for i in groups[ii][j]]).astype(int)
                      for j in range(len(groups[ii]))]

    y = tuple(y)
    class_names = tuple(class_names)
    groups = tuple(groups)
    group_name = tuple(group_name)
    if flat_back_groups:
        groups = groups[0]
        group_name = group_name[0]
    if flat_back_y:
        y = y[0]
        class_names = class_names[0]

    return (x_out, y), ids, class_names, (x, x_speech, x_rev_speech), (group_name, groups)


def categorical_to_mat(categorical):
    """

    Converts a categorical variable vector into a one-hot class matrix

    Args:
        categorical: The categorical variable vector

    Returns:
        The class matrix as an array
        The name of each class corresponding to each column of the array
        The list of indices of 'categorical' which belong to the labels in 'unique_vals'

    """
    import numpy as np
    categorical = np.array(categorical)
    unique_vals = np.unique(categorical)
    y = np.zeros((categorical.size, unique_vals.size), dtype=bool)
    groups = []
    for i, val in enumerate(unique_vals):
        y[categorical == val, i] = True
        groups.append(np.where(categorical == val)[-1])
    # groups=np.array(groups)
    return y, unique_vals, groups


def collect_wavs(filenames, dest_fs=None):
    """

    Collects and packages a set of wav files to an array of samples

    Args:
        filenames: File locations as a list
        dest_fs: Sampling frequency to use

    Returns:
        An array of the samples of the files as [N_files x N_samples]

    """
    import numpy as np
    from scipy.io.wavfile import read
    from utils_spaudio import my_resample
    if not (isinstance(filenames, list) or isinstance(filenames, tuple)):
        filenames = [filenames]
    samples = []
    max_len = 0
    for the_filename in filenames:
        fs, new_samples = read(the_filename)
        if dest_fs:
            new_samples = my_resample(new_samples, fs, dest_fs)
        if new_samples.ndim == 1:
            new_samples = np.atleast_2d(new_samples).T
        max_len = max(max_len, new_samples.shape[0])
        samples.append(new_samples)
    for i in range(len(samples)):
        this_len = samples[i].shape[0]
        missing = max_len - this_len
        if missing > 0:
            samples[i] = np.concatenate(
                (samples[i], np.zeros((missing, samples[i].shape[1]), dtype=samples[i].dtype)))

    out = np.concatenate([samples[i].T for i in range(len(samples))], axis=0)

    return out

