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

This file describes the code used for developing the work and evaluating the experiments
described in:
C. Papayiannis, C. Evers, and P. A. Naylor, "Data Augmentation of Room Classifiers using Generative Adversarial Networks," arXiv preprint arXiv:1901.03257, 2018.

This file was original distributed in the repository at:
{repo}

If you use this code in your work, then cite:
C. Papayiannis, C. Evers, and P. A. Naylor, "Data Augmentation of Room Classifiers using Generative Adversarial Networks," arXiv preprint arXiv:1901.03257, 2018.

"""

from __future__ import division


def sample_air_e2e(generator, epoch, deflatten_op,
                   savefolder='/tmp/', speech_samples=None,
                   response_filename="air_%d.wav", save_responses=True, do_plots=True, samples=10):
    """
    Samples the distribution of the noise and _filters_ it through the generator to provide an
    AIR. The function assumes a GAN which was trained using and end-to-end method (trained and
    operates in the FIR tap domain)

    Args:
        generator: The generator network
        epoch: The current epoch
        deflatten_op: The operation to deflatten the data of the GAN and make them translatable
        as an acoustic environment instance
        excitation_pool: The pool of excitation signals to use a dictionary for excitations
        savefolder: The folder in which to save the results.
        speech_samples: The samples of the speech
        response_filename: The file naming strategy to follow for saving the AIR results. Should
        include a %d, which will be the AIR index
        save_responses: Flag to enable response saving
        do_plots: Flag to enable plotting
        samples: Number of AIRs to generate
        raw_parameter_data: The raw data used to train the GAN (used to compare the distributions)

    Returns:
        Nothing

    """
    from utils_spaudio import write_wav
    from utils_base import run_command
    from time import time
    try:
        from os.path import isfile
    except ImportError:
        raise

    render = do_plots or save_responses

    timestamp = str(time())
    rejects_loc = '/tmp/rejects_' + timestamp + '/'

    r, c = int(np.ceil(samples / 2.)), 2
    noise = np.random.normal(0, 1, (samples, noise_dim))
    gen_imgs = generator.predict(noise)
    filename = savefolder + "/air_" + format(epoch, '09d') + ".pdf"
    response_savefolder = savefolder + "/air_samples/"
    response_filename = response_savefolder + response_filename
    if save_responses:
        run_command('mkdir -p ' + response_savefolder)
    savefolder += '/epoch_' + format(epoch, '09d') + '/'
    run_command('mkdir -p ' + savefolder)
    if render:
        run_command('mkdir -p ' + rejects_loc)

    net_loc = savefolder + '/gan_generator.h5'
    print('Saving model at ' + net_loc)
    generator.save(net_loc)
    print('Saved ' + net_loc)

    if do_plots:
        fig, axs = plt.subplots(r, c, figsize=(10, 6))
    cnt = 0
    max_tries = 10000
    print('Working on: ' + filename)

    airs = deflatten_op(gen_imgs)

    for i in range(r):
        for j in range(c):
            if cnt >= samples:
                break
            if render:
                print('Working on AIR with count ' + str(cnt))
            do_it = True
            tries = 0
            while do_it:
                tries += 1
                reject_name = (rejects_loc + 'reject_idx_' + str(cnt) +
                               '_try_' + str(tries) + '.wav')
                try:
                    do_it = False
                    reco = airs[cnt, :]
                    if render:
                        wavfile.write(reject_name, int(global_fs), reco)
                        run_command('rm ' + reject_name)
                except AssertionError as ME:
                    if isfile(reject_name):
                        print('Rejected response saved at ' + reject_name)
                    do_it = True
                    print('Failed ' + str(
                        tries) + ' times to create an AIR and retrying at cnt ' + str(
                        cnt) + ' with ' + ME.message)
                    if tries > max_tries:
                        print('Giving up...')
                        raise
                    noise = np.random.normal(0, 1, (1, noise_dim))
                    airs[cnt, :] = deflatten_op(generator.predict(noise))

            if do_plots:
                plotspan = min(reco.size, int(round(0.2 * global_fs)))
                axs[i, j].plot(reco[0:plotspan], linewidth=0.5)
                axs[i, j].axis('off')
            cnt += 1

    if do_plots:
        fig.savefig(filename)
        print('Saved: ' + filename)
        plt.close()

    if save_responses or (speech_samples is not None):
        for i, air_samples in enumerate(airs):
            air_samples = air_samples / float(np.abs(air_samples).max())
            if save_responses:
                write_wav(response_filename % i, global_fs, air_samples)
            if speech_samples is not None:
                rev_speech = np.convolve(speech_samples, air_samples, 'same')
                filename_wav = savefolder + '/rev_speech_' + str(i) + '.wav'
                write_wav(filename_wav, global_fs, rev_speech)
                # print('Wrote: ' + filename_wav)


def sample_air(generator, epoch, deflatten_op, excitation_pool,
               savefolder='/tmp/', speech_samples=None,
               response_filename="air_%d.wav", save_responses=True,
               do_plots=True, samples=10,
               raw_parameter_data=None):
    """
    Samples the distribution of the noise and _filters_ it through the generator to provide an
    instance of an acoustic environment. The function plots the results as a distribution of
    model parameters and also saves the produces AIRs and can convolve them with anechoic speech.

    Args:
        generator: The generator network
        epoch: The current epoch
        deflatten_op: The operation to deflatten the data of the GAN and make them translatable
        as an acoustic environment instance
        excitation_pool: The pool of excitation signals to use a dictionary for excitations
        savefolder: The folder in which to save the results.
        speech_samples: The samples of the speech
        response_filename: The file naming strategy to follow for saving the AIR results. Should
        include a %d, which will be the AIR index
        save_responses: Flag to enable response saving
        do_plots: Flag to enable plotting
        samples: Number of AIRs to generate
        raw_parameter_data: The raw data used to train the GAN (used to compare the distributions)

    Returns:
        Nothing

    """
    from utils_spaudio import write_wav
    from utils_base import run_command
    from random import randint
    from time import time
    try:
        from os.path import isfile
    except ImportError:
        raise
    from h5py import File
    from third_party import zplane

    render = do_plots or save_responses

    timestamp = str(time())
    rejects_loc = '/tmp/rejects_' + timestamp + '/'

    r, c = int(np.ceil(samples / 2.)), 2
    noise = np.random.normal(0, 1, (samples, noise_dim))
    gen_imgs = generator.predict(noise)
    filename = savefolder + "/air_" + format(epoch, '09d') + ".pdf"
    response_savefolder = savefolder + "/air_samples/"
    response_filename = response_savefolder + response_filename
    if save_responses:
        run_command('mkdir -p ' + response_savefolder)
    savefolder += '/epoch_' + format(epoch, '09d') + '/'
    run_command('mkdir -p ' + savefolder)
    if render:
        run_command('mkdir -p ' + rejects_loc)

    net_loc = savefolder + '/gan_generator.h5'
    print('Saving model at ' + net_loc)
    generator.save(net_loc)
    print('Saved ' + net_loc)

    if do_plots:
        fig, axs = plt.subplots(r, c, figsize=(10, 6))
    cnt = 0
    max_tries = 10000
    airs = []
    print('Working on: ' + filename)

    all_reflections_final = []

    excitation = excitation_pool[
                 [randint(0, excitation_pool.shape[0] - 1) for _ in range(samples)], :]
    reflections, acoustic_params, ar_coeff, ma_coeff = deflatten_op(gen_imgs)

    for i in range(r):
        for j in range(c):
            if cnt >= samples:
                break
            if render:
                print('Working on AIR with count ' + str(cnt))
            do_it = True
            tries = 0
            while do_it:
                tries += 1
                reject_name = (rejects_loc + 'reject_idx_' + str(cnt) +
                               '_try_' + str(tries) + '.wav')
                try:
                    do_it = False
                    reco, reflection_final = acenv_reconstruct(
                        reflections[cnt, ...],
                        acoustic_params[cnt, ...],
                        excitation[cnt, ...],
                        ar_coeff[cnt, ...],
                        ma_coeff[cnt, ...],
                        air_len * global_fs,
                        global_fs,
                        'reco_' + str(epoch),
                        render=render)
                    all_reflections_final.append(reflection_final)
                    if render:
                        wavfile.write(reject_name, int(global_fs), reco)
                        run_command('rm ' + reject_name)
                except AssertionError as ME:
                    if isfile(reject_name):
                        print('Rejected response saved at ' + reject_name)
                    do_it = True
                    print('Failed ' + str(
                        tries) + ' times to create an AIR and retrying at cnt ' + str(
                        cnt) + ' with ' + ME.message)
                    if tries > max_tries:
                        print('Giving up...')
                        raise
                    noise = np.random.normal(0, 1, (1, noise_dim))
                    gen_imgs[cnt, :] = generator.predict(noise)
                    excitation[cnt, ...] = excitation_pool[
                                           randint(0, excitation_pool.shape[0] - 1), :]
                    reflections[cnt, ...], acoustic_params[cnt, ...], \
                    ar_coeff[cnt, ...], ma_coeff[cnt, ...] = deflatten_op(gen_imgs[cnt, ...])

            airs.append(reco)
            if do_plots:
                plotspan = min(reco.size, int(round(0.2 * global_fs)))
                axs[i, j].plot(reco[0:plotspan], linewidth=0.5)
                axs[i, j].axis('off')
            cnt += 1

    all_reflections_final = np.concatenate(all_reflections_final, axis=0)
    raw_ref = np.concatenate(raw_parameter_data[0], axis=0)
    raw_t60 = raw_parameter_data[1][:, 0]
    raw_drr = raw_parameter_data[1][:, 1]

    get_bins = lambda x, slack: np.arange(np.min(x) - slack,
                                          np.max(x) + slack,
                                          (2 * slack + np.max(x) - np.min(x)) / 10.)

    if do_plots:
        fig.savefig(filename)
        print('Saved: ' + filename)
        plt.close()
    if raw_parameter_data is not None:
        hist_args = {'density': True}
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))
        # Amplitudes
        axs[0, 0].hist(
            raw_ref[raw_ref[:, 0] > 0, 1],
            bins=get_bins(raw_ref[raw_ref[:, 0] > 0, 1], .1), **hist_args)
        ylims = axs[0, 0].get_ylim()
        axs[0, 0].cla()
        axs[0, 0].hist(
            [raw_ref[raw_ref[:, 0] > 0, 1], all_reflections_final[:, 1]],
            bins=get_bins(raw_ref[raw_ref[:, 0] > 0, 1], .1), **hist_args)
        axs[0, 0].set_ylim(ylims)
        axs[0, 0].legend(['Amplitudes Data', 'Amplitudes GAN'])
        # Delays
        axs[0, 1].hist(
            raw_ref[raw_ref[:, 0] > 0, 0],
            bins=get_bins(raw_ref[raw_ref[:, 0] > 0, 0], 0.0005), **hist_args)
        ylims = axs[0, 1].get_ylim()
        axs[0, 1].cla()
        axs[0, 1].hist(
            [raw_ref[raw_ref[:, 0] > 0, 0], all_reflections_final[:, 0]],
            bins=get_bins(raw_ref[raw_ref[:, 0] > 0, 0], 0.0005), **hist_args)
        axs[0, 1].set_ylim(ylims)
        axs[0, 1].legend(['Delays Data (s)', 'Delays GAN (s)'])
        # T60
        axs[1, 0].hist(raw_t60,
                       bins=get_bins(raw_t60, 0.1), **hist_args)
        ylims = axs[1, 0].get_ylim()
        axs[1, 0].cla()
        axs[1, 0].hist(
            [raw_t60, acoustic_params[:, 0]],
            bins=get_bins(raw_t60, 0.1), **hist_args)
        axs[1, 0].set_ylim(ylims)
        axs[1, 0].legend(['T60 Data (s)', 'T60 GAN (s)'])
        # DRR
        axs[1, 1].hist(
            raw_drr,
            bins=get_bins(raw_drr, 1), **hist_args)
        ylims = axs[1, 1].get_ylim()
        axs[1, 1].cla()
        axs[1, 1].hist(
            [raw_drr, acoustic_params[:, 1]],
            bins=get_bins(raw_drr, 1), **hist_args)
        axs[1, 1].set_ylim(ylims)
        axs[1, 1].legend(['DRR Data (dB)', 'DRR GAN (dB)'])
        # Poles and Zeros
        for ii in range(2):
            for jj in range(2):
                axs[ii, jj].get_yaxis().set_visible(False)
        for ii in range(raw_parameter_data[2].shape[0]):
            zplane(raw_parameter_data[3][ii, ...], [1], color='b', ax=axs[2, 0],
                   ignore_extremes=True, zeros_marker='.', poles_marker='.')
            zplane([1], raw_parameter_data[2][ii, ...], color='b', ax=axs[2, 1],
                   ignore_extremes=True, zeros_marker='.', poles_marker='.')
        for ii in range(ar_coeff.shape[0]):
            zplane(ma_coeff[ii, ...], [1], color='r', ax=axs[2, 0], ignore_extremes=True,
                   zeros_marker='.', poles_marker='.')
            zplane([1], ar_coeff[ii, ...], color='r', ax=axs[2, 1], ignore_extremes=True,
                   zeros_marker='.', poles_marker='.')

        axs[2, 0].set_ylim([-1.5, 1.5])
        axs[2, 0].set_xlim([-1.5, 1.5])
        axs[2, 1].set_ylim([-1.5, 1.5])
        axs[2, 1].set_xlim([-1.5, 1.5])
        fixed_line_specs = {'markersize': 5, 'linestyle': '', 'color': 'none', 'marker': '.'}
        custom_lines = [
            plt.Line2D([0], [0], markeredgecolor='b', **fixed_line_specs),
            plt.Line2D([0], [0], markeredgecolor='r', **fixed_line_specs)]
        axs[2, 0].legend(custom_lines, ['Zeros Data', 'Zeros GAN'], ncol=2)
        axs[2, 1].legend(custom_lines, ['Poles Data', 'Poles GAN'], ncol=2)
        # axs[0,1].grid(True)
        filename = savefolder + "/distributions_" + format(epoch, '09d') + ".pdf"
        fig.savefig(filename)
        print('Saved: ' + filename)
        plt.close()

    filename_h5 = savelocation + 'ganed_params.h5'

    print('Saving GAN predicted params to ' + filename_h5)
    hf = File(filename_h5, 'w')
    hf.create_dataset('names', data=[response_filename % i for i in range(samples)])
    hf.create_dataset('y', data=reflections)
    hf.create_dataset('AcousticParams', data=acoustic_params)
    hf.create_dataset('AR', data=ar_coeff)
    hf.create_dataset('MA', data=ma_coeff)
    hf.create_dataset('Excitation', data=excitation)
    hf.close()
    print 'Wrote: ' + str(filename_h5)

    if save_responses or (speech_samples is not None):
        for i, air_samples in enumerate(airs):
            air_samples = air_samples / float(np.abs(air_samples).max())
            if save_responses:
                write_wav(response_filename % i, global_fs, air_samples)
            if speech_samples is not None:
                rev_speech = np.convolve(speech_samples, air_samples, 'same')
                filename_wav = savefolder + '/rev_speech_' + str(i) + '.wav'
                write_wav(filename_wav, global_fs, rev_speech)
                # print('Wrote: ' + filename_wav)


def acenv_reconstruct(
        reflections, acoustic_params, excitation, ar_coeff, ma_coeff, air_len, fs, name,
        no_scales=False, savefolder=None, verbose=False, render=True):
    """
    Reconstructs an acoustic environment from the GAN generated encoding

    Args:
        reflections: The array which contains the sparse reflection data
        acoustic_params: The vector of acoustic parameter values
        excitation: The excitation vector
        ar_coeff: The denominator coefficients for the IIR tail filter
        ma_coeff: The numerator coefficients for the IIR tail filter
        air_len: The number of taps in the AIR reconstruction
        fs: The sampling frequency
        name: The name of the environment
        no_scales: No reflection scaling
        savefolder: Folder to save the results to
        verbose: Flag for verbose printing
        render: Flag to enable AIR rendering

    Returns:
        The AIR reconstruction
        The reflection parameters

    """
    import numpy as np
    from acenv import AcEnv
    from acenvgenmodel_base import AcEnvGenModelBase
    try:
        from scipy.signal import tukey
    except ImportError:
        raise

    max_ref = 0.05
    reflections[:, 0][reflections[:, 0] > max_ref] = 0
    reflections = reflections[(-reflections[:, 0]).argsort(), :]
    try:
        ref_marker = np.where(reflections[:, 0] > ref_accepted_delay_th)[0][-1] + 1
    except IndexError:
        raise AssertionError('No good reflections')

    reflections = reflections[0:ref_marker, :]
    excitation = excitation * tukey(excitation.size, 0.1)

    dummy_air = np.zeros(int(air_len))
    dummy_air[int(fs * global_dpath)] = 1
    acenv = AcEnv(is_simulation=True, name=name,
                  samples=dummy_air,
                  sampling_freq=fs,
                  silent_samples_offset=False,
                  )

    final_air = None
    if render:
        acenvmodel = AcEnvGenModelBase(
            acenv, 0,
            quiet=True,
            use_matlab=False,
            offset_to_max=False,
            given_excitation=excitation,
            transition_time=0.024,
            early_part_only=False,
            enforce_dp_sanity_check=False,
            ac_params={'t60': acoustic_params[0],
                       'drr': 10 ** (acoustic_params[1] / 10.),
                       'drr_early': 10 ** (acoustic_params[2] / 10.),
                       'drr_late': 10 ** (acoustic_params[3] / 10.)})

        acenvmodel.paramset_directsound_amp = 1.
        acenvmodel.paramset_directsound_soa = global_dpath * fs

        acenvmodel.paramset_reflections_soa = reflections[:, 0] * acenvmodel.sampling_freq + \
                                              acenvmodel.paramset_directsound_soa
        min_soa = np.min(acenvmodel.paramset_reflections_soa)
        if min_soa < 0:
            acenvmodel.paramset_reflections_soa = acenvmodel.paramset_reflections_soa + 2 * min_soa
        elif min_soa == 0:
            acenvmodel.paramset_reflections_soa = acenvmodel.paramset_reflections_soa + \
                                                  np.sort(acenvmodel.paramset_reflections_soa)[1]
        acenvmodel.paramset_reflections_amp = reflections[:, 1]
        if no_scales:
            acenvmodel.paramset_reflections_amp[:] = 0
        acenvmodel.paramset_reflections_fval_reduction = np.zeros_like(
            acenvmodel.paramset_reflections_amp)

        acenvmodel.paramset_tail_modes_ar_coeff = ar_coeff
        acenvmodel.paramset_tail_modes_ma_coeff = ma_coeff

        acenvmodel.air_reconstruct(
            assert_on_0_energy=False)
        final_air = acenvmodel.air_reconstructed_from_model
        if savefolder is not None:
            acenvmodel.plot_modeling_results(
                interactive=False,
                saveloc=savefolder + name + '.pdf',
                verbose=verbose)

        if verbose:
            acenvmodel.print_reflection_report(verbose=True)

    return final_air, reflections


def get_h5_data(reflection_param_h5, fields):
    """
    Collects the data in the HDF5 dataset saved by 'acenvgenmodel_collect_results.py'. Returns
    them in terms of the array containing the data for each field, the shape of the data and the
    names corresponding to each acoustic environment being modeled.

    Args:
        reflection_param_h5: The location of the saved HDF5 dataset
        fields: The field names to extract the values for from the HDF5 dataset

    Returns:
        The values of the data in the fields as numpy arrays
        The shape of the descriptor for each environment per field
        The name of each environment

    """
    import numpy as np
    from h5py import File
    print('Reading :' + reflection_param_h5)
    hf = File(reflection_param_h5, 'r')
    field_values = []
    for i, the_field in enumerate(fields):
        field_values.append(np.array(hf.get(the_field), dtype=float))
        if the_field == 'y':
            if not model_hits:
                field_values[-1] = field_values[-1][:, :, [0, 1]]
            else:
                field_values[-1] = field_values[-1][:, :, 0:-1]
    names = np.array(hf.get('names')).flatten()
    hf.close()
    if not np.all(np.equal(field_values[0].shape[0],
                           [field_values[ii].shape[0] for ii in range(1, len(field_values))])):
        raise AssertionError('Data lengths don\'t match')
    field_shapes = [field_values[ii].shape[1:] for ii in range(len(field_values))]
    print('Got ' + str(field_values[0].shape[0]) + ' entries')
    return field_values, field_shapes, names


def flatten_h5_data(field_values):
    """
    Collects the data which were packaged by 'acenvgenmodel_collect_results.py' and packages them in
    a 2D array, which is used for the training of the GANs.

    Args:
        field_values: The values of the data stored in the HDF5 dataset stored by
        'acenvgenmodel_collect_results.py'. Each entry in the each file, represents one acoustic
        environment

    Returns:
        The array containing the flattened data

    """
    import numpy as np
    for i in range(len(field_values)):
        field_values[i] = np.flip(field_values[i].reshape(
            (field_values[i].shape[0], np.prod(field_values[i].shape[1:]))), axis=-1)
    y_all = np.concatenate(field_values, axis=1)
    return y_all


def deflatten_h5_data(y_all, field_shapes):
    """
    Reverses the operation of 'flatten_h5_data(.)'. it repacks the data generated by GANs in
     terms  of the parameters which they represent.

    Args:
        y_all: A 2D array containing the flattened data, which are encodings of acoustic
        environments
        field_shapes: The shapes of the fields. These will be used to allocate data in the 2D
        matrix to individual fields of the returned result

    Returns:
        The arrays containing the data of each of the fields

    """
    import numpy as np
    field_values = []
    counter = 0
    for i in range(len(field_shapes)):
        field_length = np.prod(field_shapes[i])
        next_counter = counter + field_length
        field_values.append(y_all[:, counter:next_counter])
        field_values[-1] = np.flip(
            field_values[-1].reshape(tuple([y_all.shape[0]] + list(field_shapes[i]))), axis=-1)
        counter = next_counter

    return tuple(field_values)


if __name__ == '__main__':
    import argparse
    import numpy as np
    from matplotlib import use
    from utils_dnntrain import get_scaler_descaler
    from utils_base import get_timestamp, isclose, matrix_stats
    from utils_spaudio import my_resample
    from scipy.io import wavfile
    from gan_model_classes import GAN, WGAN

    noise_dim = 20
    global_fs = 16E3
    global_dpath = 0.002
    model_hits = False
    ref_accepted_delay_th = 2.8E-4
    air_len = 2.5

    parser = argparse.ArgumentParser(description='Arguments for GAN training')

    parser.add_argument('--h5', type=str, dest='file_loc', default=None,
                        help='Location where the h5 file is located, '
                             'generated using ac_gm_early_collection_reflection_results.py')
    parser.add_argument('--testarray', dest='testarray', type=str,
                        default='EM32',
                        help='Array to ignore during data collection. '
                             'This will be the array which you will use to test '
                             'a network you train with. Tis script should be used '
                             'to create data for data augmentation, and therefore '
                             'you should not use data from hte test array to inject '
                             'information in the training data. This option applies only to '
                             'end-to-end models')
    parser.add_argument('--acebase', dest='acebase', type=str,
                        default='../Local_Databases/AIR/ACE16/',
                        help='Location of the train airs for the end-to-end mode')
    parser.add_argument('--saveloc', type=str, default='/tmp/', dest='saveloc',
                        help='Location to save results')
    parser.add_argument('--airname', type=str, default="air_%d.wav", dest='airname',
                        help='File-naming structure for AIR wav files')
    parser.add_argument('--epochs', type=int, default=30000, dest='epochs',
                        help='Number of epochs')
    parser.add_argument('--samples', type=int, default=10, dest='samples',
                        help='Number of samples to produce')
    parser.add_argument('--interval', type=int, default=None, dest='interval',
                        help='Sample interval')
    parser.add_argument('--speech', type=str,
                        default=None,
                        dest='speech_file',
                        help='Location where the a clean wav speech file is located')
    parser.add_argument('--nodisplay', dest='nodisplay', action="store_true",
                        default=False, help='Use the Agg backend for matplotlib')
    parser.add_argument('--noair', dest='noair', action="store_true",
                        default=False, help='Do not save the AIRs as wav files')

    parser.add_argument('--e2e', dest='e2e', action="store_true",
                        default=False, help='Treat as end to end')

    parser.add_argument('--nocr', dest='nocr', action="store_true",
                        default=False, help='Don\'t use carriage return on progress reports')
    parser.add_argument('--wgan', dest='wgan', action="store_true",
                        default=False, help='Use WassersteinGAN')
    parser.add_argument('--noplots', dest='noplots', action="store_true",
                        default=False, help='No plotting')
    args = parser.parse_args()
    print('Args given : ' + str(args))

    if args.nodisplay:
        use('Agg')
    import matplotlib.pyplot as plt

    doing_e2e = args.e2e

    file_loc = args.file_loc
    speech_file = args.speech_file

    savelocation = args.saveloc + '/gan_acenv_' + get_timestamp() + '/'
    if speech_file is not None:
        fs, speech_samples = wavfile.read(speech_file)
        speech_samples = my_resample(speech_samples, fs, global_fs)
        speech_samples = speech_samples[0:int(global_fs * 6)]
        speech_samples = speech_samples / float(np.abs(speech_samples).max())
    else:
        speech_samples = None

    if doing_e2e:
        from fe_utils import get_ace_xy
        from utils_spaudio import align_max_samples

        if file_loc is None:
            file_loc = '../results_dir/ace_h5_info.h5'
        model_framesize = 64
        max_air_len_import = air_len
        wavform_logpow = False
        get_pow_spec = False
        start_at_max = True
        feature_ex_config = {
            'max_air_read': max_air_len_import,
            'max_air_len': max_air_len_import,
            'fs': global_fs, 'forced_fs': global_fs,
            'no_fex': True,
            'framesize': model_framesize, 'keep_ids': None,
            'wavform_logpow': wavform_logpow, 'as_hdf5_ds': False,
            'get_pow_spec': get_pow_spec,
            'start_at_max': start_at_max, 'read_cached_latest': False,
            'speech_files': None}

        (_, _), ids_train, class_names_train, \
        (x_train, _, _), \
        (group_names_train, groups_train) = get_ace_xy(h5_file=file_loc,
                                                       ace_base=args.acebase,
                                                       scratchpad='/tmp/scrap/',
                                                       group_by='array',
                                                       cacheloc='/tmp/scrap/',
                                                       **feature_ex_config)

        test_array_idx = group_names_train.tolist().index(args.testarray)
        x_train = x_train[np.concatenate(
            np.array(groups_train)[np.setxor1d(range(len(group_names_train)),
                                               [test_array_idx])]).astype(int), ...]

        x_train = align_max_samples(x_train, scan_range=range(80))[0]
        x_train = x_train[:, int(np.abs(x_train[0, :]).argmax() - global_fs * global_dpath):]
        # print('True stats')
        # matrix_stats(x_train)
        print('True stats flat for ' + str(x_train.shape[0]) + ' data')
        matrix_stats(x_train, flat=True)
        scaler, descaler = get_scaler_descaler(x_train)
        x_train_sc = np.array(x_train)
        x_train = scaler(x_train).squeeze()
        # print('Norm stats')
        # matrix_stats(x_train)
        print('Norm stats flat')
        matrix_stats(x_train, flat=True)
        if np.any(np.abs(descaler(x_train) - x_train_sc) > 1E-10):
            raise AssertionError('Scaler descalers issue')
        deflatten_op = lambda y_all_in: np.atleast_2d(descaler(y_all_in).squeeze())
        epoch_op = lambda generator, epoch: sample_air_e2e(
            generator, epoch, deflatten_op,
            response_filename=args.airname,
            savefolder=savelocation,
            speech_samples=speech_samples,
            do_plots=not args.noplots,
            samples=args.samples,
            save_responses=not args.noair)
    else:
        if file_loc is None:
            raise AssertionError('If not doing end-to-end then you need to provide an h5 location')
        fileds_gen = ['y', 'AcousticParams', 'AR', 'MA']
        fields_pool = ['Excitation']
        field_values, field_shapes, _ = get_h5_data(file_loc, fileds_gen)
        raw_data = [np.array(i) for i in field_values]

        field_values_pool, field_shapes_pool, _ = get_h5_data(file_loc, fields_pool)
        x_train = flatten_h5_data(field_values)
        # print('True stats')
        # matrix_stats(x_train)
        print('True stats flat')
        matrix_stats(x_train, flat=True)
        scaler, descaler = get_scaler_descaler(x_train)
        x_train_sc = np.array(x_train)
        x_train = scaler(x_train).squeeze()
        # print('Norm stats')
        # matrix_stats(x_train)
        print('Norm stats flat')
        matrix_stats(x_train, flat=True)
        if np.any(np.abs(descaler(x_train) - x_train_sc) > 1E-10):
            raise AssertionError('Scaler descalers issue')
        deflatten_op = lambda y_all_in: deflatten_h5_data(
            np.atleast_2d(descaler(y_all_in).squeeze()),
            field_shapes)
        epoch_op = lambda generator, epoch: sample_air(
            generator, epoch, deflatten_op,
            field_values_pool[0],
            response_filename=args.airname,
            savefolder=savelocation,
            speech_samples=speech_samples,
            do_plots=not args.noplots,
            samples=args.samples,
            raw_parameter_data=raw_data,
            save_responses=not args.noair)

    gan = WGAN(x_train[0].shape, net_width=256) if args.wgan else GAN(
        x_train[0].shape, net_width=256)
    gan.train(x_train=x_train, epoch_op=epoch_op, nocr=args.nocr,
              epochs=args.epochs, batch_size=32,
              sample_interval=args.interval if args.interval is not None
              else args.epochs)
else:
    raise AssertionError('Must be main')
