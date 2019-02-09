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

This file models reverberant acoustic environments as described in the paper [1]. It was original
distributed in the repository at:
{repo}
If you use this code in your work, then cite [1].

Features:
*   Sparsely represents a given AIR in terms of individual reflections.
*   Estimates the TOA and amplitudes of reflections is AIRs
*   Estimates acoustic parameters from the AIR (T60, DRR, AR(MA) model)
*   Reconstructs the AIR from the model
*   Estimates the excitation signal used for the AIR measurement.
*   Evaluates the accuracy of the modeling using a number of measures such as the NPM and the
Itakura distance.

[1] C. Papayiannis, C. Evers, and P. A. Naylor, "Data Augmentation of Room Classifiers using Generative Adversarial Networks," arXiv preprint arXiv:1901.03257, 2018.

Usage:

Arguments for fitting model to AIR

optional arguments:
  -h, --help            show this help message and exit
  --array ARRAY         Array to use (EM32 Single Crucif Chromebook Lin8Ch)
  --channel CHANNEL     Channel to model from the array
  --fs FS               Sampling frequency to work with
  --transition TRANSITION
                        Early part duration after max energy sample
  --config CONFIG       Source receiver config (1, 2)
  --room ROOM           Room to model (EE_lobby 508 403a 503 611 502 803)
  --resultsdir RESULTSDIR
                        Directory to save results
  --interactive         Keep plots at front and pause until closed
  --early               Eary Only
  --detailedplots       Make plots at each stage
  --nodisplay           Use the Agg backend for matplotlib
  --nomatlabinit        Do not preload a matlab engine
  --tex                 Use latex style plotting fonts
  --plotexcitation      Plot the excitation signal
  --nocache             Disable reading from cached results
  --mgm                 Use a Modulated Gaussian Pulse for the excitation
  --delta               Use a delta for the excitation
  --speech SPEECH       Speech file to use for the results
  --air AIR             File path of given wav AIR. If this is given the all
                        references to ACE are lost. Working with this fixed
                        AIR. For the excitation, if no other AIRs are given
                        specifically for the excitation model then it will be
                        modeled as a delta.
  --excitationdir EXCITATIONDIR
                        File path of a folder contatining AIRs as wav files
                        used for estimating the excitation signal
  --nparams NPARAMS     Shrink model to a fixed number of parameters
  --lassocoef LASSOCOEF
                        Fraction of standard deviation of the error to move
                        away from the MSE solution of LASSO when choosing the
                        sparsity factor Lambda
  --sparsefact SPARSEFACT
                        Minimum percentage of error to accept after the
                        addition of a reflection in order to include it in the
                        model

"""

import numpy as np

np.random.seed(601)
from utils_base import run_command, eprint, get_git_hash
import warnings
from matplotlib import rc

warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    import argparse

    hostname = run_command('hostname')[0]
    print('Working on host: ' + hostname)
    if hostname == 'ee-cp510':
        air_base_loc = '/media/cp510/ExtraGiaWindows/db_exp_data/Local_Databases/AIR/ACE16/'
        speech_loc_def = '../results_dir/concWavs/media/cp510/ExtraGiaWindows/db_exp_data/' \
                         'Local_Databases/speech/TIMIT/TIMIT/TRAIN/DR1/MEDR0/concWav.wav'
    else:
        air_base_loc = '../Local_Databases/AIR/ACE16/'
        speech_loc_def = '../results_dir/concWavs/concWavs/Local_Databases/speech/TIMIT/TIMIT/TRAIN/DR1' \
                         '/MEDR0/concWav.wav'

    results_dir = '../results_dir/air_modeling_results/'

    parser = argparse.ArgumentParser(description='Arguments for fitting model to AIR')
    parser.add_argument('--array', dest='array', type=str, default='EM32',
                        help='Array to use (EM32 Single Crucif Chromebook Lin8Ch)')
    parser.add_argument('--channel', dest='channel', type=int, default=0,
                        help='Channel to model from the array')
    parser.add_argument('--fs', dest='fs', type=int, default=48000,
                        help='Sampling frequency to work with')
    parser.add_argument('--transition', dest='transition', type=float, default=0.0024,
                        help='Early part duration after max energy sample')
    parser.add_argument('--config', dest='config', type=int, default=1,
                        help='Source receiver config (1, 2)')
    parser.add_argument('--room', dest='room', type=str, default='803',
                        help='Room to model (EE_lobby 508 403a 503 611 502 803)')
    parser.add_argument('--resultsdir', dest='resultsdir', type=str, default=results_dir,
                        help='Directory to save results')
    parser.add_argument('--interactive', dest='interactive', action="store_true",
                        default=False, help='Keep plots at front and pause until closed')
    parser.add_argument('--early', dest='early', action="store_true",
                        default=False, help='Eary Only')
    parser.add_argument('--detailedplots', dest='detailedplots', action="store_true",
                        default=False, help='Make plots at each stage')
    parser.add_argument('--nodisplay', dest='nodisplay', action="store_true",
                        default=False, help='Use the Agg backend for matplotlib')
    parser.add_argument('--matlabinit', dest='matlabinit', action="store_true",
                        default=False, help='Preload a matlab engine')
    parser.add_argument('--tex', dest='tex', action="store_true",
                        default=False, help='Use latex style plotting fonts')
    parser.add_argument('--plotexcitation', dest='plotexci', action="store_true",
                        default=False, help='Plot the excitation signal')
    parser.add_argument('--nocache', dest='nocache', action="store_true",
                        default=False, help='Disable reading from cached results')
    parser.add_argument('--mgm', dest='mgm', action="store_true",
                        default=False, help='Use a Modulated Gaussian Pulse for the excitation')
    parser.add_argument('--delta', dest='delta', action="store_true",
                        default=False, help='Use a delta for the excitation')
    parser.add_argument('--speech', dest='speech', type=str, default=speech_loc_def,
                        help='Speech file to use for the results')
    parser.add_argument('--air', dest='air', type=str, default=None,
                        help='File path of given wav AIR. If this is given the all references to '
                             'ACE are lost. Working with this fixed AIR. For the excitation, '
                             'if no other AIRs are given specifically for the excitation model '
                             'then it will be modeled as a delta.')
    parser.add_argument('--excitationdir', dest='excitationdir', type=str, default=None,
                        help='File path of a folder contatining AIRs as wav files used for '
                             'estimating the excitation signal')
    parser.add_argument('--nparams', dest='nparams', type=int, default=None,
                        help='Shrink model to a fixed number of parameters')
    parser.add_argument('--lassocoef', dest='lassocoef', type=float, default=1 / 10.,
                        help='Fraction of standard deviation of the error to move away from the '
                             'MSE solution of LASSO when choosing the sparsity factor Lambda')
    parser.add_argument('--sparsefact', dest='sparsefact', type=float, default=0.8,
                        help='Minimum percentage of error to accept after the addition of a '
                             'reflection in order to include it in the model')
    args = parser.parse_args()

    nparams = args.nparams
    speech_loc = args.speech
    room = args.room
    array = args.array
    do_channel = args.channel
    model_fs = args.fs
    results_dir = args.resultsdir
    interactive = args.interactive
    args = parser.parse_args()
    early_only = args.early
    transition = args.transition
    config = args.config
    nodisplay = args.nodisplay
    plotexci = args.plotexci
    given_air = args.air
    excitationdir = args.excitationdir
    nomatlabinit = not args.matlabinit
    lassocoef = args.lassocoef
    sparsefact = args.sparsefact
    nocache = args.nocache
    detailedplots = args.detailedplots
    tex = args.tex
    mgm = args.mgm
    delta = args.delta

    print('Inputs: ' + str(args))


else:
    raise AssertionError('Must be run as main')

if tex:
    rc('font', family='serif', serif='Times')
    rc('xtick', labelsize=8)
    rc('ytick', labelsize=8)
    rc('axes', labelsize=8)
    rc('text', usetex='true')

if nodisplay:
    from matplotlib import use

    use('Agg')
import numpy as np
from scipy.io import wavfile

from acenv import AcEnv
from acenvgenmodel_base import plot_matched_excitation, get_pca_excitation, AcEnvGenModelBase
from utils_base import find_all_ft
from utils_spaudio import write_wav
from utils_spaudio import my_resample
from pickle import load, dump

try:
    import os.path
except ImportError:
    raise

cache_dir = '../results_dir/acenvgenmodel_cachedir/git_' + get_git_hash() + '/'

if nomatlabinit:
    eng = None
else:
    try:
        import matlab.engine

        print('Creating Matlab engine')
        eng = matlab.engine.start_matlab()
    except ImportError:
        eng = None

run_command('mkdir -p ' + results_dir + ' ' + cache_dir)

cap_exci_instances = np.inf

suffix = '_noabs'

if np.sum([mgm, excitationdir is not None, delta]) > 1:
    raise AssertionError('Cannot specify both use of MGM or delta and an exictation directory')

if given_air is None:

    excitation_modeling_span = (0.00025, 0.002)
    pca_explained_var_for_exci = 0.98
    # if array == 'Crucif' or array == 'Lin8Ch':
    #     pca_explained_var_for_exci = 0.70

    building_map = {'EE_lobby': 'Building_Lobby', '508': 'Lecture_Room_1', '403a': 'Lecture_Room_2',
                    '503': 'Meeting_Room_1', '611': 'Meeting_Room_2', '502': 'Office_1',
                    '803': 'Office_2'}

    this_receiver = (
        array + '/Building_Lobby/1/' + array + '_EE_lobby_1_RIR.wav',
        array + '/Building_Lobby/2/' + array + '_EE_lobby_2_RIR.wav',
        array + '/Lecture_Room_1/1/' + array + '_508_1_RIR.wav',
        array + '/Lecture_Room_1/2/' + array + '_508_2_RIR.wav',
        array + '/Lecture_Room_2/1/' + array + '_403a_1_RIR.wav',
        array + '/Lecture_Room_2/2/' + array + '_403a_2_RIR.wav',
        array + '/Meeting_Room_1/1/' + array + '_503_1_RIR.wav',
        array + '/Meeting_Room_1/2/' + array + '_503_2_RIR.wav',
        array + '/Meeting_Room_2/1/' + array + '_611_1_RIR.wav',
        array + '/Meeting_Room_2/2/' + array + '_611_2_RIR.wav',
        array + '/Office_1/1/' + array + '_502_1_RIR.wav',
        array + '/Office_1/2/' + array + '_502_2_RIR.wav',
        array + '/Office_2/1/' + array + '_803_1_RIR.wav',
        array + '/Office_2/2/' + array + '_803_2_RIR.wav',
    )

    the_name = array + '_' + room + '_' + str(config) + '_ch' + str(do_channel) + '_RIR' + suffix
    exci_name = array + '_' + '_ch' + str(do_channel) + '_RIR'
    filename = air_base_loc + array + '/' + building_map[room] + '/' + str(config) + '/' + array + \
               '_' + room + '_' + str(config) + '_RIR.wav'
else:
    excitation_modeling_span = (0.001, 0.001)
    pca_explained_var_for_exci = 0.999999999

    if excitationdir is None:
        this_receiver = None
    else:
        this_receiver = \
            find_all_ft(excitationdir, ft='.Wav', use_find=False, find_iname=False) + \
            find_all_ft(excitationdir, ft='.wav', use_find=False, find_iname=False)
        air_base_loc = ''

    the_name = given_air.replace('/', '_').replace('.', '_') + suffix
    exci_name = the_name
    filename = given_air

excitation_name = cache_dir + '/excitation_' + exci_name + '.pkl'

if mgm:
    excitation = None
    issimulation = False
elif this_receiver is None or delta:
    excitation = None
    issimulation = True
else:
    if (not nocache) and os.path.isfile(excitation_name):
        print('Using cache file for excitation ' + excitation_name)
        with open(excitation_name) as pklf:
            excitation, or_dpath = load(pklf)
    else:
        this_receiver = np.sort(this_receiver)
        this_realpath = run_command('realpath ' + filename)[0]
        print('Looking for this AIR in the excitation list as ' + this_realpath)
        this_air_in_excitations = None
        for i, rec_instance in enumerate(this_receiver):
            search_str = air_base_loc + rec_instance
            if this_realpath == run_command('realpath ' + search_str)[0]:
                this_air_in_excitations = i
                print('Found it at ' + str(this_air_in_excitations) + ' as ' + search_str)
                break
        if this_air_in_excitations is None:
            raise AssertionError(
                'Expected AIR to be in the excitation estimation signals, could not '
                'find ' + this_realpath + '. Last checked: ' +
                run_command('realpath ' + air_base_loc + rec_instance)[0])
        if len(this_receiver) < cap_exci_instances:
            keep_list = range(len(this_receiver))
        elif cap_exci_instances > this_air_in_excitations:
            keep_list = range(cap_exci_instances)
        else:
            keep_list = list(range(cap_exci_instances)) + [this_air_in_excitations]
            this_air_in_excitations = len(keep_list) - 1

        air_counter = [0]
        all_air = []
        for i in np.array(this_receiver)[keep_list]:
            new_airs = AcEnv(filename=air_base_loc + i, sampling_freq=model_fs,
                             keep_channel=do_channel,
                             silent_samples_offset=False, matlab_engine=eng).impulse_response.T
            air_counter.append(air_counter[-1] + new_airs.shape[0])
            all_air.append(new_airs)

        air_counter = np.array(air_counter)
        this_air_in_excitations = air_counter[this_air_in_excitations + 0]

        min_shape = np.inf
        for i in all_air:
            if i.shape[1] < min_shape:
                min_shape = i.shape[1]
        for i in range(len(all_air)):
            all_air[i] = all_air[i][:, 0:min_shape]
        all_air = np.concatenate(all_air, axis=0)

        trans_air, or_air = get_pca_excitation(all_air, model_fs,
                                               modeling_span=excitation_modeling_span,
                                               npccomps=pca_explained_var_for_exci, window=False,
                                               take_base_as=this_air_in_excitations)
        excitation = trans_air[this_air_in_excitations, :].flatten()
        or_dpath = or_air[this_air_in_excitations, :].flatten()
        print('Storing excitation ' + excitation_name)
        with open(excitation_name, 'w') as pklf:
            dump([excitation, or_dpath], pklf)

    # excitation /= abs(excitation).max()
    if plotexci:
        saveloc = results_dir + '/' + the_name + '_excitation.pdf'
        plot_matched_excitation(or_dpath, excitation, interactive=interactive, fs=model_fs,
                                savelocation=saveloc)

    issimulation = False

print('Loading acenv from ' + filename)
acenv = AcEnv(is_simulation=issimulation, filename=filename, name=the_name,
              sampling_freq=model_fs, silent_samples_offset=True,
              )

try:
    acenvmodel = AcEnvGenModelBase(
        acenv, do_channel,
        use_matlab=transition > 0.008,
        offset_to_max=True,
        given_excitation=excitation,
        transition_time=transition,
        early_part_only=early_only,
        use_caches=not nocache,
        plot_exci=plotexci and mgm, )
except AssertionError:
    eprint('! Failed for ' + the_name + ' with inputs ' + str(args))
    raise

acenvmodel.early_opt_max_ref = 1
acenvmodel.late_tail_gaussian_std = 1.5
acenvmodel.tail_visible_energy_ratio = 1.0
acenvmodel.early_opt_spars_prom_factor = sparsefact
acenvmodel.lasso_esigma_lambda = lassocoef
acenvmodel.get_model(make_plots=detailedplots)
if nparams is not None:
    if nparams == -1:
        nparams = acenvmodel.air_reconstruct()[1]
    else:
        acenvmodel.reduce_model_to_nparams_dp(nparams)
    acenvmodel.baseline_reports(nparams, do_plots=True, interactive=interactive,
                                savefig_loc=results_dir + '/' + the_name + '_result_modeled_air_baselines.pdf')
score_names, scores = acenvmodel.get_eval_scores(verbose=True)
acenvmodel.plot_modeling_results(
    saveloc=results_dir + '/' + the_name + '_result_modeled_air_hat.pdf', interactive=interactive)

if not early_only:
    rir = acenv.impulse_response[:, do_channel]
    rir_hat = acenvmodel.air_reconstructed_from_model.flatten()
    try:
        fs_speech, s = wavfile.read(speech_loc)
    except IOError as ME:
        print('Could not read speech file ' + speech_loc + ' with: ' + ME.message)
        exit(0)
    s = (s[0:fs_speech * 10].astype('float128') / s[0:fs_speech * 10].max()).astype(float)
    if not model_fs == fs_speech:
        s = my_resample(s, fs_speech, model_fs)
    rev = np.convolve(s, rir)
    rev_hat = np.convolve(s, rir_hat)


    def write_the_wav(filename, x):
        write_wav(filename, model_fs, x)


    write_the_wav(results_dir + '/' + the_name + '_clean.wav', s)
    write_the_wav(results_dir + '/' + the_name + '_rir_hat.wav', rir_hat)
    write_the_wav(results_dir + '/' + the_name + '_rir.wav', rir)
    write_the_wav(results_dir + '/' + the_name + '_rev.wav', rev)
    write_the_wav(results_dir + '/' + the_name + '_rev_hat.wav', rev_hat)
