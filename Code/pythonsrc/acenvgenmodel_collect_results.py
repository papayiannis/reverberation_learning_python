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

This file collects the modeling results for AIRs, so that they can be picked up by the GAN
modeling routines and estimate generative models for AIRs.

It uses the log files collected using 'ac_gm_early_tester.sh' and this
file is called from the script 'my_gan_model_worker.sh' (provided with this repo).

More information below.

This file was original distributed in the repository at:
{repo}

If you use this code in your work, then cite:
C. Papayiannis, C. Evers, and P. A. Naylor, "Data Augmentation of Room Classifiers using Generative Adversarial Networks," arXiv preprint arXiv:1901.03257, 2018.

"""


def run_extractor(file_loc, dump_dir):
    """
    The main extraction routine for an individual log file collected using
    'ac_gm_early_tester.sh'. It collects the reflection information and the acoustic parameter
    information in a tuple.

    Args:
        file_loc: The location of the log file as '<some_dir>/<AIR_base-filename>/log.txt'
        dump_dir: Location to use as a scratchpad

    Returns: (
        (
            ,Reflection info DataFrame with fields:
                TOA, Amplitude,
            ,( [list of names of further parameters], [list of numpy arrays for further parameters])
        ),
        AIR_name )

    """
    from os import system
    try:
        from os.path import dirname, basename
    except ImportError:
        raise
    import pandas as pd
    import numpy as np

    if dump_dir is None:
        outfile = file_loc.replace('.txt', '_ref_report.txt')
    else:
        from utils_base import run_command
        run_command('mkdir -p ' + dump_dir)
        outfile = dump_dir + '/' + file_loc.split('/')[-2] + '.txt'

    system(
        'first="Reflections";second="Saved";'
        ' cat ' + file_loc +
        '  | sed \'s:ToA:idx,ToA:g\'| sed \'s:TOA (r):TOAr:g\''
        '  | sed -n "/$first/, /$second/ p" | tail -n +2 | head -n -2|sed \'s/^[[:space:]]*//\'| '
        'sed \'s/  */ /g\' | tr " " ","  |cut -d"," -f2- |grep -v -- "--"> ' + outfile
    )

    try:
        reflections_df = pd.read_csv(outfile)
    except pd.errors.EmptyDataError:
        print('Could not process file ' + file_loc)
        return None, None

    reflections_df = reflections_df[
        ['TOAr', 'Amplitude']]

    system(' grep "AR coeff:" ' + file_loc + ' | sed "s/AR coeff:  //g"|tr -d " " >' + outfile)
    ar_coeff = np.genfromtxt(outfile, delimiter=',')
    system(' grep "MA coeff:" ' + file_loc + ' | sed "s/MA coeff:  //g"|tr -d " " >' + outfile)
    ma_coeff = np.genfromtxt(outfile, delimiter=',')
    system(
        ' grep "Excitation :" ' + file_loc + ' | sed "s/Excitation :  //g"|tr -d " " >' + outfile)
    excitation = np.genfromtxt(outfile, delimiter=',')
    system(
        'grep "T60" ' + file_loc + ' |  sed \'s/__init__ : T60 estimated as : //g\' | '
                                   'tr -d "s">' + outfile)
    t60 = np.atleast_1d(np.genfromtxt(outfile))
    system(
        'grep "DRR estimated" ' + file_loc + ' |  sed \'s/__init__ : DRR estimated as : //g\' '
                                             '| tr -d "dB">' + outfile)
    drr_early = np.atleast_1d(np.genfromtxt(outfile))
    system(
        'grep "DRR early" ' + file_loc + ' |  sed \'s/__init__ : DRR early estimated as : //g\' '
                                         '| tr -d "dB">' + outfile)
    drr_late = np.atleast_1d(np.genfromtxt(outfile))
    system(
        'grep "DRR late" ' + file_loc + ' |  sed \'s/__init__ : DRR late estimated as : //g\' '
                                        '| tr -d "dB">' + outfile)
    drr = np.atleast_1d(np.genfromtxt(outfile))

    vector_params = (np.atleast_2d(np.concatenate((t60, drr, drr_early, drr_late,))),
                     np.atleast_2d(ar_coeff),
                     np.atleast_2d(ma_coeff),
                     np.atleast_2d(excitation))

    system('rm ' + outfile)

    return (reflections_df,
            (['AcousticParams', 'AR', 'MA', 'Excitation'], tuple(vector_params))
            ), \
           basename(dirname(file_loc))


def collector_wrapper(file_loc_list, dump_dir):
    """

    The main collection worker. Given a set of files, which are the logs of modeling AIRs
    collected using 'ac_gm_early_tester.sh', this file extracts the information regarding the
    reflection modeling and the channel-wide parameters and creates and HDF5 dataset which
    contains the information for all responses;. The intention is to use this file and later
    train a GAN model, estimating the generative model of the process.

    Args:
        file_loc_list: a list with the file-names to the logs as
        '<some_dir>/<AIR_base-filename>/log.txt'
        max_bounds: Maximum number of material-types to consider
        dump_dir: Location to use as scratchpad

    Returns: Nothing

    """
    from h5py import File
    import numpy as np

    make_name_numerical = False

    dfs = []
    v_params = None
    v_param_names = None
    names = []

    try:
        for this_file_loc in file_loc_list:
            print('Working on ' + this_file_loc)
            (ds_tmp, (v_param_names, v_params_tmp,)), name_tmp = run_extractor(this_file_loc,
                                                                               dump_dir)
            if ds_tmp is None or name_tmp is None:
                continue
            if make_name_numerical:
                name_tmp = int(name_tmp.replace('air_', '').replace('.wav', ''))
            dfs.append(ds_tmp.as_matrix().astype(float))
            if v_params is None:
                v_params = [[] for _ in range(len(v_params_tmp))]
            for sub_i in range(len(v_params_tmp)):
                v_params[sub_i].append(v_params_tmp[sub_i])
            names.append(name_tmp)
    except KeyboardInterrupt:
        print('You want to stop collecting...')
        if len(dfs) > len(names):
            dfs = dfs[0:-1]

    print('Preparing matrices for saving h5 file')
    dfs_mats = []
    for i in range(len(names)):
        dfs_mats.append(dfs[i])
    max_len = 0
    for i in range(len(names)):
        max_len = max(max_len, dfs_mats[i].shape[0])
    max_len += 1
    for i in range(len(names)):
        npads = max_len - dfs_mats[i].shape[0]
        if npads > 0:
            dfs_mats[i] = np.concatenate((
                np.concatenate(
                    (np.zeros((npads, dfs_mats[i].shape[1]), dtype=dfs_mats[i].dtype),
                     np.flipud(dfs_mats[i]))),
                np.zeros((max_len, 1), dtype=dfs_mats[i].dtype)), axis=1)
        dfs_mats[i][npads - 1, -1] = 1
    dfs_mats = np.stack(dfs_mats)
    for i in range(len(v_param_names)):
        v_params[i] = np.concatenate(v_params[i], axis=0)

    h5_loc = dump_dir + '/reflection_y_data.h5'
    print('Saving all results to ' + h5_loc)
    hf = File(h5_loc, 'w')
    hf.create_dataset('names', data=names)
    hf.create_dataset('y', data=dfs_mats)
    for i in range(len(v_param_names)):
        hf.create_dataset(v_param_names[i], data=v_params[i])
    hf.close()
    msg = ('Wrote: ' + str(h5_loc))
    # msg += ' for '
    # for i in names:
    #     msg = msg + str(i) + ' '
    print(msg)


if __name__ == "__main__":
    """
    
    This file collects the modeling results for AIRs, so that they can be picked up by the GAN 
    modeling routines and estimate generative models for AIRs.
    
    
    It uses the log files collected using 'ac_gm_early_tester.sh' and this
    file is called from the script 'gan_model_worker.sh' (provided with this repo).
    
    Usage:
    
    Arguments for processing of results from AIR modeling

    positional arguments:
      file_loc              Location where log.txt files are located
    
    optional arguments:
      -h, --help            show this help message and exit
      --maxbounds MAXBOUNDS
                            Maximum number of bounds that can be considered. this
                            will be used to fix the columns of the csv files
      --saveloc SAVELOC     Location to save the resulting h5 file
    
    """

    import argparse

    parser = argparse.ArgumentParser(
        description='Arguments for processing of results from AIR modeling')
    parser.add_argument('--maxbounds', dest='maxbounds', type=int, default=10,
                        help='Maximum number of bounds that can be considered. '
                             'this will be used to fix the '
                             'columns of the csv files')
    parser.add_argument('file_loc', type=str, nargs='*',
                        help='Location where log.txt files are located')
    parser.add_argument('--saveloc', type=str, default='/tmp/',
                        help='Location to save the resulting h5 file')

    args = parser.parse_args()

    file_loc = args.file_loc
    if len(args.file_loc) == 1:
        if '.list' in args.file_loc[0]:
            with open(args.file_loc[0]) as f:
                lines = f.read().splitlines()
            file_loc = lines

    print('Will process ' + str(len(file_loc)) + ' files')

    maxbounds = args.maxbounds

    collector_wrapper(file_loc, args.saveloc)
