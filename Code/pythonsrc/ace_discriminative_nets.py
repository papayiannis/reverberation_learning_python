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

This file is the main worker for training and evaluating the DNNs proposed in [1].
The script /Code/pythonsrc/run_ace_discriminative_nets.sh offers usage examples for the experiments
presented in the paper.

This file was original distributed in the repository at:
{repo}

If you use this code in your work, then cite [1].

[1] C. Papayiannis, C. Evers, and P. A. Naylor, "End-to-End Classification of Reverberant Rooms using DNNs," arXiv preprint arXiv:1812.09324, 2018.

"""

from time import time, sleep

import numpy as np

from utils_base import run_command

try:
    from os.path import isdir, exists
except ImportError:
    isdir = None
    exists = None

hostname = run_command('hostname')[0]
fast_test = False and not hostname == 'sapws'
model_basename = 'ace_model'

batch_size_base_per_class_air = 1
batch_size_base_per_class_speech = 1
model_framesize_air = 128
model_framesize_speech = 320
max_speech_len = 5
max_air_len = 0.4
wavform_logpow = False
get_pow_spec = True
start_at_max = True
utt_per_env_def = 20

model_fs = 16000.
sim_data_fs = 16000.

scratchpad_def = '/tmp/ace_models_unsup/'
scratchpad = scratchpad_def


def get_model_air(input_dims, n_outputs, dense_width=32, filters=(4, 8),
                  kernel_size=((3, 3), (3, 3)),
                  strides=((1, 1), (1, 1)),
                  pooling_size=((3, 3), (2, 2)),
                  use_cnn=True, use_rnn=False):
    """
    Constructs models for environment classification based on AIRs.

    Args:
        input_dims: Dimensionality of the input
        n_outputs: Number of output classes
        dense_width: Width of FF layers
        filters: Number of Conv filters
        kernel_size: Kernel size for Conv filters
        strides: Strides of Conv filters
        pooling_size: The pooling size for th Max Poolign layers.
        use_cnn: Enable the use of convolutional layers
        use_rnn: Enable the use of recurrent layers

    Returns:
        A Keras Sequential model

    """
    from keras.models import Sequential
    from keras.layers import Dense, InputLayer, Reshape, Dropout, Conv2D, MaxPooling2D, \
        TimeDistributed, Bidirectional, GRU, Activation

    activation_layer = lambda: Activation('tanh')

    if use_rnn:
        n_recurrent = 2
    else:
        n_recurrent = 0

    print('Generating model with inputs: ' + str(input_dims))

    model = Sequential()
    model.add(InputLayer(input_shape=tuple(list(input_dims))))

    if use_cnn:
        model.add(Reshape((model.output_shape[1], model.output_shape[2], 1)))
        for i, nfilts in enumerate(filters):
            for j in range(2):
                model.add(Conv2D(nfilts, kernel_size[i],
                                 activation='linear', padding='same',
                                 strides=strides[i]))
                model.add(activation_layer())
            model.add(MaxPooling2D(pooling_size[i]))
    else:
        model.add(Reshape((model.output_shape[1], model.output_shape[2])))
        for _, _ in enumerate(filters):
            model.add(TimeDistributed(
                Dense(dense_width, activation='linear', )
            ))
            model.add(activation_layer())

    if n_recurrent > 0:
        model.add(Reshape((-1, np.prod(model.output_shape[2:]).astype(int))))
        model.add(Bidirectional(GRU(dense_width, activation='linear',
                                    return_sequences=True if n_recurrent > 1 else False)))
        model.add(activation_layer())
        for i in range(n_recurrent - 1):
            model.add(GRU(dense_width, activation='linear',
                          return_sequences=True if i < n_recurrent - 2 else False))
            model.add(activation_layer())
        model.add(Reshape((-1,)))
    else:
        model.add(Reshape((-1,)))
        model.add(Dropout(0.1))
        model.add(Dense(dense_width, activation='linear'))
        model.add(activation_layer())
        model.add(Dropout(0.1))
        model.add(Dense(dense_width, activation='linear'))
        model.add(activation_layer())
    model.add(Dense(dense_width, activation='linear'))
    model.add(activation_layer())

    model.add(Dense(n_outputs, activation='softmax'))

    return model


def get_model_speech(input_dims, n_outputs,
                     dense_width=128,
                     filters=(8, 16, 32),
                     kernel_size=[(3, 3)] * 3,
                     strides=[(1, 1)] * 3,
                     pooling_size=[(2, 3)] * 3,
                     use_cnn=False, use_rnn=False,
                     ):
    """
    Constructs models for environment classification based on reverberant speech.

    Args:
        input_dims: Dimensionality of the input
        n_outputs: Number of output classes
        dense_width: Width of FF layers
        filters: Number of Conv filters
        kernel_size: Kernel size for Conv filters
        strides: Strides of Conv filters
        pooling_size: The pooling size for th Max Poolign layers.
        use_cnn: Enable the use of convolutional layers
        use_rnn: Enable the use of recurrent layers

    Returns:
        A Keras Sequential model

    """

    from keras.models import Sequential
    from keras.layers import Dense, InputLayer, Reshape, \
        Dropout, Conv2D, MaxPooling2D, GRU, Bidirectional, TimeDistributed, \
        Activation, BatchNormalization

    activation_layer = lambda: Activation('relu')

    print('Generating model with inputs: ' + str(input_dims))

    if use_rnn:
        n_recurrent = 2
    else:
        n_recurrent = 0

    model = Sequential()
    model.add(InputLayer(input_shape=tuple(list(input_dims))))
    model.add(BatchNormalization())
    if not use_cnn:
        for _, _ in enumerate(filters):
            model.add(TimeDistributed(
                Dense(dense_width, activation='linear', )
            ))
            model.add(activation_layer())
    else:
        model.add(Reshape((model.output_shape[1], model.output_shape[2], 1)))
        for i, nfilts in enumerate(filters):
            for _ in range(2):
                model.add(Conv2D(nfilts, kernel_size[i],
                                 activation='linear', padding='same',
                                 strides=strides[i]))
                model.add(activation_layer())
            model.add(MaxPooling2D(pooling_size[i]))

    if n_recurrent > 0:
        model.add(Reshape((-1, np.prod(model.output_shape[2:]).astype(int))))
        model.add(Bidirectional(GRU(dense_width, activation='linear',
                                    return_sequences=True if n_recurrent > 1 else False)))
        model.add(activation_layer())
        for i in range(n_recurrent - 1):
            model.add(GRU(dense_width, activation='linear',
                          return_sequences=True if i < n_recurrent - 2 else False))
            model.add(activation_layer())
        model.add(Reshape((-1,)))
    else:
        model.add(Reshape((-1,)))
        model.add(Dropout(0.1))
        model.add(Dense(dense_width, activation='linear'))
        model.add(activation_layer())
        model.add(Dropout(0.1))
        model.add(Dense(dense_width, activation='linear'))
        model.add(activation_layer())
    model.add(Dense(dense_width, activation='linear'))
    model.add(activation_layer())

    model.add(Dense(n_outputs, activation='softmax'))

    return model


def show_classification_results(all_preds, y, ids, class_names, fold=None, mark_wrongs=False):
    """
    Prints the results of the classification predictions in a way which allows for a comparison
    between the predictions and the true classes.

    Args:
        all_preds: A matrix of [ N_samples x N_classes ], with 1's on the predicted class
        y: A matrix of [ N_samples x N_classes ], with 1's on the true class
        ids: The id (a string) of each sample
        class_names: The unique classes in the classification problem (as strings)
        fold: The fold in which each sample belongs to (fold of cross validation)
        mark_wrongs: Put a marker next to misclassified samples

    Returns:
        Nothing

    """
    from tabulate import tabulate
    from utils_base import float2str
    accuracy = np.sum(
        np.all(all_preds == y, axis=1)
    ) / float(y.shape[0])
    n_hots = np.sum(all_preds, axis=1)
    if ~np.all(n_hots == 1):
        too_hot = np.where(~(n_hots == 1))[-1]
        raise AssertionError(
            'Predictions do not make sense because the following idxs had more than one hots ' +
            str(too_hot) + ' with the following hots ' + str(n_hots[too_hot]))
    n_hots = np.sum(y, axis=1)
    if ~np.all(n_hots == 1):
        too_hot = np.where(~(n_hots == 1))[-1]
        raise AssertionError(
            'Ground truths do not make sense because the following idxs had more than one hots ' +
            str(too_hot) + ' with the following hots ' + str(n_hots[too_hot]))
    results = np.concatenate((
        np.atleast_2d(ids).T,
        np.atleast_2d(class_names[np.argmax(all_preds, axis=1)]).T
    ), axis=1)
    headers = ('AIR', 'Prediction')
    if fold is not None:
        results = np.concatenate((
            results,
            np.atleast_2d(fold).T
        ), axis=1)
        headers = tuple(list(headers) + ['Fold'])
    if mark_wrongs:
        correct = [i.replace('EE_lobby', 'EE-lobby').split('_')[1] for i in results[:, 0]
                   ] == results[:, 1]
        results = results[~correct, :]
        print('Showing ' + str(np.sum(~correct)) + ' wrongs of ' + str(correct.size))
    print(tabulate(results, headers=headers))

    print('Overall Accuracy: ' + float2str(accuracy, 3))


def create_new_associations(train_associations, utts_per_channel, bs_h5,
                            verbose=True):
    """
    Creates associations between AIRs and speech files, which to not mix the speech data used
    across different rooms. This is useful in the case where new AIRs are added to the training
    (as possibly a result of data augmentation) and you do not want to add more speech samples
    but to strictly reuse the same ones you used in a previous experiment, in order not to
    introduce any extra variability.

    Args:
        train_associations: A pandas DataFrame with the index being the ACE AIR filename (it can
        be full or just the filename), one column called 'speech', which contains the path to
        the wav speech utterance used and one column called 'offsets', which contains the sample
        offset used for the utterance convolution
        utts_per_channel: Number of speech utterances to associate to each new AIR
        bs_h5: The HDF5 dataset location which contains the new AIR information. The dataset has
        2 fields, one is 'filenames', which contains the locations of the wav AIRs and the other is
        'chan', which indicates the number of channels in the audio file
        verbose: Verbose reporting

    Returns:

    """
    from h5py import File
    import numpy as np
    from random import randint
    try:
        from os.path import basename
    except ImportError:
        raise

    hf = File(bs_h5, 'r')
    wav_files = (np.array(hf.get('filenames')).astype(str)).tolist()

    basenames = [thename.split('/')[-1].replace('EE_lobby', 'EE-lobby')
                 for thename in wav_files]
    room = [thename.split('_')[1] for thename in basenames]

    basenames_train = [thename.split('/')[-1].replace('EE_lobby', 'EE-lobby')
                       for thename in train_associations.index]
    room_train = [thename.split('_')[1] for thename in basenames_train]

    new_association = {'airs': ([None] * len(wav_files)), 'speech': [], 'offsets': []}

    if verbose:
        print('Creating speech associations:')
    for i in range(len(wav_files)):
        for _ in range(utts_per_channel):
            same_idxs = np.where(room[i] == np.array(room_train))[-1].tolist()
            new_idx = same_idxs[randint(0, len(same_idxs) - 1)]
            new_association['speech'].append(train_associations['speech'][new_idx])
            new_association['offsets'].append(train_associations['offsets'][new_idx])
            print(basenames[i] + ' -> ' + train_associations.index[new_idx].split('/')[-1] + ',' +
                  train_associations['speech'][
                      new_idx] + '@' + str(
                        train_associations['offsets'][new_idx]))

    return new_association


def train_eval(h5_loc, ace_base, timestamp,
               use_cnn=False, use_rnn=False, bootstrap_h5=None,
               read_cache=True, cacheloc_master='/tmp/',
               speech_dir=None, fold_set=None, epochs=None, test_array=None,
               train_assoc_loc_save=None, bs_assoc_loc_read=None):
    """
    Worker which trains and evaluates DNN solutions for room classification, based on the data
    provided with the ACE challenge database.

    Args:
        h5_loc: Location of HFD5 dataset file for the ACE database, which is provided with this
        repository at Code/results_dir/ace_h5_info.h5. Contains information about the filenames,
        number of channels and also ground truth acoustic parameter values.
        ace_base: The folder containing the ACE wav data.
        timestamp: A timestamp to use for file saving
        use_cnn: Use CNN layers
        use_rnn: Use RNN layers
        bootstrap_h5: An HDF5 dataset file which points to further AIRs to be used additionally
        to the ACE database ones.  The dataset has 2 fields, one is 'filenames', which contains
        the locations of the wav AIRs and the other is 'chan', which indicates the number of
        channels in the audio file.
        read_cache: Enable the reading of any cached data, if any.
        cacheloc_master: Location for saving and reading cached data.
        speech_dir: Location of speech data. Given as a list of [location of train data,
        location of test data].
        fold_set: Folds to run (for cross validation).
        epochs: Epochs for which to train the network.
        test_array: Array to test (Mobile, Curcif, EM32, Chromebook, Linear)
        train_assoc_loc_save: Location to save speech->AIR associations
        bs_assoc_loc_read: Location to pick up ready  speech->AIR association seeds for new AIRs

    Returns: Nothing

    """

    from subprocess import call

    from fe_utils import get_ace_xy
    from pandas import DataFrame as df
    try:
        from os.path import isfile
    except ImportError:
        raise
    from utils_dnntrain import model_trainer, get_scaler_descaler, PostEpochWorker, \
        accuracy_eval, batch_gen, multi_batch_gen

    np.random.seed(601)

    experiment = 'room'
    cacheloc_train = cacheloc_master + '/train_set/'
    cacheloc_bs = cacheloc_master + '/bs_set_%d/'
    cacheloc_test = cacheloc_master + '/train_test/'
    print('Cache location train : ' + cacheloc_train)
    print('Cache location test : ' + cacheloc_test)
    print('Cache location bs : ' + cacheloc_bs)

    call(["mkdir", "-p", scratchpad])
    model_filename = scratchpad + '/' + model_basename + timestamp + '.h5'

    feature_ex_config = {
        'max_air_len': max_speech_len
        if speech_dir is not None else max_air_len,
        'fs': sim_data_fs, 'forced_fs': model_fs,
        'max_speech_read': max_speech_len, 'drop_speech': True,
        'as_hdf5_ds': True,
        'framesize': model_framesize_speech
        if speech_dir is not None else model_framesize_air,
        'keep_ids': None, 'utt_per_env': utt_per_env if speech_dir is not None else 1,
        'write_cached_latest': True, 'wavform_logpow': wavform_logpow,
        'read_cached_latest': read_cache, 'get_pow_spec': get_pow_spec,
        'start_at_max': start_at_max if speech_dir is None else False,
    }

    if train_assoc_loc_save is not None and bs_assoc_loc_read is not None and not (
            train_assoc_loc_save == bs_assoc_loc_read):
        raise AssertionError(
            'The save location and read locations for the speech associations you gave do not match'
            ', this does not make sense')
    associations_loc = (scratchpad + '/associations_train_' + timestamp + '.csv'
                        if train_assoc_loc_save is None
                        else train_assoc_loc_save)
    if train_assoc_loc_save is not None:
        print('Using given association location for train : ' + train_assoc_loc_save)
    (x_out_train, (y_train, y_position_train)), ids_train, class_names_train, \
    (_, _, _), \
    ((group_names_array_train, group_names_position_train, group_names_room_train),
     (groups_array_train, groups_position_train, group_room_train)
     ) = get_ace_xy(h5_file=h5_loc, ace_base=ace_base,
                    scratchpad=scratchpad + '/train/',
                    speech_files=speech_dir[
                        0] if speech_dir is not None else None,
                    group_by=('array', 'position', 'room'),
                    cacheloc=cacheloc_train,
                    y_type=(experiment, 'position'),
                    copy_associations_to=associations_loc,
                    **feature_ex_config)
    if speech_dir is not None:
        if not isfile(associations_loc):
            print(
                '! The association of airs to speech did not happen now probably because you read a '
                'cache so i will not bother with this')
            do_associations = False
            train_associations = None
            if train_assoc_loc_save is not None:
                print('I could not save the association file for '
                      'the train speech where you told me to : '
                      + train_assoc_loc_save)
        else:
            print('! I will make sure that the bootstrap AIR speech associations are linked to the '
                  'train ones')
            train_associations = df.from_csv(associations_loc)
            do_associations = True
        if bs_assoc_loc_read is not None:
            print('! Overwriting the bootstrap AIR speech associations from train (if any) '
                  'and using the given ones')
            train_associations = df.from_csv(bs_assoc_loc_read)
            do_associations = True
    else:
        do_associations = False
        train_associations = None

    groups_array_train = [groups_array_train[ii] for ii in group_names_array_train.argsort()]
    group_names_array_train = group_names_array_train[group_names_array_train.argsort()]
    groups_position_train = [groups_position_train[ii] for ii in
                             group_names_position_train.argsort()]
    group_names_position_train = group_names_position_train[group_names_position_train.argsort()]
    if bootstrap_h5 is not None:
        x_bs = [None for _ in range(len(bootstrap_h5))]
        y_bs = [None for _ in range(len(bootstrap_h5))]
        for i in range(len(bootstrap_h5)):
            (x_bs[i], y_bs[i]), _, _, \
            (_, _, _), \
            (_, _) = get_ace_xy(h5_file=bootstrap_h5[i], ace_base='',
                                scratchpad=scratchpad + '/bs/',
                                speech_files=speech_dir[
                                    0] if speech_dir is not None else None,
                                group_by=None, cacheloc=cacheloc_bs % i,
                                given_associations=
                                create_new_associations(
                                    train_associations,
                                    utt_per_env,
                                    bootstrap_h5[i]) if do_associations else None,
                                y_type=experiment, **feature_ex_config)
        print('Got ' + str([
            x_bs[i].shape[0] for i in
            range(len(x_bs))]) + ' bootstrap samples with y columns ' + str(
            [y_bs[i].shape[1] for i in range(len(y_bs))]))
    else:
        (x_bs, y_bs) = (None, None)

    (x_out_test, y_test), ids_test, class_names_test, \
    (_, _, _), \
    ((group_names_array_test, group_names_position_test),
     (groups_array_test, groups_position_test)
     ) = get_ace_xy(h5_file=h5_loc, ace_base=ace_base,
                    scratchpad=scratchpad + '/test/',
                    speech_files=speech_dir[
                        1] if speech_dir is not None else None,
                    group_by=('array', 'position'), cacheloc=cacheloc_test,
                    y_type=experiment, **feature_ex_config)
    groups_array_test = [groups_array_test[ii] for ii in group_names_array_test.argsort()]
    group_names_array_train = group_names_array_train[group_names_array_train.argsort()]
    groups_position_test = [groups_position_test[ii] for ii in
                            group_names_position_test.argsort()]
    group_names_position_test = group_names_position_test[group_names_position_test.argsort()]
    if not group_names_array_test.size == group_names_array_train.size:
        raise AssertionError('Test and train sets do not match')
    if ~np.all(group_names_array_train == group_names_array_test):
        raise AssertionError('Test and train sets do not match')
    if not group_names_position_test.size == group_names_position_train.size:
        raise AssertionError('Test and train sets do not match')
    if ~np.all(group_names_position_test == group_names_position_train):
        raise AssertionError('Test and train sets do not match')

    # else:
    #     (x_out_test, y_test) = (x_out_train, y_train)
    #     (ids_test, class_names_test) = (ids_train, class_names_train)

    array_idx_train = None
    array_idx_test = None
    if test_array is not None:
        array_idx_train = group_names_array_train.tolist().index(test_array)
        array_idx_test = group_names_array_test.tolist().index(test_array)

    if y_bs is not None:
        batch_size_base_per_class_bs = int(np.round(
            batch_size_base_per_class * float(
                y_position_train.shape[1]) / float(y_bs[0].shape[1])) / float(len(y_bs)))
        if batch_size_base_per_class_bs < 1:
            raise AssertionError('Unexpected scenario')
        print('Batches will include : *Train* ' + str(
            batch_size_base_per_class) + ' samples from each of the ' + str(
            y_position_train.shape[1]) + ' classes and *Test* ' + str(
            batch_size_base_per_class_bs) + ' samples from each of the ' + str(y_bs[0].shape[1]))
    else:
        batch_size_base_per_class_bs = None

    if speech_dir is None:
        scaler, _ = get_scaler_descaler(x_out_train)
        x_out_train = scaler(x_out_train)
        x_out_test = scaler(x_out_test)

    if epochs is None:
        epochs = 1000 if experiment == 'room' else 50

    callback_eval_func = accuracy_eval

    all_preds = []
    all_folds = []

    from keras import backend as K
    from keras.models import load_model

    if fold_set is not None:
        if test_array is not None:
            raise AssertionError(
                'You cannot define a test array and a test fild set on the same run')
        experiment_group_names = group_names_position_train[fold_set]
        new_groups = []
        for i in fold_set:
            new_groups.append(groups_position_train[i])
        experiment_groups_train = new_groups
        new_groups = []
        for i in fold_set:
            new_groups.append(groups_position_test[i])
        experiment_groups_test = new_groups
    elif test_array is not None:
        experiment_groups_train = [
            np.setxor1d(range(len(ids_train)), groups_array_train[array_idx_train])]
        experiment_groups_test = [groups_array_test[array_idx_test]]
        experiment_group_names = [test_array]
        fold_set = [0]
    else:
        fold_set = range(len(groups_position_test))
        experiment_groups_train = groups_position_train
        experiment_groups_test = groups_position_test
        experiment_group_names = group_names_position_train

    print('Train set : ')
    for i in experiment_groups_train[0]:
        print('Train : ' + ids_train[i])
    print('Test set : ')
    for i in experiment_groups_test[0]:
        print('Test : ' + ids_test[i])

    for i, this_outgroup in enumerate(experiment_groups_train):
        eval_idxs = experiment_groups_test[i]

        val_idxs = []
        for i_g in range(len(groups_position_train)):
            possible_choices = [j for j in groups_position_train[i_g].tolist() if
                                j not in this_outgroup.tolist()]
            if len(possible_choices) > 0:
                val_idxs += np.random.choice(
                    possible_choices, int(.15 / len(groups_position_train) * x_out_train.shape[0]),
                    replace=False).tolist()
        val_idxs = np.sort(val_idxs).tolist()

        train_idxs = np.array([j for j in range(x_out_train.shape[0]) if j not in
                               (this_outgroup.tolist() + val_idxs)]).astype(int)
        print('For fold (zero counting absolute) ' + str(fold_set[i]) + ' (at step: ' + str(i + 1) +
              ' of ' +
              str(len(fold_set)) + '), at ' + experiment_group_names[i]
              + ' with ' + str(train_idxs.size) + ' train '
              + ' and ' + str(eval_idxs.size) + ' test samples')
        if x_bs is not None:
            print('Including ' + str(x_bs[0].shape[0]) + ' bootstrap data augmentation samples')
        # print('Validation samples (' + str(len(val_idxs)) + ')' + str(
        #     [ids_train[id_idx] for id_idx in val_idxs]))
        # continue

        evaluation_eval_func = lambda cmodel: callback_eval_func(x_out_test[eval_idxs, :, :],
                                                                 y_test[eval_idxs, :], cmodel,
                                                                 prefix='Test')

        model_filename_fold = model_filename.replace('.h5', '_fold_' + str(i) + '.h5')
        if exists(model_filename_fold):
            print('Loading pre-traiend model from ' + model_filename_fold)
            model = load_model(model_filename_fold)
            print('Loaded pre-traiend model from ' + model_filename_fold)
            model.summary()
        else:
            print('File ' + model_filename_fold + ' does not exist so i will train the model')
            callbacks = [PostEpochWorker(
                (x_out_train[val_idxs, :, :],
                 x_out_test[eval_idxs, :, :]),
                (y_train[val_idxs, :], y_test[eval_idxs, :]),
                model_filename,
                eval_fun=(
                    lambda x, y, cmodel: callback_eval_func(x, y, cmodel, prefix='Val'),
                    lambda x, y, cmodel: callback_eval_func(x, y, cmodel, prefix='Test')),
                eval_every_n_epochs=100),
            ]

            if bootstrap_h5 is None:
                effective_batch_size = batch_size_base_per_class * y_position_train.shape[1]
                this_batch_gen = batch_gen(
                    x_out_train, y_train, batch_size_base_per_class,
                    y_to_balance=y_position_train, sub_idxs=train_idxs)
                steps_per_epoch = int(round(y_train.shape[0] / effective_batch_size))
            else:
                effective_batch_size = (batch_size_base_per_class * y_position_train.shape[1] +
                                        np.sum(
                                            [y_bs[ii].shape[1] * batch_size_base_per_class_bs for ii
                                             in range(len(y_bs))]))
                steps_per_epoch = int(round(np.sum(
                    [y_train.shape[0]] + [ii.shape[0] for ii in y_bs]) / effective_batch_size))
                this_batch_gen = multi_batch_gen(
                    [x_out_train] + x_bs, [y_train] + y_bs,
                    [batch_size_base_per_class] +
                    [batch_size_base_per_class_bs
                     for _ in
                     range(len(y_bs))],
                    y_to_balance=[y_position_train] + [None] * len(x_bs),
                    sub_idxs=[train_idxs] + [None] * len(x_bs))
            print('For a batch size of ' + str(effective_batch_size) + ' i will do ' +
                  str(steps_per_epoch) + ' steps per epoch.')
            this_batch_gen_val = batch_gen(
                x_out_train, y_train, batch_size_base_per_class,
                y_to_balance=y_position_train, sub_idxs=val_idxs)
            _, filename = model_trainer(
                this_batch_gen, x_out_train.shape[1:], y_train.shape[1],
                get_model_air if speech_dir is None else get_model_speech,
                tensorlog=True, callbacks=callbacks, epochs=epochs,
                loss_patience=10, scratchpad=scratchpad,
                model_filename=model_filename_fold,
                print_summary=i == 0, use_cnn=use_cnn, use_rnn=use_rnn,
                val_gen=this_batch_gen_val, val_patience=15, steps_per_epoch=steps_per_epoch,
            )
            model = callbacks[0].best_val_model

        new_preds = evaluation_eval_func(model)
        all_preds.append(new_preds)
        all_folds.append(np.zeros(new_preds.shape[0], dtype=int) + fold_set[i])
        idxs_to_do = np.concatenate(experiment_groups_test[0:i + 1]).astype(int)
        show_classification_results(
            np.concatenate(all_preds, axis=0)[np.argsort(idxs_to_do), ...],
            y_test[np.sort(idxs_to_do), ...],
            ids_test[np.sort(idxs_to_do).tolist()],
            class_names_test, mark_wrongs=True,
            fold=np.concatenate(all_folds)[np.argsort(idxs_to_do)]
        )
        K.clear_session()
    print('Overall Test results for all folds:')
    idxs_to_do = np.concatenate(experiment_groups_test).astype(int)
    show_classification_results(
        np.concatenate(all_preds, axis=0)[np.argsort(idxs_to_do), ...],
        y_test[np.sort(idxs_to_do), ...],
        ids_test[np.sort(idxs_to_do).tolist()],
        class_names_test,
        fold=np.concatenate(all_folds)[np.argsort(idxs_to_do)]
    )


if __name__ == '__main__':
    """
    This file is the main worker for training and evaluating the DNNs proposed in:
    C. Papayiannis, C. Evers, and P. A. Naylor, "End-to-End Classification of Reverberant Rooms using DNNs," arXiv preprint arXiv:1812.09324, 2018.
    
    usage: ace_discriminative_nets.py [-h] [--ace ACE_BASE] [--h5 H5]
                                  [--bsh5 [BSH5 [BSH5 ...]]]
                                  [--experiment EXPERIMENT]
                                  [--speech [SPEECH [SPEECH ...]]]
                                  [--readcache] [--saveloc SCRATCHPAD]
                                  [--testarray TESTARRAY] [--resume]
                                  [--fold [FOLD [FOLD ...]]] [--epochs EPOCHS]
                                  [--utts UTT_PER_ENV] [--cnn] [--rnn]
                                  [--cacheloc CACHELOC]
                                  [--assoclocsave ASSOCLOCSAVE]
                                  [--assoclocread ASSOCLOCREAD]

    Arguments for training of ACE models and encodings
    
    optional arguments:
      -h, --help            show this help message and exit
      --ace ACE_BASE        Location of the ACE database
      --h5 H5               Location of HFD5 dataset file for the ACE database,
                            which is provided with this repository at
                            Code/results_dir/ace_h5_info.h5. Contains information
                            about the filenames, number of channels and also
                            ground truth acoustic parameter values. If you want to
                            create a new one, then use fe_utils.compile_ace_h5
      --bsh5 [BSH5 [BSH5 ...]]
                            Location of database as h5 for bootstrap training data
      --experiment EXPERIMENT
                            Experiment type (room,type,position)
      --speech [SPEECH [SPEECH ...]]
                            Locations where the wav files
      --readcache           Do not load new data, just read the last cached data
      --saveloc SCRATCHPAD  Location to save results
      --testarray TESTARRAY
                            Array to exclude from training and use as test (
                            Chromebook, Mobile, EM32, Lin8Ch, Crucif )
      --resume              Treat the --saveloc as the absolute results directory
                            and not recreate models which already exist in the
                            path
      --fold [FOLD [FOLD ...]]
                            Do only this cross-validation fold
      --epochs EPOCHS       Number of epochs
      --utts UTT_PER_ENV    Number of speech utterances per AIR
      --cnn                 Add CNN layers to the net
      --rnn                 Add RNN layers to the net
      --cacheloc CACHELOC   Locations where the cache is located and/or will be
                            stored
      --assoclocsave ASSOCLOCSAVE
                            Locations to save the association file for the train
                            set
      --assoclocread ASSOCLOCREAD
                            Locations to read the association file for the
                            bootstrap set

    
    """

    import argparse

    parser = argparse.ArgumentParser(
        description='Arguments for training or ACE models and encodings')
    parser.add_argument('--ace', dest='ace_base', type=str,
                        default='../Local_Databases/AIR/ACE16/',
                        help='Location of the ACE database')
    parser.add_argument('--h5', dest='h5', type=str, default='../results_dir/ace_h5_info.h5',
                        help='Location of HFD5 dataset file for the ACE database, which is '
                             'provided with this repository at Code/results_dir/ace_h5_info.h5. '
                             'Contains information about the filenames, number of channels and '
                             'also ground truth acoustic parameter values. If you want to create '
                             'a new one, then use fe_utils.compile_ace_h5')
    parser.add_argument('--bsh5', dest='bsh5', type=str, default=None, nargs='*',
                        help='Location of database as h5 for bootstrap traniing data')
    parser.add_argument('--experiment', dest='experiment', type=str, default='room',
                        help='Experiment type (room,type,position)')
    parser.add_argument('--speech', dest='speech', type=str, default=None, nargs='*',
                        help='Locations where the wav files')
    parser.add_argument('--readcache', dest='readcache', action="store_true",
                        default=False,
                        help='Do not load new data, just read the last cached data')
    parser.add_argument('--saveloc', dest='scratchpad', type=str, default=scratchpad_def,
                        help='Location to save results')
    parser.add_argument('--testarray', dest='testarray', type=str, default=None,
                        help='Array to exclude from training and use as test '
                             '( Chromebook, Mobile, EM32, Lin8Ch, Crucif )')
    parser.add_argument('--resume', dest='resume', action="store_true",
                        default=False, help='Treat the --saveloc as the absolute results'
                                            ' directory and not recreate models which already exist'
                                            ' in the path')
    parser.add_argument('--fold', dest='fold', type=int, default=None, nargs='*',
                        help='Do only this cross-validation fold')
    parser.add_argument('--epochs', dest='epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--utts', dest='utt_per_env', type=int, default=utt_per_env_def,
                        help='Number of speech utterances per AIR')
    parser.add_argument('--cnn', dest='cnn', action="store_true",
                        default=False, help='Add CNN layers to the net')
    parser.add_argument('--rnn', dest='rnn', action="store_true",
                        default=False, help='Add RNN layers to the net')
    parser.add_argument('--cacheloc', dest='cacheloc', type=str, default='/tmp/',
                        help='Locations where the cache is located and/or will be stored')
    parser.add_argument('--assoclocsave', dest='assoclocsave', type=str, default=None,
                        help='Locations to save the association file for the train set')
    parser.add_argument('--assoclocread', dest='assoclocread', type=str, default=None,
                        help='Locations to read the association file for the bootstrap set')

    args = parser.parse_args()

    timestamp = str(time()) + '_' + str(np.random.rand(1)[0])

    experiment = args.experiment
    ace_base = args.ace_base
    h5 = args.h5
    speech = args.speech
    utt_per_env = args.utt_per_env if args.speech is not None else 1
    scratchpad = args.scratchpad
    if not args.resume:
        new_scratchpad = args.scratchpad + '/run_' + timestamp + '/'
        while isdir(new_scratchpad):
            sleep(np.random.random(1)[0])
            timestamp = str(time())
            new_scratchpad = args.scratchpad + '/run_' + timestamp + '/'
        scratchpad = new_scratchpad
    else:
        scratchpad = args.scratchpad
        timestamp = [idx.split('_')[1] for idx in scratchpad.split('/') if 'run' in idx][0]
        print('You wanted to resume so i will write everything at : ' + scratchpad)
        print('I think the timestamp is ' + timestamp)

    batch_size_base_per_class = (
        batch_size_base_per_class_air if speech is None else batch_size_base_per_class_speech)

    run_command('mkdir -p ' + scratchpad)

    train_eval(h5, ace_base, timestamp, speech_dir=speech, read_cache=args.readcache,
               use_cnn=args.cnn, use_rnn=args.rnn,
               cacheloc_master=args.cacheloc, bootstrap_h5=args.bsh5,
               fold_set=args.fold, epochs=args.epochs, test_array=args.testarray,
               train_assoc_loc_save=args.assoclocsave, bs_assoc_loc_read=args.assoclocread)
