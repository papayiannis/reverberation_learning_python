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

[1] C. Papayiannis, C. Evers and P. A. Naylor,
"End-to-End Classification of Reverberant Rooms Using DNNs,"
in IEEE/ACM Transactions on Audio, Speech, and Language Processing,
vol. 28, pp. 3010-3017, 2020, doi: 10.1109/TASLP.2020.3033628.

"""

import argparse
from os.path import exists
from subprocess import call
from time import time

import numpy as np
from keras import backend as K
from keras.layers import Dense, InputLayer, Reshape, \
    Dropout, Conv2D, MaxPooling2D, GRU, Bidirectional, TimeDistributed, \
    Activation, BatchNormalization
from keras.layers import Layer
from keras.models import Sequential
from keras.models import load_model
from tabulate import tabulate

from fe_utils import get_ace_xy
from utils_base import float2str
from utils_base import run_command
from utils_dnntrain import model_trainer, get_scaler_descaler, PostEpochWorker, \
    accuracy_eval, batch_gen

_TIMESTAMP = str(time()) + '_' + str(np.random.rand(1)[0])
HOSTNAME = run_command('hostname')[0]
FAST_TEST = False and not HOSTNAME == 'sapws'
MODEL_BASENAME = 'ace_model'

MAX_EPOCHS = 50  # 50
UTT_PER_ENV_DEF = 20  # 20
MAX_STEPS_PER_EPOCH = 1e100  # 1e100

BATCH_SIZE_BASE_PER_CLASS_AIR = 1
BATCH_SIZE_BASE_PER_CLASS_SPEECH = 1
MODEL_FRAMESIZE_SPEECH = 320
MAX_SPEECH_LEN = 5
WAVFORM_LOGPOW = False
GET_POW_SPEC = True
START_AT_MAX = True

MODEL_FS = 16000.
SIM_DATA_FS = 16000.

SCRATCHPAD_DEF = '/tmp/ace_models_unsup/'
SCRATCHPAD = SCRATCHPAD_DEF


class Attention(Layer):
    # https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(Attention, self).get_config()


def get_model_speech(input_dims, n_outputs, dense_width=128, filters=(8, 16, 32), kernel_size=((3, 3),) * 3,
                     strides=((1, 1),) * 3, pooling_size=((2, 3),) * 3, use_cnn=False, use_rnn=False,
                     use_attention=False):
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
        use_attention: Enable Attention

    Returns:
        A Keras Sequential model

    """

    activation_layer = lambda: Activation('relu')

    print(f'Generating model with inputs: {input_dims}')

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
                                    return_sequences=True if n_recurrent > 1 else use_attention)))
        model.add(activation_layer())
        for i in range(n_recurrent - 1):
            model.add(GRU(dense_width, activation='linear',
                          return_sequences=True if i < n_recurrent - 2 else use_attention))
        if use_attention:
            model.add(Attention())

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
        print(f'Showing {(np.sum(~correct))} wrongs of ' + str(correct.size))
    print(tabulate(results, headers=headers))

    print(f'Overall Accuracy: {float2str(accuracy, 3)}')


def train_eval(h5_loc, ace_base, timestamp,
               use_cnn=False, use_rnn=False,
               read_cache=True, cacheloc_master='/tmp/', split_type='position',
               speech_dir=None, use_attention=False):
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
        use_attention: Use Attention mechanism layers
        to the ACE database ones.  The dataset has 2 fields, one is 'filenames', which contains
        the locations of the wav AIRs and the other is 'chan', which indicates the number of
        channels in the audio file.
        read_cache: Enable the reading of any cached data, if any.
        cacheloc_master: Location for saving and reading cached data.
        split_type: Choice between array and position. The cross validation folds
        speech_dir: Location of speech data. Given as a list of [location of train data,
        location of test data].

    Returns: Nothing

    """

    np.random.seed(601)

    experiment = 'room'
    cacheloc_train = cacheloc_master + '/train_set/'
    cacheloc_bs = cacheloc_master + '/bs_set_%d/'
    cacheloc_test = cacheloc_master + '/train_test/'
    print(f'Cache location train :  {cacheloc_train}')
    print(f'Cache location test :  {cacheloc_test}')
    print(f'Cache location bs :   {cacheloc_bs}')

    call(["mkdir", "-p", SCRATCHPAD])
    model_filename = f'{SCRATCHPAD}/{MODEL_BASENAME}_{timestamp}.h5'

    feature_ex_config = {
        'max_air_len': MAX_SPEECH_LEN,
        'fs': SIM_DATA_FS, 'forced_fs': MODEL_FS,
        'max_speech_read': MAX_SPEECH_LEN, 'drop_speech': True,
        'as_hdf5_ds': True,
        'framesize': MODEL_FRAMESIZE_SPEECH,
        'keep_ids': None, 'utt_per_env': UTT_PER_ENV_DEF,
        'write_cached_latest': True, 'wavform_logpow': WAVFORM_LOGPOW,
        'read_cached_latest': read_cache, 'get_pow_spec': GET_POW_SPEC,
        'start_at_max': False,
    }

    (x_out_train, (y_train, y_position_train)), ids_train, class_names_train, \
    (_, _, _), \
    ((group_names_array_train, group_names_position_train, group_names_room_train),
     (groups_array_train, groups_position_train, group_room_train)
     ) = get_ace_xy(h5_file=h5_loc, ace_base=ace_base,
                    scratchpad=SCRATCHPAD + '/train/',
                    speech_files=speech_dir[0],
                    group_by=('array', 'position', 'room'),
                    cacheloc=cacheloc_train,
                    y_type=(experiment, 'position'),
                    **feature_ex_config)

    groups_array_train = [groups_array_train[ii] for ii in group_names_array_train.argsort()]
    group_names_array_train = group_names_array_train[group_names_array_train.argsort()]
    groups_position_train = [groups_position_train[ii] for ii in
                             group_names_position_train.argsort()]
    group_names_position_train = group_names_position_train[group_names_position_train.argsort()]

    (x_out_test, y_test), ids_test, class_names_test, \
    (_, _, _), \
    ((group_names_array_test, group_names_position_test),
     (groups_array_test, groups_position_test)
     ) = get_ace_xy(h5_file=h5_loc, ace_base=ace_base,
                    scratchpad=SCRATCHPAD + '/test/',
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

    scaler, _ = get_scaler_descaler(x_out_train)
    x_out_train = scaler(x_out_train)
    x_out_test = scaler(x_out_test)
    callback_eval_func = accuracy_eval
    all_preds = []
    all_folds = []

    if split_type == 'position':
        fold_set = range(len(groups_position_test))
        experiment_groups_train = groups_position_train
        experiment_groups_test = groups_position_test
        experiment_group_names = group_names_position_train
    elif split_type == 'array':
        fold_set = range(len(groups_array_test))
        experiment_groups_train = groups_array_train
        experiment_groups_test = groups_array_test
        experiment_group_names = group_names_array_train
    else:
        raise AssertionError(f'Bad split type {split_type}')

    print('Train set : ')
    for i in experiment_groups_train[0]:
        print(f'Train : {ids_train[i]}')
    print('Test set : ')
    for i in experiment_groups_test[0]:
        print(f'Test : {ids_test[i]}')

    for i, this_outgroup in enumerate(experiment_groups_train):
        eval_idxs = experiment_groups_test[i]

        val_idxs = []
        for i_g in range(len(groups_position_train)):
            possible_choices = [j for j in groups_position_train[i_g].tolist() if j not in this_outgroup.tolist()]
            if len(possible_choices) == 1:
                if np.random.rand() < .15:
                    val_idxs += possible_choices
            elif len(possible_choices) > 0:
                val_idxs += np.random.choice(
                    possible_choices, int(.15 / len(groups_position_train) * x_out_train.shape[0]),
                    replace=False).tolist()
        val_idxs = np.sort(val_idxs).tolist()

        train_idxs = np.array([j for j in range(x_out_train.shape[0]) if j not in
                               (this_outgroup.tolist() + val_idxs)]).astype(int)
        print(f'For fold (zero counting absolute) {fold_set[i]} (at step: {i + 1}'
              f' of {len(fold_set)}), at {experiment_group_names[i]}'
              f' with {train_idxs.size} train '
              f' and {eval_idxs.size} test samples')

        evaluation_eval_func = lambda cmodel: callback_eval_func(
            x_out_test[eval_idxs, :, :], y_test[eval_idxs, :], cmodel, prefix='Test')

        model_filename_fold = model_filename.replace('.h5', '_fold_' + str(i) + '.h5')
        if exists(model_filename_fold):
            print('Loading pre-trained model from  {model_filename_fold}')
            model = load_model(model_filename_fold)
            print('Loaded pre-trained model from  {model_filename_fold}')
            model.summary()
        else:
            print(f'File {model_filename_fold} does not exist so i will train the model')
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

            effective_batch_size = BATCH_SIZE_BASE_PER_CLASS_SPEECH * y_position_train.shape[1]
            this_batch_gen = batch_gen(
                x_out_train, y_train, BATCH_SIZE_BASE_PER_CLASS_SPEECH,
                y_to_balance=y_position_train, sub_idxs=train_idxs)
            steps_per_epoch = min(int(round(y_train.shape[0] / effective_batch_size)), MAX_STEPS_PER_EPOCH)

            print(f'For a batch size of {effective_batch_size} i will do {steps_per_epoch} steps per epoch.')
            this_batch_gen_val = batch_gen(
                x_out_train, y_train, BATCH_SIZE_BASE_PER_CLASS_SPEECH,
                y_to_balance=y_position_train, sub_idxs=val_idxs)
            _, filename = model_trainer(
                this_batch_gen, x_out_train.shape[1:], y_train.shape[1],
                get_model_speech,
                tensorlog=True, callbacks=callbacks, epochs=MAX_EPOCHS,
                loss_patience=10, scratchpad=SCRATCHPAD,
                model_filename=model_filename_fold, use_attention=use_attention,
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
    C. Papayiannis, C. Evers and P. A. Naylor, "End-to-End Classification of Reverberant Rooms Using DNNs," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 28, pp. 3010-3017, 2020, doi: 10.1109/TASLP.2020.3033628.
    
    For usage help, run : python ace_discriminative_nets.py --help 
    """

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
    parser.add_argument('--speech', dest='speech', type=str, required=True, nargs='*',
                        help='Locations where the wav files')
    parser.add_argument('--readcache', dest='readcache', action="store_true",
                        default=False,
                        help='Do not load new data, just read the last cached data')
    parser.add_argument('--attention', action="store_true",
                        default=False, help='Use attention')
    parser.add_argument('--cnn', dest='cnn', action="store_true",
                        default=False, help='Add CNN layers to the net')
    parser.add_argument('--rnn', dest='rnn', action="store_true",
                        default=False, help='Add RNN layers to the net')
    parser.add_argument('--cacheloc', dest='cacheloc', type=str, default='/tmp/',
                        help='Locations where the cache is located and/or will be stored')
    parser.add_argument('--split-type', type=str, default='position', choices=('position', 'array'),
                        help='ACE Splits')

    args = parser.parse_args()

    run_command(f'mkdir -p {SCRATCHPAD}')

    train_eval(args.h5, args.ace_base, _TIMESTAMP, speech_dir=args.speech, read_cache=args.readcache,
               use_cnn=args.cnn, use_rnn=args.rnn, use_attention=args.attention,
               cacheloc_master=args.cacheloc, split_type=args.split_type)
