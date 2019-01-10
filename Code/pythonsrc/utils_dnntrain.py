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

This file contains a number of routines useful in the training and evaluation of DNNs using Keras.

This file was original distributed in the repository at:
{repo}

If you use this code in your work, then cite:
{placeholder}

"""

from random import shuffle

import numpy as np
from keras.callbacks import Callback


def multi_batch_gen(x_data_list, y_out_list, samples_per_class, y_to_balance=None, sub_idxs=None,
                    **kwargs):
    """
    Batch generator for keras model training. Based on the function in this file 'batch_gen', this
    function allows for batch generators to be created which fuse together a number of datasets.

    In addition to the documentation of 'batch_gen', this function allows for a batch generator
    to be created which accepts a list arrays X, which contain training data and their
    corresponding labels Y. Then the batches will be constructed as if the data were part of only
    one dataset, using samples from all of them during hte batch construction. It is useful if
    the arrays are too big to be loaded onto RAM and they are stored in HDF5 datasets for instance.

    Args:
        x_data_list: List of X data as
            [[N_samples_1 X data_dimensionality],
             [N_samples_2 X data_dimensionality],...]
        y_out_list: List of labels or Y data as:
            [[N_samples_1 X out_data_dim],...
             [N_samples_2 X out_data_dim],...]
        samples_per_class: List of number of samples per class to take, per set
        y_to_balance: List of labels or Y data to be used to balance selections from each batch
        as:
            [[N_samples_1 X 1],...
             [N_samples_2 X 1],...]
        sub_idxs:
        **kwargs: List of lists of indices of samples to consider, indicating their locations in
        the X and Y array.

    Returns: The generator.

    """
    nsets = len(x_data_list)
    if y_to_balance is None:
        y_to_balance = [None] * nsets
    if sub_idxs is None:
        sub_idxs = [None] * nsets

    if not nsets == len(y_out_list):
        raise AssertionError('Invalid inputs')
    if not nsets == len(samples_per_class):
        raise AssertionError('Invalid inputs')
    if y_to_balance is None:
        y_to_balance = [None for _ in range(nsets)]
    if sub_idxs is None:
        sub_idxs = [None for _ in range(nsets)]
    if not nsets == len(samples_per_class):
        raise AssertionError('Invalid inputs')

    gens = []
    for i in range(nsets):
        gens.append(
            batch_gen(x_data_list[i], y_out_list[i], samples_per_class[i], sub_idxs=sub_idxs[i],
                      y_to_balance=y_to_balance[i], **kwargs))

    while True:
        xs = [None for _ in range(nsets)]
        ys = [None for _ in range(nsets)]
        for i in range(nsets):
            xs[i], ys[i] = gens[i].next()
        yield np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def batch_gen(x_data, y_out, samples_per_class, y_to_balance=None, no_check=True, verbose=False,
              sub_idxs=None, augmentation_func=None, aug_prob=1.):
    """
    Batch generator for keras model training. Features:
    *   It is capable of handling simultaneously multiple Inputs, useful for Functional API models.
    *   Performs batch balancing in terms of the classes in Y. It accounts for class imbalance
    in the data. It can do that based on the give Y labels or balancing data can be given.
    *   A smaller subset of the data can be made visible to batch generator and the rest will be
    ignore and not used for the training.
    *   A data augmentation function can be given, which will modify a percentage of the batch
    samples. The percentage is configurable with an augmentation probability.
    *   Able to directly work on HDF5 datasets.

    (Array refers to numpy array or HDF5 datasets)
    Args:
        x_data: Array of training input data as [N_samples X data_dimensionality], or a list of
        such arrays, which would correspond to the set of inputs to the network for Functional
        API nets with multiple inputs.
        y_out: Labels of the data, or y data for regression as an array
        [N_samples X out_data_dimensionality].
        samples_per_class: The number of samples to include in each batch from each class
        y_to_balance: An array of labels for the data as [N_samples X 1], used to balance the data.
        no_check: Skip any checks
        verbose: Verbose output
        sub_idxs: Indices of samples to consider, indicating their locations in the X and Y
        array. Passed as an iterable.
        augmentation_func: Function used to modify samples as a data augmentation strategy.
        aug_prob: Probability of a sample in the batch to be modified, using the augmentation_func.

    Note: If you do not want the balancing operation then pass  y_to_balance=[0]*N_samples
        and samples_per_class=desired_batch_size

    Returns: The generator

    """
    turn_to_list = False
    if not isinstance(x_data, list) or isinstance(x_data, tuple):
        if isinstance(augmentation_func, list) or isinstance(augmentation_func, tuple):
            raise AssertionError('Unexpected condition')
        x_data = [x_data]
        turn_to_list = True
    nsets = len(x_data)
    if len(x_data) > 1:
        for i in range(1, nsets):
            if not x_data[i].shape[0] == x_data[i].shape[i]:
                raise AssertionError('Input error')

    if sub_idxs is None:
        sub_idxs = np.arange(0, x_data[0].shape[0]).astype(int)
    else:
        sub_idxs = np.sort(sub_idxs)

    if augmentation_func is None:
        augmentation_func = [lambda x: x for _ in range(nsets)]
    elif turn_to_list:
        augmentation_func = [augmentation_func]
    if not (isinstance(augmentation_func, list) or isinstance(augmentation_func, tuple)):
        raise AssertionError('Invalid Input')
    for i in range(nsets):
        if augmentation_func[i] is None:
            augmentation_func[i] = lambda x: x

    if not len(augmentation_func) == nsets:
        raise AssertionError('Input error')

    if verbose:
        from fe_utils import print_split_report
    if y_to_balance is None:
        if y_out is None:
            raise AssertionError('Cannot work without any y\'s')
        y_to_balance = y_out

    index_pools = []
    for i in range(y_to_balance.shape[1]):
        index_pools.append(np.where(y_to_balance[sub_idxs, i])[-1].tolist())
        shuffle(index_pools[-1])
    else:
        samples_per_pool = samples_per_class
    counter = [0 for _ in range(len(index_pools))]
    while True:
        these_idxs = []
        for i in range(len(index_pools)):
            if len(index_pools[i]) > 0:
                subsamples = min(samples_per_pool, len(index_pools[i]))
                repeats = int(np.ceil(samples_per_pool / float(subsamples)))
                for _ in range(repeats):
                    these_idxs += (index_pools[i][counter[i]:counter[i] + samples_per_pool])
                remove = subsamples * repeats - samples_per_pool
                if remove > 0:
                    these_idxs = these_idxs[0:-remove]
                counter[i] += len(these_idxs)
        for i in range(len(index_pools)):
            if counter[i] >= len(index_pools[i]):
                if verbose:
                    print('Reshuffling batch gen pool ' + str(i) + ' because i gave you ' +
                          str(counter[i]) + ' samples already')
                shuffle(index_pools[i])
                counter[i] = 0
        if not no_check and (not len(these_idxs) == samples_per_pool * y_out.shape[1]):
            raise AssertionError('Generator failure')
        if verbose:
            print('New batch ready: ')
            print_split_report(y_to_balance[sub_idxs[these_idxs], ...])
        these_idxs = np.sort(these_idxs).tolist()

        if aug_prob == 0:
            if isinstance(x_data, list) or isinstance(x_data, tuple):
                out_x_aug = [x_data[k][sub_idxs[these_idxs], ...] for k in range(len(x_data))]
            else:
                out_x_aug = x_data[sub_idxs[these_idxs], ...]
        else:
            out_x_aug = [np.concatenate([
                augmentation_func[k](x_data[k][i:i + 1, ...])
                if np.random.rand() < aug_prob
                else
                x_data[k][i:i + 1, ...]
                for i in sub_idxs[these_idxs]
            ],
                axis=0) for k in range(nsets)]
        returner = lambda x: x
        if turn_to_list:
            returner = lambda x: x[0]
        if y_out is None:
            out_y = None
        elif isinstance(y_out, list) or isinstance(y_out, tuple):
            out_y = [y_out[k][sub_idxs[these_idxs], ...] for k in range(len(y_out))]
        else:
            out_y = y_out[sub_idxs[these_idxs], ...]
        yield returner(out_x_aug), out_y


def model_trainer(the_batch_gen, in_shape, out_shape, get_model, val_patience=15,
                  loss_patience=10, val_gen=None,
                  tensorlog=False, callbacks=[], model_filename=None, epochs=1000,
                  save_model_image=True, print_summary=True, steps_per_epoch=10,
                  scratchpad='/tmp/', **kwargs):
    """
    A function which trains a Keras model for classification. It handles callbacks
    and puts together the training strategy, given a batch generator.

    Args:
        the_batch_gen: Batch generator, able to be used with Keras fit_generator
        in_shape: Input dimensionality for the model
        out_shape: Number of classes
        get_model: A function which constructs the model as
            get_model(input_shape, out_shape, **kwargs)
            and returns only a Keras Sequential model
        val_patience: Validation loss patience for Early Stopping.
        loss_patience: Training loss patience for Early Stopping.
        val_gen: Batch generator, able to be used with Keras fit_generator. Used for generating
        validation data.
        tensorlog: Location to save Tensorboard logs.
        callbacks: A list of callbacks to append to the network.
        model_filename: The filename for the model to be saved in.
        epochs: Number of training epochs
        save_model_image: Ask for the model diagram to be saved.
        print_summary: Print a summary of the model structure.
        steps_per_epoch: Number of generator batches to be used per epoch.
        scratchpad: Location for any data saving
        **kwargs: Passed to get_model

    Returns:

    """
    from time import time
    from keras.callbacks import EarlyStopping, TensorBoard

    timestamp = str(time())
    if model_filename is None:
        from subprocess import call
        call(["mkdir", "-p", scratchpad])
        model_filename = scratchpad + '/ace_model' + timestamp + '.h5'
    input_shape = in_shape
    model = get_model(input_shape, out_shape, **kwargs)
    if print_summary:
        model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    if loss_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor='loss', min_delta=0, patience=loss_patience, verbose=1,
                mode='auto'))
    if val_patience is not None:
        if val_patience > 0:
            callbacks.append(
                EarlyStopping(
                    monitor='val_loss', min_delta=0, patience=val_patience, verbose=1,
                    mode='auto'))
    if tensorlog:
        tensordir = model_filename.replace('.h5', '_tensorlog')
        callbacks.append(
            TensorBoard(log_dir=tensordir, histogram_freq=0,
                        batch_size=16,
                        write_graph=False, write_grads=False, write_images=True,
                        embeddings_freq=0, embeddings_layer_names=None,
                        embeddings_metadata=None))
        print('Will save Tensorboard Logs at : ' + tensordir)
    if save_model_image:
        imgdir = model_filename.replace('.h5', '.pdf')
        try:
            from keras.utils import plot_model
            plot_model(model, to_file=imgdir, show_shapes=True)
        except ImportError:
            print('Could not save model image')
        print('Saved model image at: ' + imgdir)

    print('Training...')
    model.fit_generator(the_batch_gen, epochs=epochs,
                        validation_data=val_gen, validation_steps=3,
                        verbose=0, callbacks=callbacks, steps_per_epoch=steps_per_epoch, )
    for i in callbacks:
        if type(i) is PostEpochWorker:
            best_val_model = i.best_val_model
            if best_val_model is not None:
                model = best_val_model
    try:
        model.save(model_filename)
        print('Saved : ' + model_filename)
    except IOError as ME:
        print('Could not save model ' + model_filename + ' because ' + ME.message)
    return model, model_filename


def accuracy_eval(x, y, cmodel, prefix=None):
    """
    Accepts input data, labels and a model to predict the labels, which are then evaluated in
    terms of their accuracy.

    Can be combined with PostEpochWorker, to provide an evaluation of the accuracy in a custom
    way during the training of DNNs as:

     PostEpochWorker(
                (x_out_train[val_idxs, :, :],
                 x_out_test[test_idxs, :, :]),
                (y_train[val_idxs, :],
                y_test[test_idxs, :]),
                model_filename,
                eval_fun=(
                    lambda x, y, cmodel: accuracy_eval(x, y, cmodel, prefix='Val'),
                    lambda x, y, cmodel: accuracy_eval(x, y, cmodel, prefix='Test')),
                eval_every_n_epochs=100)

    Args:
        x: Input data
        y: Labels
        cmodel: Trained model for inference
        prefix: Prefix for the reporting

    Returns: The predictions

    """
    from utils_base import float2str
    y_pred = np.argmax(cmodel.predict(x), axis=1).flatten()
    acc = np.sum(y_pred == np.argmax(y, axis=1)).flatten() / float(x.shape[0])
    print(((prefix + ' ') if prefix is not None else '') + 'Accuracy: ' + float2str(acc, 4))
    y_pred_out = np.zeros_like(y)
    for i in range(y_pred_out.shape[0]):
        y_pred_out[i, y_pred[i]] = True
    n_hots = np.sum(y_pred_out, axis=1)
    if ~np.all(n_hots == 1):
        too_hot = np.where(~(n_hots == 1))[-1]
        raise AssertionError(
            'Predictions do not make sense because the following idxs had more than one hots ' +
            str(too_hot) + ' with the following hots ' + str(n_hots[too_hot]))
    return y_pred_out


def get_scaler_descaler(x, verbose=False):
    """
    Creates scaling and descaling functions for preparation of training data and reconstruction
    from DNNs

    Args:
        x: Input data
        verbose: Verbose reporting

    Returns:
        Scaler function object
        Descaler function object

    """
    import numpy as np
    from utils_base import float2str
    if x.ndim > 2:
        conced_x = np.concatenate(x, axis=0)
    else:
        conced_x = np.array(x)
    subval = np.min(conced_x, axis=0)
    scale_val = np.max(conced_x, axis=0) - subval
    scale_val[scale_val == 0] = 1

    subval.shape = tuple([1, 1] + list(subval.shape))
    scale_val.shape = tuple([1, 1] + list(scale_val.shape))

    if verbose:
        print('Will construct scaler with subs: ' + float2str(
            subval) + "\n" + '... and scalers ' + float2str(
            scale_val))

    def scaler(y):
        return (y - subval) / scale_val

    def descaler(y):
        return y * scale_val + subval

    return scaler, descaler


class PostEpochWorker(Callback):
    def __init__(self, x_data, y_data, model_filename, eval_fun=None, eval_every_n_epochs=1,
                 save_best=True):
        """
        Produces an instance of a keras Callback. It allows for running a set of routines when a
        number of training epochs has elapsed.

        Each routine accepts the X and Y data and the best performing model up to the current
        epoch. The best performing model is evaluated by the validation accuracy (or the train
        accuracy if no validation is done). It allows for you to evaluate you model during
        training and print performance reports of the model. it also allows you to have snapshots
        of your model during training.

        Args:
            x_data: The list of X data
            y_data: The list of Y data
            model_filename: The filename to use for saving the model snapshots
            eval_fun: The list of functions to run
            eval_every_n_epochs: The number of epochs after each to run the routines
            save_best: Set True to save the model snapshots
        """
        Callback.__init__(self)
        self.save_best = save_best
        self.eval_every_n_epochs = eval_every_n_epochs
        self.eval_fun = eval_fun
        if eval_fun is None:
            self.eval_fun = []

        elif not isinstance(eval_fun, list) and not isinstance(eval_fun, tuple):
            self.eval_fun = [eval_fun]
        else:
            self.eval_fun = eval_fun
        if not isinstance(x_data, list) and not isinstance(x_data, tuple):
            self.x_test = [x_data]
        else:
            self.x_test = x_data
        if not isinstance(y_data, list) and not isinstance(y_data, tuple):
            self.y_test = [y_data]
        else:
            self.y_test = y_data
        if not isinstance(self.y_test, type(self.x_test)) or not isinstance(self.eval_fun,
                                                                            type(self.x_test)):
            assert AssertionError('Given types for x_y or incorrect')
        self.best_val_loss = None
        self.model_filename = model_filename
        self.best_val_model = None
        self.update_since_last = False

    def on_train_begin(self, logs={}):
        self.losses = []
        if self.save_best:
            print('Will save best model as ' + self.model_filename)

    def run_eval(self, epoch, logs={}):
        import numpy as np
        if epoch is None:
            eval_go = True
        else:
            eval_go = ((epoch + 1) % self.eval_every_n_epochs == 0) or self.eval_every_n_epochs == 1
        if eval_go:
            if len(self.eval_fun) > 0:
                if self.update_since_last:
                    print('Running eval at epoch ' + (
                        str(epoch) if epoch is not None else '*last*') +
                          '                                                                       ')
                    for i in range(len(self.eval_fun)):
                        if self.eval_every_n_epochs > 1:
                            self.eval_fun[i](np.array(self.x_test[i]), np.array(self.y_test[i]),
                                             self.best_val_model)
                        else:
                            self.eval_fun[i](np.array(self.x_test[i]), np.array(self.y_test[i]),
                                             self.model)
                else:
                    print 'Skipping eval of epoch ' + (
                        str(epoch) if epoch is not None else '*last*') \
                          + ' since this is a dead season' + \
                          '                                          ' + '\r',
            self.update_since_last = False

    def on_epoch_end(self, epoch, logs={}):
        from utils_base import float2str

        cval_loss = logs.get('val_loss')
        loss_name = 'val_loss'
        if cval_loss is None:
            loss_name = 'loss'
            cval_loss = logs.get('loss')
        if self.best_val_loss is None or cval_loss <= self.best_val_loss:
            self.update_since_last = True
            self.best_val_loss = cval_loss
            self.best_val_model = self.model
            if self.save_best:
                try:
                    self.model.save(self.model_filename)
                except TypeError:
                    print('Could not save model ' + self.model_filename)
            print 'At epoch : ' + str(
                epoch) + ' found new best ' + loss_name + ' model with ' + loss_name + ' ' + \
                  float2str(self.best_val_loss, 12) + '                         ' + '\r',

        self.run_eval(epoch)

    def on_train_end(self, logs=None):
        print '                                                                          \r',
        self.run_eval(None)
