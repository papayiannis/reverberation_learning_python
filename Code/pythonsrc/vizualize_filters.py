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

The purpose of the file is to allow for the visualization of results obtained using CNNs. The
filter kernels can be visualized and also data can be filtered by the network and plotted.

More information given below. It is used for the results shown in:
{placeholder}

This file was original distributed in the repository at:
{repo}

If you use this code in your work, then cite:
{placeholder}

"""

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D
from keras.models import load_model
from matplotlib import rc
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter

from ace_discriminative_nets import model_fs, \
    model_framesize_air, max_air_len, get_pow_spec, start_at_max, max_speech_len, \
    model_framesize_speech
from fe_utils import data_post_proc
from utils_base import run_command, add_axis_back, float2str
from utils_spaudio import my_resample

zoom_scale_for_thesis = 3
trim_speech_to = None


def apply_relu_gate(x_in):
    """
    Applies ReLU gating at the input.

    Args:
        x_in: Iterable

    Returns: Gated input

    """
    # print('Applying ReLU to ' + str(x_in.shape))
    x_in = np.array(x_in, dtype=x_in.dtype)
    x_in[x_in < 0] = 0.
    # print('Got from ReLU ' + str(x_in.shape))
    return x_in


def float_norm(mat):
    """
    Normalizes input between 0 and 1

    Args:
        mat: Iterable

    Returns: Normalized input

    """
    # mat = np.log(mat ** 2)
    valid_idx = ~np.isnan(mat) & ~np.isinf(mat)
    mat = mat - mat[valid_idx].min()
    max_scale = np.abs(mat[valid_idx]).max()
    if not max_scale == 0:
        mat = mat / max_scale
    mat[np.isnan(mat)] = 0
    mat[np.isinf(mat)] = 0
    if np.any(mat > 1):
        raise AssertionError('Unexpected condition at high bound ' + float2str(mat.max()))
    if np.any(mat < 0):
        raise AssertionError('Unexpected condition at low bound ' + float2str(mat.min()))
    return mat


def colorfy(mat, smooth=False):
    """
    Create a color grid based on the input array

    Args:
        mat: Input data
        smooth: Smooth the data using a Gaussian kernel

    Returns: The color pixel grid

    """

    mat = mat - np.min(mat)
    max_scale = np.max(mat)
    if not max_scale == 0:
        mat = mat / max_scale
    mat = mat ** 2

    mat = add_axis_back(float_norm(np.array(mat)))
    mat = np.concatenate(
        (
            mat,
            np.zeros_like(mat),
            np.zeros_like(mat),
        ), axis=2
    )
    if smooth:
        mat = gaussian_filter(mat,
                              sigma=(.5, .5, 0), order=0)
    return mat


def viz_net_individual(model, outdir, nrows=4, interactive=False):
    """
    Vizualizes individual filters

    Args:
        model: The Keras model, which contains convolutional layers
        outdir: Directory to output the vizualizations
        nrows: Number of rows in the plots
        interactive: Interactive plotting

    Returns: Nothing

    """
    conv_layers = []
    conv_layers_idxs = []
    for idx, i in enumerate(model.layers):
        if isinstance(i, Conv2D):
            conv_layers.append(i)
            conv_layers_idxs.append(idx)

    for filter_idx in range(len(conv_layers)):
        plt.close()
        this_filter_idx = filter_idx
        conv_layer = conv_layers[this_filter_idx]
        filters = conv_layer.get_weights()[0]

        n_filters = filters.shape[-1]
        ncols = int(np.ceil(n_filters / float(nrows)))
        fig = plt.figure(1, figsize=(1.65 * zoom_scale_for_thesis *
                                     (ncols / 2.), 1.65 * zoom_scale_for_thesis *
                                     (nrows / 2.)))
        fig.subplots_adjust(left=0.05, right=0.95)
        grid = AxesGrid(fig, 111,
                        nrows_ncols=(nrows, ncols,),
                        axes_pad=0.1,
                        share_all=True
                        )
        for i in range(n_filters):
            grid[i].imshow(colorfy(filters[:, :, 0, i]),
                           origin='upper', extent=(-3, 4, -4, 3),
                           interpolation='nearest')
            grid[i].set_xticks([])
            grid[i].set_yticks([])

        run_command('mkdir -p ' + outdir)

        filename = outdir + '/filt_viz' + str(filter_idx) + '.png'
        plt.savefig(filename)
        print('Saved: ' + filename)
        if interactive:
            plt.show()
        if len(conv_layers) > 1:
            print('Done with ' + str(filter_idx))


def viz_net_other(model, x, suffix, outdir, nrows=4, interactive=False, doing_speech=False,
                  add_axis_labels=False):
    """
    Visualizing data as feature maps passed through the layers of a CNN

    Args:
        model: CNN model
        x: Audio data to be filtered, as audio samples in np.array
        suffix: Suffix used for plot saving
        outdir: The output directory for the results
        nrows: Number of rows to be used for the plotting
        interactive: Interactive plotting (waits for you to close the plots)
        doing_speech: Indicates whether the input is a speech recording. The alternative is to
        use a measured AIR.
        add_axis_labels: Label the axis

    Returns: Nothing

    """
    fex_op = lambda x_in: data_post_proc(
        x=x_in,
        fs=model_fs,
        start_at_max=start_at_max if not doing_speech else False,
        framesize=model_framesize_air if not doing_speech else model_framesize_speech,
        get_pow_spec=get_pow_spec,
        max_len=max_air_len if not doing_speech else max_speech_len,
        wavform_logpow=False)[0, :, :]
    samples = fex_op(x)

    plt.close()

    try:
        print('Samples size :' + str(samples.shape))
    except AttributeError:
        pass

    preds = model.predict(np.stack([samples]))
    n_filters = preds.shape[-1]
    nrows = min(nrows, n_filters)
    ncols = int(np.ceil(n_filters / float(nrows)))
    fig = plt.figure(1, figsize=(1.65 * zoom_scale_for_thesis *
                                 (ncols / 2.), 1.65 * zoom_scale_for_thesis *
                                 (nrows / 2.)))
    fig.subplots_adjust(left=0.05, right=0.95)
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(nrows, ncols,),
                    axes_pad=0.05,
                    share_all=True,
                    )
    for i in range(n_filters):
        if trim_speech_to is not None:
            these_preds = preds[0,
                          0:int(trim_speech_to / float(max_speech_len) *
                                preds.shape[1]), :, i]
        else:
            these_preds = preds[0, :, :, i]
        activations_plt = colorfy(np.transpose(np.flip(these_preds, 1)))

        if np.any(np.isnan(activations_plt)):
            raise AssertionError('Nan in activations')

        grid[i].imshow(activations_plt,
                       origin='upper', extent=(-3, 4, -4, 3),
                       interpolation='nearest')
        grid[i].set_xticks([])
        grid[i].set_yticks([])
        if add_axis_labels:
            if i % nrows == 0:
                grid[i].set_ylabel('Freq. Bin')
            if (ncols * nrows) - ncols <= i:
                grid[i].set_xlabel('Frame')

    # plt.tight_layout()
    run_command('mkdir -p ' + outdir)
    filename = outdir + '/filt_viz' + suffix + '.png'
    plt.savefig(filename)
    print('Saved: ' + filename)
    if interactive:
        plt.show()
    print('Done')


def viz_net(model_loc, air_loc=None, nrows=4, interactive=False, channel=0,
            layer_idx=0, speechfile=None):
    """
    Visualization worker. Accepts the model and a set of audio data which can be filtered by the
    network to provide the visualizations. If no input audio data are given, then the filter
    kernels are visualised.

    Args:
        model_loc: The location of the model saved by keras an HDF5 dataset
        air_loc: Location of AIR file .wav
        nrows: Number of rows used in plotting
        interactive: Interactive plotting (waits for you to close the plots)
        channel: Channel of the AIR to do
        layer_idx: =n. The output of the n-th convolutional layer will be used to collect the
        feature maps.
        speechfile: A speech file which will be convolved with the AIR before filtering.

    Returns:

    """

    try:
        from os.path import basename
    except ImportError:
        raise
    from scipy.signal import fftconvolve

    outdir = '/tmp/training_surface_models/' + basename(model_loc).replace('.h5', '') + '/'

    i_made_the_model = False
    try:
        model = load_model(model_loc)
    except ValueError as ME1:
        i_made_the_model = True
        try:
            from ace_discriminative_nets import get_model_speech
            print('Failed to use default load for model ' + model_loc +
                  ' wil try for speech cnn because ' + ME1.message)
            model = get_model_speech((500, 161), 7, use_cnn=True)
            model.load_weights(model_loc, by_name=True)
            print('CNN model constructed OK')
        except ValueError as ME2:
            try:
                print('Failed to use cnn model ' + model_loc +
                      ' wil try for speech cnn-rnn because ' + ME2.message)
                from ace_discriminative_nets import get_model_speech
                model = get_model_speech((500, 161), 7, use_cnn=True, use_rnn=True)
                model.load_weights(model_loc, by_name=True)
                print('CNN-RNN model constructed OK')
            except ValueError as ME3:
                print('Failed to use cnn-rnn model ' + model_loc +
                      ' wil try for speech cnn-rnn because ' + ME3.message)
                raise ME1

    if air_loc is None:
        viz_net_individual(model, outdir, nrows=nrows, interactive=interactive)
        return

    conv_layers = []
    conv_layers_idxs = []
    for idx, i in enumerate(model.layers):
        if isinstance(i, Conv2D):
            conv_layers.append(i)
            conv_layers_idxs.append(idx)

    from keras.models import Model

    if layer_idx == -1:
        effective_idx = 1
    else:
        if layer_idx >= len(conv_layers_idxs):
            effective_idx = conv_layers_idxs[-1] + 2
            print(
                'I will assume that you want the next layer after the last conv layer which i will '
                'assume is a max poolign layer')
        else:
            effective_idx = conv_layers_idxs[layer_idx]
    print('Picking Layer ' + model.layers[effective_idx].name)
    model = Model(inputs=[model.input], outputs=[model.layers[effective_idx].output])

    if speechfile is not None:
        fs_speech, x_speech = wavfile.read(speechfile)
    else:
        x_speech = None
        fs_speech = None

    for this_air in air_loc:
        if this_air == 'white':
            suffix = '_white'
            in_shape = model.input_shape[1:]
            x = np.atleast_2d(np.random.normal(0, 1., size=np.prod(in_shape) * 4))
        else:
            suffix = '_' + run_command('basename ' + this_air)[0]
            print('Loading: ' + this_air)
            fs, x = wavfile.read(this_air)
            x = x[:, channel]
            if speechfile is not None:
                print('Will convolve with ' + speechfile)
                if not fs_speech == fs:
                    x_speech_effective = my_resample(x_speech[0:int(max_speech_len * fs_speech)],
                                                     fs_speech, fs)
                else:
                    x_speech_effective = x_speech
                x_speech_effective.setflags(write=1)
                if trim_speech_to is not None:
                    x_speech_effective[int(trim_speech_to * fs_speech):] = 0
                x = fftconvolve(x_speech_effective, x, mode='same')
            else:
                if i_made_the_model:
                    raise AssertionError(
                        'Because the default model-load failed i was going '
                        'to try to construct speech '
                        'models but you did not provide any speech data')
            if x.ndim > 1:
                x = x[:, 0]
            x = np.atleast_2d(x)
        suffix += '_l' + str(layer_idx)
        viz_net_other(model, x, suffix, outdir, nrows=nrows, interactive=interactive,
                      doing_speech=speechfile is not None)


if __name__ == "__main__":
    """
    The purpose of the file is to allow for the visualization of results obtained using CNNs. The 
    filter kernels can be visualized and also data can be filtered by the network and plotted.
    
    Arguments for visualizing CNN models

    optional arguments:
      -h, --help            show this help message and exit
      --evalmodel MODEL_LOC
                            Model to load and evaluate
      --air [AIR [AIR ...]]
                            Wav file of AIR to load and filter
      --speech SPEECH       Wav file of speech utterance to convolve with the AIR
      --interactive         Interactive plotting
      --layer LAYER         Index of conv layer to plot. The n-th convolutional
                            layer. Not the absolute layer index. Starts from 0.
      --channel CHANNEL     AIR Channel to use
      --rows ROWS           Number of rows of plot
    
    """

    np.random.seed(784)

    rc('font', family='Times New Roman', serif='Times')
    rc('xtick', labelsize=8 * zoom_scale_for_thesis)
    rc('ytick', labelsize=8 * zoom_scale_for_thesis)
    rc('axes', labelsize=8 * zoom_scale_for_thesis)
    rc('text', usetex='true')
    # exit(0)

    import argparse

    parser = argparse.ArgumentParser(description='Arguments for visualizing CNN models')
    parser.add_argument('--evalmodel', dest='model_loc', type=str,
                        help='Model to load and evaluate')
    parser.add_argument('--air', dest='air', type=str, default=None, nargs='*',
                        help='Wav file of airs to load and filter')
    parser.add_argument('--speech', dest='speech', type=str, default=None,
                        help='Wav file of speech utterance to convolve with the AIR')
    parser.add_argument('--interactive', dest='interactive', action="store_true",
                        default=False, help='Interactive plotting')
    parser.add_argument('--layer', dest='layer', type=int, default=0,
                        help='Index of conv layer to plot. The n-th convolutional layer. '
                             'Not the absolute layer index. Starts from 0.')
    parser.add_argument('--channel', dest='channel', type=int, default=0,
                        help='AIR Channel to use')
    parser.add_argument('--rows', dest='rows', type=int, default=4,
                        help='Number of rows of plot')

    args = parser.parse_args()

    # print('Got args ' + str(args))

    viz_net(args.model_loc, air_loc=args.air, interactive=args.interactive, layer_idx=args.layer,
            speechfile=args.speech, channel=args.channel, nrows=args.rows)
