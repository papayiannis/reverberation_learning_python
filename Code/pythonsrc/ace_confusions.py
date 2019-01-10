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

This file provides a worker for plotting confusion matrices for the classification of data from
the ACE challenge database (http://www.ee.ic.ac.uk/naylor/ACEweb/index.html).

More information below.

This file was original distributed in the repository at:
{repo}

If you use this code in your work, then cite:
C. Papayiannis, C. Evers, and P. A. Naylor, "End-to-End Classification of Reverberant Rooms using DNNs," arXiv preprint arXiv:1812.09324, 2018.

"""

import numpy as np

from sklearn.metrics import confusion_matrix


def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(10, 10)):
    """

    ## From : https://gist.github.com/hitvoice/36cf44689065ca9b927431546381a3f7
    ## Github: Runqi Yang @hitvoice

    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """

    from utils_base import float2str

    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if False and i == j:
                s = cm_sum[i]
                annot[i, j] = float2str(p, 1) + "\%\n" + str(c)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = float2str(p, 1) + "\%\n" + str(c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cbar=False, cmap='Blues',
                annot_kws={'family': 'serif', 'size': font_size})
    return fig


if __name__ == "__main__":

    """

    This file provides a worker for plotting confusion matrices for the classification of data from
    the ACE challenge database (http://www.ee.ic.ac.uk/naylor/ACEweb/index.html).
    
    The predictions csv should be in this format:
    Chromebook_502_1_RIR_13_ch0,611
    .
    .
    .
    Mobile_508_1_RIR_14_ch2,EE-lobby
    
    You get the preds.csv file by running 
    ace_discriminative_nets.py [arguments] > logfile.txt
    And then:
    tac logfile.txt | \ 
    grep -m 1 -B 1000000 'AIR  ' | tac  | tail -n +3  | head -n -1 | \ 
    sed 's/[[:space:]]\{1,\}/,/g' | cut -d, -f1,2 > preds.csv

    """

    import matplotlib as mpl

    mpl.use('pdf')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    import argparse

    parser = argparse.ArgumentParser(
        description='Arguments for plotting ACE classification confusions.')
    parser.add_argument('--preds', dest='preds', type=str,
                        default='/tmp/log_cnn_rnn_res.csv',
                        help="The location of the csv file which incldues the predictions in this "
                             "format (per line):\n"
                             "ACE_AIR_NAME")
    parser.add_argument('--trues', dest='trues', type=int,
                        default=None,
                        help='Number of positive samples per class. Assumed to be the same for '
                             'all classes. You can use this in the case where the log you have '
                             'provided includes only the mismatches (to make it smaller for very '
                             'large experiments). Then in this case, the number of samples per '
                             'class will be made to match the number you give here and it will '
                             'assume that the remaining samples were correctly predicted.')
    parser.add_argument('--tex', dest='tex', action="store_true",
                        default=False, help='Render using latex')

    args = parser.parse_args()

    per_class = args.trues

    maps = (('403a', 'L2'),
            ('502', 'O1'),
            ('508', 'L1'),
            ('503', 'M1'),
            ('611', 'M2'),
            ('803', 'O2'),
            ('EE-lobby', 'BL'))

    font_size = 8

    if args.tex:
        plt.rc('font', family='serif', serif='Times')
        plt.rc('text', usetex=True)
        plt.rc('xtick', labelsize=font_size)
        plt.rc('ytick', labelsize=font_size)
        plt.rc('axes', labelsize=font_size)

    data = pd.read_csv(args.preds,
                       index_col=None,
                       header=None, )
    data = data[data.columns].astype(str).values

    for i in range(data.shape[0]):
        data[i, 0] = data[i, 0].replace('EE_lobby', 'EE-lobby')
        data[i, 0] = data[i, 0].split('_')[1]
        for j in range(len(maps)):
            data[i, 0] = data[i, 0].replace(maps[j][0], maps[j][1])
            data[i, 1] = data[i, 1].replace(maps[j][0], maps[j][1])
    labels = np.sort(np.unique(data[:, 0]))

    if per_class is not None:
        for i in labels:
            currents = int(np.sum([j == i for j in data[:, 0]]))
            if currents < per_class:
                missing = per_class - currents
                data = np.concatenate((data, [[i, i] for _ in range(missing)]), axis=0)
                print('Added ' + str(missing) + ' for ' + i)
            elif currents > per_class:
                raise AssertionError('Unexpected')

    fig = cm_analysis(data[:, 0].flatten(), data[:, 1].flatten(), labels.flatten(),
                      figsize=(3.5, 3.5))

    # fig.set_size_inches(width, height)
    fname = '/tmp/conf_plot.pdf'
    fig.savefig(fname)
    print('Done with ' + fname)
