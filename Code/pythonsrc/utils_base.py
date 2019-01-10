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
This file defines a set of basic functions to be used across a variety of applications

This file was original distributed in the repository at:
{repo}

If you use this code in your work, then cite:
C. Papayiannis, C. Evers, and P. A. Naylor, "End-to-End Classification of Reverberant Rooms using DNNs," arXiv preprint arXiv:1812.09324, 2018.

"""

from __future__ import print_function

import inspect
import sys

import numpy as np


def matrix_stats(x, flat=False):
    """
    Prints statistics for the data in a given matrix

    Args:
        x: The input data (np.array)
        flat: Flatten the data and provide global statistics instead of per column

    Returns:
        Nothing

    """
    args = {'axis': 0} if not flat else {}
    print_func = lambda x: float2str(x, num_decimals=5)
    print('For data with shape ' + str(x.shape))
    print('Mean : ' + print_func(np.mean(x, **args)))
    print('Median : ' + print_func(np.median(x, **args)))
    print('Min : ' + print_func(np.min(x, **args)))
    print('Max : ' + print_func(np.max(x, **args)))
    print('aMin : ' + print_func(np.min(np.abs(x), **args)))
    print('aMax : ' + print_func(np.max(np.abs(x), **args)))
    print('StD : ' + print_func(np.std(x, **args)))


def isiter(x):
    """
    Checks if x is iterable

    Args:
        x: Check this

    Returns:
        Iterable or not

    """
    try:
        if x[0] > 0:
            pass
    except TypeError:
        return False
    except IndexError:
        pass
    return True


def flatten_list(x):
    """
    Flattens lists of lists (of lists, of lists...)

    Args:
        x: The list of lists

    Returns:
        The flat list

    """
    if not isiter(x):
        raise TypeError('You did not given me something which makes sense')
    for i in range(len(x)):
        if isiter(x[i]):
            x[i] = flatten_list(x[i])
        else:
            x[i] = [x[i]]

    xx = np.concatenate([i for i in x]).tolist()
    return xx


def add_axis_back(x, times=1, make_copy=False):
    """
    Add an axis to as the last dimension of a numpy array with dimensionality 1. useful for
    adding channels to DNN training data when they natively are not present in the data.

    Args:
        x: The original data
        times: Number of axis to add
        make_copy: Make a copy of the array before changing it

    Returns:
        The array

    """
    if make_copy:
        x = np.array(x)
    for i in range(times):
        x.shape = tuple(list(x.shape) + [1])
    return x


def add_axis_front(x, times=1, make_copy=False):
    """
    Add an axis to as the first dimension of a numpy array with dimensionality 1. useful for
    adding channels to DNN training data when they natively are not present in the data.

    Args:
        x: The original data
        times: Number of axis to add
        make_copy: Make a copy of the array before changing it

    Returns:
        The new array

    """
    if make_copy:
        x = np.array(x)
    for i in range(times):
        x.shape = tuple([1] + list(x.shape))
    return x


def repack_array_list(array_in, shapes=None, orientation='portrait'):
    """
    Pack an array into a list of vectors. The vectors can be either the rows or the columns.
    This operation is the opposite of flatten_array_list.

    Args:
        array_in: The array
        shapes: The original shapes of the vectors. This assumes that you had a list of vectors
        and each one had its own size. You had to put them into a 2D array so you made some of
        them longer or shorter. This list will contain the shapes of the original vectors.
        orientation: Setting this to 'portrait'means that you stuck the original arrays so that
        they are the columns of the array. Anything else means that they are the rows.

    Returns:
        The list of vectors

    """
    doing_landscape = not orientation == 'portrait'
    outlist = []
    array_in = np.atleast_2d(array_in)
    if shapes is None:
        if doing_landscape:
            shapes = (array_in.shape[1:], 1) * array_in.shape[0]
        else:
            shapes = (0, array_in.shape[1:]) * array_in.shape[0]
    row_counter = 0
    for i in shapes:
        if doing_landscape:
            if len(i) == 1:
                i = [i[0], 1]
            new_row_counter = row_counter + i[0]
            outlist.append(array_in[row_counter:new_row_counter, :][:, 0:i[1]])
        else:
            if len(i) == 1:
                i = [1, i[0]]
            new_row_counter = row_counter + i[1]
            outlist.append(array_in[:, row_counter:new_row_counter][0:i[0], :])
    return outlist


def flatten_array_list(list_in, orientation='portrait'):
    """
    PAcks a list of vectors into an array. The vectors can be either the rows or the columns of
    the new array. This operation is the opposite of repack_array_list.

    Args:
        list_in: The lsit of vectors
        orientation: Setting this to 'portrait'means that you will stick the original arrays so
        that they are the columns of the array. Anything else means that they are the rows.

    Returns:
        The new array
        The shapes of the vectors in the given list
    """

    doing_landscape = not orientation == 'portrait'
    if type(list_in) is np.ndarray:
        out_mat = np.atleast_2d(list_in)
        return out_mat, (out_mat.shape,)
    if len(list_in) < 2:
        out_mat = np.atleast_2d(list_in)
        return out_mat, (out_mat.shape,)
    max_y = 0
    for i in list_in:
        max_y = max(max_y, np.atleast_2d(i).shape[1 - doing_landscape])
    if doing_landscape:
        out_array = np.zeros((max_y, np.sum([np.atleast_2d(i).shape[1] for i in list_in])))
    else:
        out_array = np.zeros((np.sum([np.atleast_2d(i).shape[0] for i in list_in]), max_y))
    counter = 0
    or_shapes = []
    for i in range(len(list_in)):
        or_shapes.append(np.array(list_in[i]).shape)
        twoddlist = np.atleast_2d(list_in[i])
        n_padding = max_y - twoddlist.shape[1 - doing_landscape]
        next_counter = counter + twoddlist.shape[0 + doing_landscape]
        if not doing_landscape:
            newis = np.concatenate(
                (twoddlist, np.zeros((twoddlist.shape[0], n_padding),
                                     dtype=list_in[i].dtype)), axis=1)
        else:
            newis = np.concatenate(
                (np.zeros((twoddlist.shape[0], n_padding),
                          dtype=list_in[i].dtype), twoddlist), axis=1)
        if not doing_landscape:
            out_array[counter:next_counter, :] = newis
        else:
            out_array[:, counter:next_counter] = newis
        counter = next_counter
    return out_array, tuple(or_shapes)


def get_git_hash():
    """

    Get the Git has of the current commit of the repo in this directory

    Returns:
        The hash

    """
    return run_command('git rev-parse HEAD')[0]


def eprint(*args, **kwargs):
    """
    Prints to stderr

    Args:
        *args: Passed to print
        **kwargs: Passed to print

    Returns:

    """
    print(*args, file=sys.stderr, **kwargs)


def run_command_list_stdout(command):
    """
    String to be run in bash

    Args:
        command: The command.

    Returns:
        The stdout as a list of strings. Each string element is a returned line

    """
    std_out = run_command(command)[0]
    return std_out.rstrip().split("\n")


def run_command(command):
    """
    String to be run in bash

    Args:
        command: The command.

    Returns:
        The stdout

    """
    from subprocess import Popen, PIPE
    proc = Popen(command.split(' '), stdout=PIPE, stderr=PIPE)
    std_out, std_err = proc.communicate()
    if len(std_err) > 0:
        print('stderr: ' + std_err)
    return std_out.rstrip(), std_err.rstrip()


def join_strings(str_iter, delim=', '):
    """
    Joins elements of ant iterable as a string delimited by delim

    Args:
        str_iter: An iterable
        delim:  The delimiter

    Returns:
        The concatenated string

    """
    out = ''
    for i in str_iter:
        out += delim + str(i)
    return out[len(delim):]


def find_all_ft(directory_location, ft=".Wav", use_find=True, find_iname=False):
    """
    Finds all files of a specific filetype in a given set of directories (and subdirectories of
    them)

    Args:
        directory_location: The list of directories to look into
        ft: The extension of the file to look for
        use_find: Use the unix `find` command to do this
        find_iname: Use case insensitive search

    Returns:
        The list of files found

    """
    if isinstance(directory_location, str):
        directory_location = [directory_location]
    if use_find:
        name = 'name'
        if find_iname:
            name = 'iname'
        directories = ''
        for i in directory_location:
            directories += i + ' '
        directories = directories[0:-1]
        all_files = run_command(
            'find ' + directories + ' -type f -' + name + ' *' + ft)[0].rstrip().split("\n")
    else:
        if find_iname:
            raise AssertionError(
                'You are expecting case insensitive searching but you are not using \'use_find\' '
                'which allows this')
        from os import walk
        from os.path import join
        print('Finding all ' + ft + ' in ' + directory_location)
        all_files = []
        if not isinstance(directory_location, list):
            directory_location = [directory_location]
        for this_dir in directory_location:
            for root, dirs, files in walk(this_dir):
                for file in files:
                    if file.endswith(ft):
                        all_files.append(join(root, file))
    print('Found ' + str(len(all_files)) + ' of ' + ft + ' in ' + str(directory_location))
    return all_files


def float2str(floatval, num_decimals=2):
    """
    Convert a float (or a numy array) to a string with a given precision

    Args:
        floatval: The valueof the float
        num_decimals: Decimal points to use

    Returns:
        The string

    """
    floatval = np.atleast_1d(np.array(floatval, dtype=float)).flatten()
    conv_rule = lambda x: (
            "{0:." + str(num_decimals if x >= 0 else num_decimals - 1) + "f}").format(x)
    if floatval.size == 0:
        return ""
    elif floatval.size == 1:
        return conv_rule(floatval[0])
    else:
        stris = ''
        for i in floatval:
            stris += ', ' + conv_rule(i)
        return stris[1:]


def getfname():
    """
    Get the name of the calling function

    Returns:
        The name

    """
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    return calframe[1][3]


def matmax(alike):
    """
    Finds the maximum value and the index of it

    Args:
        alike: An iterable

    Returns:
        The maximum value
        The index of the maximum value

    """
    maxi = np.argmax(alike)
    maxv = alike[maxi]
    return [maxv, maxi]


def matmin(alike):
    """
    Finds the minimum value and the index of it

    Args:
        alike: An iterable

    Returns:
        The minimum value
        The index of the minimum value

    """
    mini = np.argmin(alike)
    minv = alike[mini]
    return [minv, mini]


def column_vector(alike):
    """
    Converts the input to a column vector (numpy array)

    Args:
        alike: Input

    Returns:
        The column vector

    """
    alike = np.atleast_1d(np.array(alike)).flatten()
    nelements = alike.size
    outmat = np.array(alike)
    outmat.shape = (nelements, 1)
    return outmat


def row_vector(alike):
    """
        Converts the input to a row vector (numpy array)

        Args:
            alike: Input

        Returns:
            The row vector

        """
    npa = np.atleast_1d(np.array(alike)).flatten()
    nelements = npa.size
    npa.shape = (1, nelements)
    return npa


def get_timestamp():
    """
    Generate a timestamp from the current time

    Returns:
        The timestamp as a string

    """
    from time import time
    timestamp = str(time())
    return timestamp


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    """
    Check if floats are equal to a given precision

    Args:
        a: Float to compare
        b: Float to compare
        rel_tol: Relative tolerance
        abs_tol: Absolute tolerance

    Returns:
        Yes/no

    """
    res = np.abs(a - b) <= np.maximum(rel_tol * np.maximum(np.abs(a), np.abs(b)), abs_tol)
    return res


if __name__ == '__main__':
    print('Your repo git hash: ' + get_git_hash())
