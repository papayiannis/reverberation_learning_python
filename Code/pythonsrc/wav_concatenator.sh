#!/bin/bash

# Copyright 2017 Constantinos Papayiannis
#
# This file is part of Matlab and Reverberation Learning Toolbox for Python.
#
# Matlab and Reverberation Learning Toolbox for Python is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Matlab and Reverberation Learning Toolbox for Python is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Matlab and Reverberation Learning Toolbox for Python.  If not, see <http://www.gnu.org/licenses/>.

##################################################################################################################
##################################################################################################################
#
# Description:
# Given a location, the structure within is replicated in a target location and all the wav files within each directory are concatenated to a single wav file. This is used for example in the cases where speech database contain many small audio files as speech utterances and each folder is a different speaker, in order to create a long speech utterance from a single speaker
#
# Usage:
# bash wav_concatenator.sh <original_location> <target_location> 
# 
# Example:
# bash wav_concatenator.sh ../../TIMIT/TRAIN ./TIMIT_TRAIN_CONCS
#
# This file was original distributed in the repository at:
# {repo}
#
#################################################################################################################
#################################################################################################################

sox_flags=(  )
# if doing SPHERE files then uncomment this
# sox_flags=( -t sph )

set -e


if [ "$#" -lt 2 ]; then
    echo 'Illegal number of parameters, expected 2. 
    1) Original location which contains wav files in the structure below it 
    5) Target directory 
    Example:
    bash wav_concatenator.sh ../../TIMIT/TRAIN ./TIMIT_TRAIN_CONCS 
    '
    exit 1
fi

target=`realpath $2`
base=`realpath $1`

run_dir=`pwd`
cd $base
dirs=(`find ./ -type d -mindepth 1`)
for i in ${dirs[@]}; do
    mkdir -p $target/$i
    target_file=$target/$i/concWav.wav
    wavs=( `find $i -maxdepth 1 -iname '*.wav'` )
    if [ ${#wavs[@]} -gt 0 ]; then
        echo Doing $base/$i
        sox ${sox_flags[*]} ${wavs[@]} $target_file
        echo Wrote $target_file from `realpath $i` with duration `soxi -d $target_file`s, from ${wavs[*]}
    fi
done

cd $run_dir


