#!/bin/bash

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

##################################################################################################################
##################################################################################################################
#
# Description:
# This script uses the code provided with this repository to run the experiments described in [1].
# The code is able to train and evaluate DNNs for the task of 'Room Classification' using the data provided with
# the ACE challenge database [2]. The trained models are CNN-RNNs 
#
# A copy of the necessary files from the ACE database is provided with this repo. To unpack it do
# Code/Local_Databases/AIR$  tar zxf ACE16.tar.gz
# The corpus was published under 'Creative Commons Attribution-NoDerivatives 4.0 International Public License' and in the package you can find a copy of the license
#
# Usage:
#       bash run_cnnrnn_net.sh.sh <ACE data directory> <Speech directory> <ACE h5 data> <cache enable> <<ace_discriminative_nets.py args>>
#
#       <ACE data directory>: Location of ACE challenge data
#       <Speech directory> : Must include a TRAIN and a TEST subdirectory, with the corresponding speech files included. The provided script wav_concatenator.sh can be used to create longer speech utterances, which are used in this experiment. The experiment will use 5s of speech per AIR and will assume that speech utterances are longer. It uses offsetting of longer utterances as a primitive data augmentation method. The script has been successfully trailed with the TIMIT database. It creates an concatenation of all the wav files in a directory, for each directory. Since TIMIT has one directory per speaker it created a long utterance per speaker, ideal for the task.
#       <ACE h5 data>: Location of HFD5 dataset file for the ACE database, which is provided with this repository at Code/results_dir/ace_h5_info.h5. Contains information about the filenames, number of channels and also ground truth acoustic parameter values. If you want to create a new one, then use fe_utils.compile_ace_h5
#       <cache enable> : 0 or 1 to read any available caches
#
# Example:
#       bash run_cnnrnn_net.sh.sh ../Local_Databases/AIR/ACE16 ../results_dir/concWavs/concWavs/Local_Databases/speech/TIMIT/TIMIT/ ../results_dir/ace_h5_info.h5 0
#
#
# This file was original distributed in the repository at:
# {repo}
# If you use this code in your work, then cite [1].
#
# [1] C. Papayiannis, C. Evers, and P. A. Naylor, "End-to-End Classification of Reverberant Rooms using DNNs," arXiv preprint arXiv:1812.09324, 2018.
# [2] http://www.ee.ic.ac.uk/naylor/ACEweb/index.html
#
#################################################################################################################
#################################################################################################################

set -e

if [ "$#" -lt 5 ]; then
    echo 'Illegal number of parameters, expected >= 5. 
    1) ACE dir (/media/cp510/ExtraGiaWindows/db_exp_data/Local_Databases/AIR/ACE16) 
    2) speech dir (/home/cp510/GitHub/base_git_repo/Code/results_dir/concWavs) 
    3) h5 loc (/home/cp510/GitHub/base_git_repo/Code/results_dir/ace_h5_info.h5) 
    4) 0/1 whether cache should be read or not. 
    5+)Passed to ace_discriminative_nets.py 
    Example:
    run_cnnrnn_net.sh.sh ../Local_Databases/AIR/ACE16 ../results_dir/concWavs/ ../results_dir/ace_h5_info.h5 1 
    '
    exit 1
fi

python_loc=$HOME/anaconda2/bin/python
$python_loc utils_base.py

ace_dir=$1
speech_dir=$2
h5_loc=$3
force_readcache=$4
shift
shift
shift
shift
extras=$@

base_scrap=/tmp/
if [ `hostname | cut -d'-' -f1` == 'login' ]; then
    base_scrap=$WORK
fi

save_loc_base=$base_scrap/ace_discriminative_nets_eval/

tmp_args=( ${extras[@]} )
args_speech=( --utts 20  --experiment room  --ace $ace_dir --h5 $h5_loc  --speech $speech_dir/TRAIN $speech_dir/TEST   --cacheloc $base_scrap ${tmp_args[@]})
args_air=( --experiment room --ace $ace_dir --h5 $h5_loc    --cacheloc $base_scrap  ${tmp_args[@]})
# Change this to speech if you want to do speech
final_args=${args_air[@]}

echo Args : ${args_final[*]}

    
echo Running Speech CNN RNN
this_saveloc=$save_loc_base/speech/speech_cnn_rnn
mkdir -p $this_saveloc
$python_loc -u ace_discriminative_nets.py --saveloc $this_saveloc ${final_args[*]} --cnn --rnn $readcache_arg | tee $this_saveloc/log.txt



