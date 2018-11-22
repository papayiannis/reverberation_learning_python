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

# Give the following arguments
# Location of saved modeling results for ace which contains the result in the form the-location/the-name-of-the-environment-containing-the-room-name/log.txt
# Scratchpad location

##################################################################################################################
##################################################################################################################
#
# Description:
# 
# This script collects the results of modeling ACE AIRs [2] using the proposed model in [1]. Then these results are 
# used to train GANs for the rooms in ACE. This will result in 1 GAN for each of the 7 rooms, part of the ACE
# challenge measurements. Each GAN is trained using data for each room and then produces 100 acoustic environment 
# instances from each model. These instances are to be used for training DNNs, providing a method for data
# augmentation. 
#
# To collect the results prior to running this step, run:
# bash ace_acenvgenmodeling.sh /tmp/modeling_results
#
# Usage:
#       bash gan_model_worker.sh <ace_acenvgenmodeling results location> <GAN results location to use>
#
# Example:
#       bash gan_model_worker.sh /tmp/modeling_results/ /tmp/gan_results/
#
#
# This file was original distributed in the repository at:
# {repo}
# If you use this code in your work, then cite [1].
#
# [1] C. Papayiannis, C. Evers, and P. A. Naylor, "Data Augmentation using GANs for the Classification of Reverberant Rooms," (to be submitted), 2019
# [2] http://www.ee.ic.ac.uk/naylor/ACEweb/index.html
#
#################################################################################################################
#################################################################################################################


set -e 

if [ "$#" -lt 2 ]; then
    echo 'Illegal number of parameters, expected >= 2. 
    1) The location where the logs of ace_acenvgenmodeling.sh results were saved
    2) The location where the results and the data augmentation h5 dataset will be saved
    3+) Arguments passed to gan_model.py
    Example:
    bash gan_model_worker.sh /tmp/modeling_results/ /tmp/gan_results/
 
    '
    exit 1
fi

ace_results=$1
saveloc=$2
test_array='Mobile'

shift
shift

mkdir -p $saveloc

for i in 611 403a 803 503 502 508 EE_lobby; do 
    $HOME/anaconda2/bin/python -u acenvgenmodel_collect_results.py  ` find $ace_results/*${i}* -name log.txt  | grep -v "$test_array"` --saveloc $saveloc/ref_rep_$i/
    log_dest=$saveloc/log_${i}.txt
    echo Working for room $i and saving log at $log_dest
    $HOME/anaconda2/bin/python -u gan_model.py --h5 $saveloc/ref_rep_$i/reflection_y_data.h5 --saveloc $saveloc/gan_$i/ --airname GAN_${i}_%d_RIR.wav  --nodisplay $* | tee $log_dest 
done

aug_h5=$saveloc/gan_aug_data.h5
python -c "from fe_utils import compile_ace_h5; compile_ace_h5('$saveloc','$aug_h5',ft='RIR.wav');"
echo Saved data augmentation AIR dataset at $aug_h5

echo All done
