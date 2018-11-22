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
# This file performs the modelling of the acoustic environments provided with the ACE Challenge [2], using the 
# approach proposed in [1]. The model expresses the AIRs, which describe each acoustic environment, using a sparse 
# model for the early reflections and a stochastic model for the tail. The aim of the modelling is to compactly and 
# accurately describe the AIRs in a low dimensional space, which can be used by GANs to estimate a generative model 
# for the reverberation effect.
#
# A copy of the necessary files from the ACE database is provided with this repo. To unpack it do
# Code/Local_Databases/AIR$  tar zxf ACE16.tar.gz
# The corpus was published under 'Creative Commons Attribution-NoDerivatives 4.0 International Public License' and in the package you can find a copy of the license
#
# Usage:
#       bash ace_acenvgenmodeling.sh <the_save_location>
#
# Example:
#       bash ace_acenvgenmodeling.sh /tmp/modeling_results 
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

trap ctrl_c INT

function ctrl_c() {
    echo "Trapped waiting to finish..."
    wait
    exit 0
}

if [ "$#" -lt 1 ]; then
    echo 'Illegal number of parameters, expected >= 1. 
    1 ) Save location for results 
    2+) Arguments passed to acenvgenmodel_worker.py
    Example:
    bash ace_acenvgenmodeling.sh /tmp/modeling_results
    '
    exit 1
fi


save_base="$1"
shift

max_jobs=7

options=( --nomatlabinit   --fs 16000  --transition 0.024  --sparsefact 0.8 --lassocoef 0.8  --nodisplay  )

for array in Mobile  Crucif EM32 Chromebook Lin8Ch; do
    nchannels=-1
    case "$array" in
        Mobile)
            nchannels=2
            ;;
    
        Crucif)
            nchannels=4
            ;;
    
        EM32)
            nchannels=31
            ;;
    
        Chromebook)
            nchannels=1
            ;;
    
        Lin8Ch)
            nchannels=7
            ;;
        *)
            echo Unexpected condition
            exit 1
            ;;
    esac
    for room in 611 508 403a 503 EE_lobby 502 803; do
        for config in 1 2; do
            for channel in `seq 0 $nchannels`; do
                saveloc="$save_base/${array}_${room}_${config}_ch${channel}/";
                logfile="$saveloc/log.txt"
                echo "Doing : Room : $room and Array : $array Config : $config @ $logfile"
                mkdir -p "$saveloc"
                python_cmd='python'
                [ $(hostname) == 'sapthree' ] && python_cmd="$HOME/anaconda2/bin/python"
                stdbuf -i0 -o0 -e0 $python_cmd -u ./acenvgenmodel_worker.py \
                    --resultsdir $saveloc --array $array --room $room  \
                    --config $config --channel $channel ${options[@]} \
                    $* 2>&1 > $logfile  | \
                    grep -v 'Using TensorFlow backend.' &
                current_jobs=`jobs | grep -c Running`
                while [ $current_jobs -ge $max_jobs ]; do
                    sleep 2
                    current_jobs=`jobs | grep -c Running`
                done
            done
        done
    done
done
echo "Waiting for $(jobs | grep -c Running) jobs to finish"
wait

mkdir -p "$save_base/all_hats"
for i in `find  "$save_base" -name '*hat*.pdf' ! -name '*all_hats*'`; do
    cp $i "$save_base"/all_hats/`echo $i | rev | sed 's:/:_:'|rev|xargs basename`
done

suffixes_all=(' Final' ' AR' ' Truncation' ' Sparse')
suffix_model=('')
logfiles=($(find $save_base -name log.txt))

suffixes=${suffix_model[@]}
if grep -q 'nparams' <(echo $*); then
    echo "Detected that you want to test the baselines by fixing the number of parameters therefore will collect performance measures for baselines ${suffixes_all[*]}"
    suffixes=${suffixes_all[@]}
fi

get_measures () { sed -n "/Evaluation Measures$1/, /Evaluation Measures Done/ p"  | grep -v "Evaluation Measures$1" | grep -v "Evaluation Measures Done"; }

for eval_name_idx in `seq 0 $((${#suffixes[@]}-1))`; do
    eval_name="${suffixes[$eval_name_idx]}"
    results_file="$save_base/results_collection$(echo "$eval_name" | tr ' ' '_'| tr '[:upper:]' '[:lower:]').csv"
    echo "Saving results at $results_file for $eval_name getting info from ${logfiles[0]}"
    column_names=$(echo $(for i in `cat ${logfiles[0]} | get_measures "$eval_name" | cut -d',' -f1`; do echo $i ${i}_dB; done) | sed 's: :,:g')
    echo "Columns are : $column_names"

    echo "Name,Array,Room,Config,DRRdB,T60,NRef,$column_names" | tr -d ' '  > $results_file

    for thefile in ${logfiles[@]}; do
        entry=`echo $thefile | rev | cut -d'/' -f2 | rev | sed 's:EE_lobby:EE-lobby:g'`
        drr=$(grep 'DRR estimated as' $thefile | cut -d' ' -f7| sed 's:dB::')
        t60=$(grep 'T60 estimated as' $thefile | cut -d' ' -f7| sed 's:s::')
        nparams=$(grep  -B 1 'Saved' $thefile | tail -n 2 |head -n 1| sed "s/^[ \t]*//" | cut -d' ' -f1)
        nparams=$(($nparams+1))
        echo $entry "," `echo $entry | cut -d'_' -f1-3 |sed 's:_:,:g'` ",$drr,$t60,$nparams,"\
        "$(cat $thefile | get_measures "$eval_name" |  cut -d',' -f2-3 | xargs echo | sed 's: :,:g'|sed 's:,,:,:g')"  |\
        tr -d 'dB' |tr -d ' ' | sed 's:EE-lobby:EE_lobby:g' >> $results_file
    done
    echo "Done with $results_file"
done
