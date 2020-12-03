
# Reverberation Learning Toolbox for Python

Copyright 2019 [Constantinos Papayiannis](https://www.linkedin.com/in/papayiannis/)  
  
# Introduction

The work in [1] has been one of the first steps in using deep learning to classify reverberant environments based on their acoustics. DNNs were provided with reverberant speech signals and the signals were classified in terms of the room where the recording was made in. This repository includes the code that was used in [1]. It is intended to be used by other researchers that aim to build on the work and follow the many promising research paths that stem from it. It also contains useful code for DSP, speech processing and deep learning using Keras.

  
# Setup

To use the repository start by setting up your environment. I assume you have Anaconda and you are working with Python 3.   

```bash
# Get the repository
git clone https://github.com/papayiannis/reverberation_learning_python
cd reverberation_learning_python
# Get dependencies
conda install numpy keras scipy tabulate matplotlib pandas seaborn h5py scikit-learn
# Unpack the AIR data
cd Code/Local_Databases/AIR
tar zxf ACE16.tar.gz 
```

# Room classification


To train a DNN for room classification from reverberant speech, do the following  

```bash
# Unpack the AIR data
cd Code/Local_Databases/AIR
tar zxf ACE16.tar.gz 
cd ../../pythonsrc
mkdir -p /tmp/train_test_speech
ln -s $TRAIN_SPEECH_LOC /tmp/train_test_speech/TRAIN
ln -s $TEST_SPEECH_LOC /tmp/train_test_speech/TEST
# Run the training example for a CNN-RNN room classifier using ACE AIRs and your speech files
bash run_ace_discriminative_nets.sh ../Local_Databases/AIR/ACE16 \
  /tmp/train_test_speech/ ../results_dir/ace_h5_info.h5 0 5 
```  

The index 8 choses an Attention-CRNN. The locations ```$TRAIN_SPEECH_LOC``` and ```$TEST_SPEECH_LOC``` contain respectively locations where speech wav files are included, for training and for testing of the trained DNNs. The experiments have used TIMIT but any other dataset can be used in practice. 

# Bibliography  

[1]: C. Papayiannis, C. Evers and P. A. Naylor, "End-to-End Classification of Reverberant Rooms Using DNNs," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 28, pp. 3010-3017, 2020, doi: 10.1109/TASLP.2020.3033628.

  
_Reverberation Learning Toolbox for Python is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version._









