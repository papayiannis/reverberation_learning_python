# Reverberation Learning Toolbox for Python

_Copyright 2018 [Constantinos Papayiannis](https://www.linkedin.com/in/papayiannis/)_  
_Reverberation Learning Toolbox for Python is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version._

# Introduction

Training DNNs has been a huge step in the modelling of complex relationships between the observed world and its high level properties, which we use to describe it. Their contribution to the field of image-object recognition, speech recognition and language processing has pushed the boundaries o state-of-the-art. Their potential benefits have not been thoroughly discussed in the literature however in terms of their potential benefits in the modelling and classification of the reverberation effect. Some works have looked into dereverberation using DNNs, however working with the motivation to _understand_ the effect has been almost non-existent. 

The work in [@placeholder] has been one of the first steps in using deep learning to understand the underlying model of the reverberation effect. This repository includes the code that was used to classify reverberant acoustic environments in terms of rooms. It also contains useful code for DSP, speech processing and deep learning using Keras.

# Description

This repository offers a toolbox which allows the user to

* Prepare reverberant environment training data for DNN classifiers
  * Can create directly data from AIRs.
  * Can create reverberant speech data, combining the AIRs and anechoic speech.
  * Flexibility over classification tasks considered. 
    * Can provide labels for rooms, position, room type and microphone arrays.
* Train DNNs for room classification with configurable architectures. Supports:
  * FF Networks
  * CNNs
  * RNNs
  * Bidirectional RNNs
  * CNN-RNNs
* Provides dedicated routines for collecting data from the ACE challenge corpus [@Eaton2015].
* Provides scripts for running the experiments of [@placeholder].
* Provides reusable and useful routines for
	* CNN filter kernel and feature map visualizations using audio. 
	* Keras model learning
		* Batch Generation
		* Batch balancing
		* Multi-purpose and customizable callbacks 
	* Signal processing and signal manipulation.
	* Reverberation and acoustic parameter estimation.
  
# Setup

To use the repository start by setting up your environment. I assume you have anaconda and you are working with Python2.7. The code has not been checked with Python3.

```bash
# Get the repository
git clone https://github.com/papayiannis/reverb_learning
cd reverb_learning
# Get dependencies
conda install numpy keras scipy tabulate matplotlib pandas seaborn h5py scikit-learn
# Unpack the AIR data
cd Code/Local_Databases/AIR
tar zxf ACE16.tar.gz 
cd ../../pyhtonsrc
# Run the training example for a CNN-RNN room classifier using ACE AIRs
bash run_ace_discriminative_nets.sh ../Local_Databases/AIR/ACE16 \
  /tmp/ ../results_dir/ace_h5_info.h5 0 4 
```

# Room classification

The work in [@placeholder] discussed methods and strategies for training DNNs for room classification of reverberant acoustic environments. 

## AIRs

The first set of experiments looked at how AIRs can be used to train DNNs and how they can be used for inference. The 4 candidate architectures for the task, confided in the paper are shown below.

![AIR](doc/figures/room_dnn/all_air.png)

To train all these networks in sequence, do the following  

```bash
# Unpack the AIR data
cd Code/Local_Databases/AIR
tar zxf ACE16.tar.gz 
cd ../../pyhtonsrc
# Run the training example for a CNN-RNN room classifier using ACE AIRs
bash run_ace_discriminative_nets.sh ../Local_Databases/AIR/ACE16 \
  /tmp/ ../results_dir/ace_h5_info.h5 0 1 2 3 4 
```
The indices 1, 2, 3 and 4 refer respectively to the FF, CNN, RNN and CNN-RNN models.

## Speech

To perform the following task from reverberant speech, the corresponding 4  architectures for the task are

![AIR](doc/figures/room_dnn/all_speech.png)

To train all these networks in sequence, do the following  

```bash
# Unpack the AIR data
cd Code/Local_Databases/AIR
tar zxf ACE16.tar.gz 
cd ../../pyhtonsrc
mkdir -p /tmp/train_test_speech
ln -s $TRAIN_SPEECH_LOC /tmp/train_test_speech/TRAIN
ln -s $TEST_SPEECH_LOC /tmp/train_test_speech/TEST
# Run the training example for a CNN-RNN room classifier using ACE AIRs and your speech files
bash run_ace_discriminative_nets.sh ../Local_Databases/AIR/ACE16 \
  /tmp/train_test_speech/ ../results_dir/ace_h5_info.h5 0 1 2 3 4 
```  

The indices 1, 2, 3 and 4 again refer respectively to the FF, CNN, RNN and CNN-RNN models. The locations ```$TRAIN_SPEECH_LOC``` and ```$TEST_SPEECH_LOC``` contain respectively locations where speech wav files are included, for training and for testing of the trained DNNs. The experiments have used TIMIT, however this is not free, so I cannot provide it here. You can use any other speech data you want.



# Abbreviations

AIR: Acoustic Impulse Response  
DNN: Deep Neural Network  
CNN: Convolutional Neural Network  
RNN: Recurrent Neural Network  
FF: Feed Forward

# Bibliography  

[@placeholder]: {placeholder} p  
[@Eaton2015]: J. Eaton; N. D. Gaubitch; A. H. Moore; P. A. Naylor, "Estimation of room acoustic parameters: The ACE Challenge," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 24, no.10, pp.1681-1693, Oct. 2016.









