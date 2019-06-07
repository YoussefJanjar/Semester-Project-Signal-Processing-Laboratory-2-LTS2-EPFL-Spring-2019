# Semester project

This repository contains the python code for DOA of sound classification using deep learning, more specifically a CNN architecture.The Training and Testing sets are generated using the Pyroomacoustics package. 
In this project we will try to reproduce a convolution neural network (CNN) based classification method for broadband DOA estimation, where the phase component of the short-time Fourier transform coefficients of the received microphone signals are directly fed into the CNN and the features required for DOA estimation are learned during training. Since only the phase component of the input is used, the CNN can be trained with synthesized noise signals, thereby making the preparation of the training data set easier compared to using speech signals. Through experimental evaluation, the ability of the proposed noise trained CNN framework to generalize to speech sources is demonstrated. We will try to verify the results of the state of the art work on the subject.

## Installation
The python dependencies can be installed by using the requirements file 
```
pip install -r requirements.txt
```

## Usage

You can now run the script
```
python cnn_test_main.py
```