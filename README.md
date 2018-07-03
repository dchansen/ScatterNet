This repository contains the code and the trained model for the article **ScatterNet: a convolutional neural network for cone-beam CT intensity correction**.


**scatter_correct.py** - Scatter corrects the input mha files using the pretrained model. Please not that this is a proof of concept.

**train_network.py** - The code for training the model. This needs to be edited to load your datasets.

This code should not be used clinically without thorough testing, and even then no guarantees are made for correctness, usefulness or applicability.  