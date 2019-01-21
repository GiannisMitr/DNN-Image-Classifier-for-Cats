# DNN Image Classifier for Cats
A deep neural network with configurable number of hidden layers that classifies cat vs non-cat Images.
An L layer network is created, with RELU activations in all layers except for the output layer that uses a  SIGMOID.   
Regularization, momentum and mini-batch techniques are NOT used.
Implemented using NumPy.

## Usage

Execute                      *cat_image_classifier.py*, by configuring *layers_dims* array you can set the number of layers and their size.     
By setting
```layers_dims = [12288, 20, 15, 10, 5, 1] ```
We create a 5 hidden layer network, with 12288 nodes on input layer, 20 nodes on layer one and so on.

The code will:
* Train the defined model for configurable number of iterations
* Use trained parameters to compute accuracy for train and test datasets
* Try to predict the label on a custom image provided by the user.

Inspiration and help from Coursera deeplearning specialization