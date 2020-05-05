My model produces an accuracy of about 95 after running and resetting. I studied the Keras CNN model example and based my model off of that. I used batch normalization to reduce the amount of covariance shift, the amount by what the hidden unit values shift around. 

1.What is one-hot encoding? Why is this important and how do you implement it in keras?

One-hot encoding is defined as a representation of categorical variables as binary vectors; it is critical because many ML algorithms cannot operate on the label data directly, and instead require input variables to be numeric (this is in part why one-hot encoding improves the ML algorithm’s ability to predict).  In the Keras library there is a function to_categorical() used to hot encode integer data, and if the sequence starts at 0 and is not representative of all values, you can specify num_classes in the argument of the function. 

2.What is dropout and how does it help overfitting?
Dropout drops the outputs from a layer of a neural network, and as a result, in each layer neurons are able to learn several characteristics of the neural network. It is a form of regularization and constrains the network’s adaptation to the data at training, preventing it from becoming too smart in learning the input data; this is why it helps to avoid overfitting.  

3.How does ReLU differ from the sigmoid activation function?
*Def ReLU is h = max(0, a) where a = Wx + b

While the sigmoid exists between 0 and 1 (used for models where you have to predict the probability as an output), ReLU has a range [0 to infinity), and any negative input given to the function turns the value into zero, and as a result the negative values are not mapped properly. Most importantly the sigmoid causes a vanishing gradient (the derivative of the sigmoid function is always smaller than one), and the ReLU function does not because the gradient is either 0 for a < 0 or 1 for a > 0. 

4.Why is the softmax function necessary in the output layer?
The final layer of a neural network is where the model generates the final activations that will be interpreted as probabilities for classification purposes. Softmax, a logistic activation function, guarantees that the final activations all add up to 1 so they can be interpreted as probabilities. 

5.This is a more practical calculation. Consider the following convolution network:
a.Input image dimensions = 100x100x1
b.Convolution layer with filters=16 and kernel_size=(5,5)c.
MaxPooling layer with pool_size = (2,2)
What are the dimensions of the outputs of the convolution and max pooling layers?

Convolution:
Output height is equal to input height - filter + 2(padding)/stride + 1 
(100 - 5 + 0/1) + 1 = 96 

Output width is equal to input width - filter + 2(padding)/stride + 1 
(100 - 5 + 0/1) + 1 = 96 

Depth = K = 16 

The dimensions of the outputs of the convolution layer are 96 x 96 x 16 

Max Pooling:
Height = (input height - F)/stride + 1

96 - 5/1 + 1 = 92

Width = (input width - F)/stride + 1 
96 - 5/1 + 1 = 92 

Depth2 = Depth1 = 16 

The dimensions of the outputs of the pooling layer are 92 x 92 x 16 

