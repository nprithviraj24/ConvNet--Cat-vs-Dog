## Keras

This section apprises my understanding of Keras, a deep learning framework.

### Sequential
    - Linear stack of array.

#### API
    .add() 
    [Referred from](https://keras.io/layers/core/)
    This add function is responsible for adding layers to the stack of array:
        ##### - **Conv2D:** Convolution layer- 1D and 2D
            For 2D, we have to specify the number of filters and the size of the filter.
        ##### - Dense: 
            a dense layer is simply a layer where each unit or neuron is connected to each neuron in the next layer.
            Note: Used for making the layer fully connected.
        ##### - MaxPooling2D:  As the name suggests, the output of it is pooling of input layer. Max pooling with a given pool size in the parameters.
        ##### - Activation:
            The type of activation function applied to the eah pixel value. There are different activation function available in keras such as sigmoid, relu etc, where relu is most preferred.
        **Note**
        ##### - Flatten:
            Dimensionality reduction. Doesn't affect the batch size. 

