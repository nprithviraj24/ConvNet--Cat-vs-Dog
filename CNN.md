# Convolutional Neural Network
Concoction of different articles based on Internet.

### Prerequisites
- Basic understanding of Neural Networks.
- Decent programming skills in Python.
- Flexibility with Deep Learning frameworks.

### Questions that needs to be answered.
- Why CNN(or ConvNet)?
- What are images, what do they consist of?
- What is convolution layer, how is it different from normal Neural network? Explain with different parts in CNN.
- Since its same as Neural Networks, what are different layers that are involved in CNN?
- What are pooling layers?
- What are normalization layer?
- What happens in Fully Connected Layer?


#### What is Convolutional Neural Network(CNN) and why do we need it?
CNN or Convolutional Neural Network or **ConvNet** are similar to Neural Networks(hereafter, NN), in fact it can be formally called as application of Neural Networks. Same as NN, CNN are made of neurons that have learnable wights and biases. Each neuron recieves some inputs, performs a **dot product** and optionally follows it with a non-linearity.
    CNN are category of NN that have been proven very effective in areas such as image recognition and classification. ConvNets have been successful in indentifying faces, objects and traffic signs apart from powering vision in robots and self driving cars.

#### What are images, what do they consist of?
Such a naive question, isn't it? Images are input to CNN, so there is paramount need to understand the intracacies in images. CNNs operate over Volumes of images.
Unlike neural networks, where the input is a vector, here the input is a multi-channeled image (3 channeled in this case). 
<br />
A small sample on how images are used: 

![Image](https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-07-at-4-59-29-pm.png?w=748)
<br />

Essentially, every image can be represented as a matrix of pixel values.
**Channel** is a conventional term used to refer to a certain component of an image. An image from a standard digital camera will have three channels – red, green and blue – you can imagine those as three 2d-matrices stacked over each other (one for each color), each having pixel values in the range 0 to 255. Example for Channeled image: 
<br />
![Channel image](https://static1.squarespace.com/static/54856bade4b0c4cdfb17e3c0/t/57edf15c9f74563967b893a2/1475211614805/?format=750w)
<br />
A **grayscale** image, on the other hand, has just one channel. For the purpose of this post, we will only consider grayscale images, so we will have a single 2d matrix representing an image. The value of each pixel in the matrix will range from 0 to 255 – zero indicating black and 255 indicating white. 
<br/>
Example for Grayscale image: 
<p align="center">
    <img src ="https://ujwlkarn.files.wordpress.com/2016/08/8-gif.gif?w=192&h=192" />
</p>