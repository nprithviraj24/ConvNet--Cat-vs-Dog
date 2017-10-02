# Convolutional Neural Network
Concoction of different articles based on ConvNet.

## Contents
- [Prerequisites](https://github.com/nprithviraj24/ConvNet--Cat-vs-Dog/blob/master/CNN.md#prerequisites-1)
- [Questions](https://github.com/nprithviraj24/ConvNet--Cat-vs-Dog/blob/master/CNN.md#questions-that-needs-to-be-answered)
- [Glossary](https://github.com/nprithviraj24/ConvNet--Cat-vs-Dog/blob/master/CNN.md#glossary-1)
- [Additional Resources](https://github.com/nprithviraj24/ConvNet--Cat-vs-Dog/blob/master/CNN.md#additional-resources)


## Prerequisites
- Basic understanding of Neural Networks.
- Decent programming skills in Python.
- Flexibility with Deep Learning frameworks.

## Questions that needs to be answered:
- [Why CNN(or ConvNet)?](https://github.com/nprithviraj24/ConvNet--Cat-vs-Dog/blob/master/CNN.md#what-is-convolutional-neural-networkcnn-and-why-do-we-need-it)
- [What are images, what do they consist of?](https://github.com/nprithviraj24/ConvNet--Cat-vs-Dog/blob/master/CNN.md#what-are-images-what-do-they-consist-of)
- [What is convolution layer, how is it different from normal Neural network?](https://github.com/nprithviraj24/ConvNet--Cat-vs-Dog/blob/master/CNN.md#what-is-convolution-layer-how-is-it-different-from-normal-neural-network-explain-with-different-parts-in-cnn) 
- Since its same as Neural Networks, what are different layers that are involved in CNN?
- What are pooling layers?
- What are normalization layer?
- What happens in Fully Connected Layer?

<br />
<br />

### What is Convolutional Neural Network(CNN) and why do we need it?
CNN or Convolutional Neural Network or **ConvNet** are similar to Neural Networks(hereafter, NN), in fact it can be formally called as application of Neural Networks. Same as NN, CNN are made of neurons that have learnable wights and biases. Each neuron recieves some inputs, performs a **dot product** and optionally follows it with a non-linearity.
    CNN are category of NN that have been proven very effective in areas such as image recognition and classification. ConvNets have been successful in indentifying faces, objects and traffic signs apart from powering vision in robots and self driving cars.

### What are images, what do they consist of?
Such a naive question, isn't it? Images are input to CNN, so there is paramount need to understand the intracacies in images. CNNs operate over Volumes of images.
Unlike neural networks, where the input is a vector, here the input is a multi-channeled image (3 channeled in this case). 
<br />
A small sample on how images are used: 

![Image](https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-07-at-4-59-29-pm.png?w=748)
<br />

Essentially, every image can be represented as a matrix of pixel values. <br />
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

<br />
<br />

### What is convolution layer, how is it different from normal Neural network?
The main innovation of the convolutional neural network is the **Convolutional layer.** A convolution layer applies a set of **Sliding windows** across an image. These sliding windows are termed filters, and they detect different primitive shapes or patterns. The primary purpose of Convolution in case of a ConvNet is to **extract features** from the input image. 
<br />
<p align="center">
    <img src="https://static1.squarespace.com/static/54856bade4b0c4cdfb17e3c0/t/57eded43440243e527d246a7/1475213244328/?format=500w">
</p>
    <div align="center" font-size="75%">
    The filter on the left might activate strongest when it encounters a horizontal line; the one in the middle for a vertical line.
    </div>
<br />
<br />
In the convolution layer, the filters are passed across the input, row by row, and they activate when they detect their shape. Now, rather than treating each of the **Height** x **Width** x **Depth** pixel values in isolation, the "windows" treat them in small local groups. These sliding filters are how the CNN can learn meaningful features and locate them in any part of the image.
<br />
Convolution preserves the spatial relationship between pixels by learning image features using small squares of input data. Here is the pictorial representation: 

<p align="center">
    <img src="http://deeplearning.stanford.edu/wiki/images/6/6c/Convolution_schematic.gif">
</p>
<br />

We slide the orange matrix over our original image (green) by 1 pixel (also called ‘stride’) and for every position, we compute element wise multiplication (between the two matrices) and add the multiplication outputs to get the final integer which forms a single element of the output matrix (pink). Note that the 3×3 matrix “sees” only a part of the input image in each stride.
<br />

So technically, we call this orange matrix as the **filter**, **kernal** or **feature detector.** 
<br />
<br />
### What are the different layers in CNN? Brief them.

Purpose: A ConvNet is made up of Layers. Every Layer has a simple API: It transforms an input 3D volume to an output 3D volume with some differentiable function that may or may not have parameters.

Layers used to build ConvNets

As we described above, a simple ConvNet is a sequence of layers, and every layer of a ConvNet transforms one volume of activations to another through a differentiable function. We use three main types of layers to build ConvNet architectures: Convolutional Layer, Pooling Layer, and Fully-Connected Layer (exactly as seen in regular Neural Networks). We will stack these layers to form a full ConvNet architecture.

Example Architecture: Overview. We will go into more details below, but a simple ConvNet for CIFAR-10 classification could have the architecture [INPUT - CONV - RELU - POOL - FC]. In more detail:

INPUT [32x32x3] will hold the raw pixel values of the image, in this case an image of width 32, height 32, and with three color channels R,G,B.

#### There are four main operations in the ConvNet shown in Figure 3 above:

* Convolution: CONV layer will compute the output of neurons that are connected to local regions in the input, each computing a dot product between their weights and a small region they are connected to in the input volume. This may result in volume such as [32x32x12] if we decided to use 12 filters.
* Non Linearity (ReLU) : RELU layer will apply an elementwise activation function, such as the <strong>max(0,x)</strong> thresholding at zero. This leaves the size of the volume unchanged ([32x32x12]).
* Pooling or Sub Sampling: POOL layer will perform a downsampling operation along the spatial dimensions (width, height), resulting in volume such as [16x16x12].
* Classification (Fully Connected Layer) : FC (i.e. fully-connected) layer will compute the class scores, resulting in volume of size [1x1x10], where each of the 10 numbers correspond to a class score, such as among the 10 categories of CIFAR-10. As with ordinary Neural Networks and as the name implies, each neuron in this layer will be connected to all the numbers in the previous volume.

<p align="center">
    <img src="http://adilmoujahid.com/images/cnn-architecture.png">
</p>
<br />

In this way, ConvNets transform the original image layer by layer from the original pixel values to the final class scores. Note that some layers contain parameters and other don’t. In particular, the CONV/FC layers perform transformations that are a function of not only the activations in the input volume, but also of the parameters (the weights and biases of the neurons). On the other hand, the RELU/POOL layers will implement a fixed function. The parameters in the CONV/FC layers will be trained with gradient descent so that the class scores that the ConvNet computes are consistent with the labels in the training set for each image

#### Briefing each of them:

##### Convolution Layer
 Let's take a following example to explain convolution layer in practice, for educational purpose we are considering `grayscale` images only.
<br /> 
![GIF](/giphy.gif)
<br />

A **filter** (with red outline) slides over the input image (convolution operation) to produce a feature map. The convolution of another filter (with the green outline), over the same image gives a different feature map as shown. It is important to note that the Convolution operation captures the local dependencies in the original image. Also notice how these two different filters generate different feature maps from the same original image. Remember that the image and the two filters above are just numeric matrices as we have discussed above.

In practice, a CNN learns the values of these filters on its own during the training process (although we still need to specify parameters such as number of filters, filter size, architecture of the network etc. before the training process). The more number of filters we have, the more image features get extracted and the better our network becomes at recognizing patterns in unseen images.
<br />

##### Introduction to Non-Linearity:

ReLU is an element wise operation (applied per pixel) and replaces all negative pixel values in the feature map by zero. The purpose of ReLU is to introduce **non-linearity** in our ConvNet, since most of the real-world data we would want our ConvNet to learn would be non-linear (Convolution is a linear operation – element wise matrix multiplication and addition, so we account for non-linearity by introducing a non-linear function like ReLU).

##### Pooling

After obtaining features using convolution, we would next like to use them for classification. In theory, one could use all the extracted features with a classifier such as a softmax classifier, but this can be computationally challenging. Consider for instance images of size 96x96 pixels, and suppose we have learned 400 features over 8x8 inputs. Each convolution results in an output of size (96−8+1)∗(96−8+1)=7921(96−8+1)∗(96−8+1)=7921, and since we have 400 features, this results in a vector of 892∗400=3,168,400892∗400=3,168,400 features per example. Learning a classifier with inputs having 3+ million features can be unwieldy, and can also be prone to over-fitting.

To address this, first recall that we decided to obtain convolved features because images have the “stationarity” property, which implies that features that are useful in one region are also likely to be useful for other regions. Thus, to describe a large image, one natural approach is to aggregate statistics of these features at various locations. For example, one could compute the mean (or max) value of a particular feature over a region of the image. These summary statistics are much lower in dimension (compared to using all of the extracted features) and can also improve results (less over-fitting). We aggregation operation is called this operation ”‘pooling”’, or sometimes ”‘mean pooling”’ or ”‘max pooling”’ (depending on the pooling operation applied)..

##### Padding

Now, let’s take a look at padding. Before getting into that, let’s think about a scenario. What happens when you apply three 5 x 5 x 3 filters to a 32 x 32 x 3 input volume? The output volume would be 28 x 28 x 3. Notice that the spatial dimensions decrease. As we keep applying conv layers, the size of the volume will decrease faster than we would like. In the early layers of our network, we want to preserve as much information about the original input volume so that we can extract those low level features. Let’s say we want to apply the same conv layer but we want the output volume to remain 32 x 32 x 3. To do this, we can apply a zero padding of size 2 to that layer. Zero padding pads the input volume with zeros around the border. If we think about a zero padding of two, then this would result in a 36 x 36 x 3 input volume.

<p align="center">
    <img src="https://adeshpande3.github.io/assets/Pad.png">
</p>
<br />

If you have a stride of 1 and if you set the size of zero padding to

<p align="center">
 <img src="https://adeshpande3.github.io/assets/ZeroPad.png">
</p>
<br />
where K is the filter size, then the input and output volume will always have the same spatial dimensions.

The formula for calculating the output size for any given conv layer is

<p align="center">
 <img src="https://adeshpande3.github.io/assets/Output.png">
</p>
<br /> 
where O is the output height/length, W is the input height/length, K is the filter size, P is the padding, and S is the stride.
<br />



## Glossary

##### Depth :
 Depth corresponds to the number of filters we use for the convolution operation. In the network shown in Figure 7, we are performing convolution of the original boat image using three distinct filters, thus producing three different feature maps as shown. You can think of these three feature maps as stacked 2d matrices, so, the ‘depth’ of the feature map would be three.
<br />
##### Stride: 
Stride is the number of pixels by which we slide our filter matrix over the input matrix. When the stride is 1 then we move the filters one pixel at a time. When the stride is 2, then the filters jump 2 pixels at a time as we slide them around. Having a larger stride will produce smaller feature maps.
<br />
##### Zero-padding:
 Sometimes, it is convenient to pad the input matrix with zeros around the border, so that we can apply the filter to bordering elements of our input image matrix. A nice feature of zero padding is that it allows us to control the size of the feature maps. Adding zero-padding is also called wide convolution, and not using zero-padding would be a narrow convolution.
 
 ## Additional resources.
 
 - [CNN for Dummies](https://medium.com/technologymadeeasy/for-dummies-the-introduction-to-neural-networks-we-all-need-c50f6012d5eb)
 - [Most comprehensive guide for CNN](http://cs231n.github.io/convolutional-networks/)
 - [Backpropogation in Convolutional Neural Network](http://jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/)
 - [TensorFlow implementation](http://www.subsubroutine.com/sub-subroutine/2016/9/30/cats-and-dogs-and-convolutional-neural-networks)
 


