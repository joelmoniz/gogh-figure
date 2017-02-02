# gogh-figure  
  
Fast, Lightweight Style Transfer using Deep Learning: A re-implementation of "A Learned Representation For Artistic Style" (which proposed using Conditional Instance Normalization), "Instance Normalization: The Missing Ingredient for Fast Stylization", and the fast neural-style transfer method proposed in "Perceptual Losses for Real-Time Style Transfer and Super-Resolution" using Lasagne and Theano.

## Results

### Conditional Instance Normalization
This repository contains a re-implementation of the paper [A Learned Representation For Artistic Style](https://arxiv.org/abs/1610.07629) and its [Google Magenta TensorFlow implementation](https://github.com/tensorflow/magenta/tree/master/magenta/models/image_stylization). The major differences are as follows:

1. The batch size has been changed (from 16 to 4); this was found to reduce training time without affecting the quality of the images generated.
2. Training is done with the COCO dataset, as opposed to with ImageNet
3. The style loss weights have been divided by the number of layers used to calculate the loss (though the values of the weights themselves have been increased so that the actual weights effectively remain the same)

The following are the results when this technique was applied to style images described in the paper (to generate pastiches of a set of 32 paintings by various artists, and of 10 paintings by Monet, respectively):

#### Misc. 32

<p align='center'>
  <img src='data/readme/im1.jpg' title='Joel Moniz' alt='Joel Moniz' height="225px">    
</p>
<p align='center'>
  <img src='data/readme/cin/misc32/im0.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im1.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im2.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im3.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im4.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im5.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im6.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im7.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im8.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im9.jpg' height="210px">   
  <img src='data/readme/cin/misc32/im10.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im11.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im12.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im13.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im14.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im15.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im16.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im17.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im18.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im19.jpg' height="210px">   
  <img src='data/readme/cin/misc32/im20.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im21.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im22.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im23.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im24.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im25.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im26.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im27.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im28.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im29.jpg' height="210px">   
  <img src='data/readme/cin/misc32/im30.jpg' height="210px">    
  <img src='data/readme/cin/misc32/im31.jpg' height="210px">  
</p>

#### Monet 10

<p align='center'>
  <img src='data/readme/cin/monet10/im0.jpg' height="210px">    
  <img src='data/readme/cin/monet10/im1.jpg' height="210px">    
  <img src='data/readme/cin/monet10/im2.jpg' height="210px">    
  <img src='data/readme/cin/monet10/im3.jpg' height="210px">    
  <img src='data/readme/cin/monet10/im4.jpg' height="210px">    
  <img src='data/readme/cin/monet10/im5.jpg' height="210px">    
  <img src='data/readme/cin/monet10/im6.jpg' height="210px">    
  <img src='data/readme/cin/monet10/im7.jpg' height="210px">    
  <img src='data/readme/cin/monet10/im8.jpg' height="210px">    
  <img src='data/readme/cin/monet10/im9.jpg' height="210px">   
</p>

### Real-Time Style Transfer
This repository also contains a re-implementation of the paper [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) and the author's torch implementation([fast-neural-style](https://github.com/jcjohnson/fast-neural-style)) with the following differences:

1. The implementation uses the `conv2_2` layer of the VGG-Net for the content loss as in the paper, as opposed to the `conv3_3` layer as in the author's implementation.
2. The following architectural differences are present in the transformation network, as recommended in the paper [A Learned Representation For Artistic Style](https://arxiv.org/abs/1610.07629):  
  a. Zero-padding is replaced with mirror-padding  
  b. Deconvolutions are replaced by a nearest-neighbouring upsampling layer followed by a convolution layer  
  These changes obviate the need of a total variation loss, in addition to providing other advantages.
3. The implementation of the total variational loss is in accordance with [this](https://github.com/Lasagne/Recipes/blob/master/examples/styletransfer/Art%20Style%20Transfer.ipynb) one, different from the author's implementation. Total varition loss is no longer required, however (refer point 2). 
4. The implementation uses Instance Normalization (proposed in the paper ["Instance Normalization: The Missing Ingredient for Fast Stylization"](https://arxiv.org/abs/1607.08022)) by default: although Instance Normalization has been used in the repo containing the author's implementation of the paper, it was proposed after the paper itself was released.
5. The style loss weights have been divided by the number of layers used to calculate the loss (though the values of the weights themselves have been increased so that the actual weights effectively remain the same)

<p align='center'>  
  <img src='data/styles/candy.jpg' height="225px">    
  <img src='data/readme/im1.jpg' title='Joel Moniz' alt='Joel Moniz' height="225px">    
  <img src='data/readme/fns/im1_candy.jpg' title='Still Joel Moniz, but stylized' alt='Still Joel Moniz, but stylized' width="530px">  
</p>

## References
### Papers

This repository re-implements 3 research papers:

1. [A Learned Representation For Artistic Style](https://arxiv.org/abs/1610.07629)
2. [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
3. [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)

### Implementations

  * The author's implementation of the paper "Perceptual Losses for Real-Time Style Transfer and Super-Resolution" ([fast-neural-style](https://github.com/jcjohnson/fast-neural-style))
  * The Google Magenta implementation of the paper "A Learned Representation for Artistic Style" ([magenta: image-stylization](https://github.com/tensorflow/magenta/tree/master/magenta/models/image_stylization))
  * The [neural-enhance](https://github.com/alexjc/neural-enhance) repository
  * The Lasagne implementation of of the paper "A Neural Algorithm of Artistic Style" ([lasagne/recipes: styletransfer](https://github.com/Lasagne/Recipes/blob/master/examples/styletransfer/Art%20Style%20Transfer.ipynb))
  * The Keras implementation of of the paper "A Neural Algorithm of Artistic Style" ([keras: neural_style_transfer](https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py))
