# gogh-figure  
  
Fast, Lightweight Style Transfer using Deep Learning

## Results

<!--### Real-Time Style Transfer -->
This repository contains a re-implementation of the paper [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) and the author's torch implementation([fast-neural-style](https://github.com/jcjohnson/fast-neural-style)) with the following differences:

1. The implementation uses the `conv2_2` layer of the VGG-Net for the content loss as in the paper, as opposed to the `conv3_3` layer as in the author's implementation.
2. The following architectural differences are present in the transformation network, as recommended in the paper [A Learned Representation For Artistic Style](https://arxiv.org/abs/1610.07629):  
  a. Zero-padding is replaced with mirror-padding  
  b. Deconvolutions are replaced by a nearest-neighbouring upsampling layer followed by a convolution layer  
  These changes obviate the need of a total variation loss, in addition to providing other advantages.
3. The implementation of the total variational loss is in accordance with [this](https://github.com/Lasagne/Recipes/blob/master/examples/styletransfer/Art%20Style%20Transfer.ipynb) one, different from the author's implementation. Total varition loss is no longer required, however (refer point 2). 

<p align='center'>  
  <img src='data/styles/candy.jpg' height="225px">    
  <img src='data/readme/im1.jpg' title='Joel Moniz' alt='Joel Moniz' height="225px">    
  <img src='data/readme/fns/im1_candy.jpg' title='Still Joel Moniz, but stylized' alt='Still Joel Moniz, but stylized' width="530px">  
</p>

## References
### Papers

This repository re-implements 2<!--3--> research papers:

<!--1. [A Learned Representation For Artistic Style](https://arxiv.org/abs/1610.07629)-->
2. [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
3. [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)

### Implementations

  * The author's implementation of the paper "Perceptual Losses for Real-Time Style Transfer and Super-Resolution" ([fast-neural-style](https://github.com/jcjohnson/fast-neural-style))
  * The Google Magenta implementation of the paper "A Learned Representation for Artistic Style" ([magenta: image-stylization](https://github.com/tensorflow/magenta/tree/master/magenta/models/image_stylization))
  * The [neural-enhance](https://github.com/alexjc/neural-enhance) repository
  * The Lasagne implementation of of the paper "A Neural Algorithm of Artistic Style" ([lasagne/recipes: styletransfer](https://github.com/Lasagne/Recipes/blob/master/examples/styletransfer/Art%20Style%20Transfer.ipynb))
  * The Keras implementation of of the paper "A Neural Algorithm of Artistic Style" ([keras: neural_style_transfer](https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py))
