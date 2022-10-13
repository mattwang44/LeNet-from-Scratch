# LeNet5 Implementation FROM SCRATCH
This is an implementation of LeNet5 from [Yann LeCun's paper](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) in 1998, using Numpy & OOP only (without any auto-differentiate tools or deep learning frameworks).

Yann LeCun's demo in 1993: 
<p align="center">
<a href="http://www.youtube.com/watch?feature=player_embedded&v=FwFduRA_L6Q
" target="_blank"><img src="http://img.youtube.com/vi/FwFduRA_L6Q/0.jpg" 
alt="LeNet demo" width="450" height="340" border="10" /></a></p>

## Result of Training 

Highest accuracy of 98.6% on MNIST testing dataset has achieved in 20 epoches of training (93.5% after 1st epoch). The training (20 epoches, batch size = 256) takes about 2 hours using CPU only (3.5 hours if evaluate after each epoch).

<p align="center"> 
  <img width="460" height="300" src="./image/errRate.png">
</p>

Feature maps in each layer:
<p align="center">
  <img width="800" src="./image/all2.jpg">
</p>


## File Structure

```
LeNet5_from_scratch/
├── LeNet5_train.ipynb                 # Notebook for training and shows the results
├── RBF_initial_weight.ipynb           # Notebook shows the fixed weight (ASCII bitmap) in the RBF layer
├── ExeSpeedTest.ipynb                 # Comparison of different version of Conv. & Pooling functions
├── Best_model.pkl                     # The model with 98.6% accuracy both on training and testing data 
│                                      # Please download at [tinyurl.com/mrybvje9] or train one by yourself :)
│
├── MNIST_auto_Download.py             # Python script for auto-download MNIST dataset (like folder below)
├── MNIST/                             # Folder contains MNIST training and testing data
│   ├── train-images-idx3-ubyte        # Training images
│   ├── train-labels-idx1-ubyte        # Training labels
│   ├── t10k-images-idx3-ubyte         # Testing images
│   └── t10k-labels-idx1-ubyte         # Testing labels
│
└── utils/
    ├── __init__.py 
    ├── Convolution_util.py            # Convolution forward and backward
    ├── Pooling_util.py                # Pooling forward and backward
    ├── Activation_util.py             # Activation functions
    ├── utils_func.py                  # Other functions like normalize(), initialize(), zero_pad(), etc
    ├── RBF_initial_weight.py          # Setting fixed weight (ASCII bitmap) in the RBF layer
    └── LayerObjects.py                # All the layer objects
```

## Structure of ConvNet

The structure in the original paper is:

<p align="center">
  <img width="800" src="./image/lenet5.png">
</p>

The structure used in this repo have a few modification:

1. Substitute the **sub-sampling** with **average pooling**, which is more accpetable choice without trainable parameters in the layer and needless to be followed by an activation funciton. (I've tried using max-pooling, but it blurs the feature maps in this case and gives low accuracy.)

2. **momentum optimizer** (momentum=0.9) is used to accelerate the training process (for faster convergence).


## Bug Alert

Stochastic Diagonal Levenberg-Marquaedt method from the original paper is also used in this implementation to determine the learning rate for each trainable layer. However, resulting range of learning rates is much smaller than the one given in the paper (maybe bugs exist in the SDLM code). Therefore, 100x original global learning rates is applied and it works fine then.


## Reference

1. [Yann LeCun's paper](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
    * Masterpiece of CNN. Still so much knowledge that I don't fully understand even after this project.
2. [Marcel Wang's blog](http://hemingwang.blogspot.tw/search/label/_社團：技術：mAiLab)
    * Special thanks to Marcel Wang for encouraging everyone to do this project.
3. [Deep Learning Specialization by Andrew Ng](https://www.coursera.org/specializations/deep-learning)
    * Epic lectures & inspiring assignments. Couldn't done this if I didn't take the courses.
1. [agjayant's repo](https://github.com/agjayant/CNN-Numpy)
2. [HiCraigChen's repo](https://github.com/HiCraigChen/LeNet)



## Todo list

1. Compare RBF layer with softmax layer (cross entropy) or simply a FC layer
2. Accelerate with Cython or PyCuda
3. Try using sub-sampling layer
