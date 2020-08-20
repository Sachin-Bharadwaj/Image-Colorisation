# Image-Colorisation

Using Resnet based encoder and transposed convolution based decoder, AutoEncoder is used for image colorization by exploting lab colorspace. The performance on validation set is not great but for illustration purpose, it suffices and can be improved further.

**Left: Gray scale image as inpput, Middle: Ground Truth, Right: Prediction**

**Training set**
![Alt Text](https://github.com/Sachin-Bharadwaj/Image-Colorisation/blob/master/train_gif.gif)

**Validation set**
![Alt Text](https://github.com/Sachin-Bharadwaj/Image-Colorisation/blob/master/val_gif.gif)

The below gifs are trained on cricket video from youtube, weights are different from what is there in this github repo but architecture is same

**Training set**
![Alt Text](https://github.com/Sachin-Bharadwaj/Image-Colorisation/blob/master/train_cricket_gif.gif)

**Validation set: this is different distribution from train set, was just curious on how would the network perform**
![Alt Text](https://github.com/Sachin-Bharadwaj/Image-Colorisation/blob/master/val_cricket_gif.gif)
