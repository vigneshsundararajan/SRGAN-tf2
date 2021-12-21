# Using a GAN with a Perceptual Loss function for Image Super Resolution

This project focuses on reimplementing the paper by [Ledig, et al. (2017)](https://arxiv.org/pdf/1609.04802v5.pdf) in which a GAN is trained on an unspecified subset of 350,000 images from the ImageNet dataset.

## Implementation details
In this implementation, I decided to use the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset consisting of 900 High Resolution (2K) images, which are then downsampled bicubically by a factor of 4, and then fed into the GAN as input. The first 800 images were used for training and the final 100 were used for testing. The Generator and Discriminator were trained for 50,000 steps in increments of 50, and their losses were reported and visualized accordingly.

## Requirements
This implementation was run on Google Colab, but nevertheless has the following libraries as requirements:
- TensorFlow 2.3.1 or above
- Matplotlib 3.1.1 or above
- Python 3.5 or above
- Pillow 6.0.0 or above
- Numpy 1.21.4 or above

The above requirements can be installed by 
```bash
$ cd <project-root>
$ pip install -r requirements.txt
```
## Results
![Screenshot 2021-12-20 at 9 02 11 PM](https://user-images.githubusercontent.com/68025565/146858246-92d5d309-3c2d-4fdd-b329-a91ff722cb28.png)

## References
[https://github.com/jlaihong/image-super-resolution](https://github.com/jlaihong/image-super-resolution) 

[https://github.com/krasserm/super-resolution](https://github.com/krasserm/super-resolution)

[https://arxiv.org/pdf/1609.04802v5.pdf](https://arxiv.org/pdf/1609.04802v5.pdf)

[https://medium.com/@ramyahrgowda/srgan-paper-explained-3d2d575d09ff](https://medium.com/@ramyahrgowda/srgan-paper-explained-3d2d575d09ff)
