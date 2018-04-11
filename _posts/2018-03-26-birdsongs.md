---
layout: post
---

# Classification of Bird Songs Using the Inception-v3 Network

## Introduction

While deep learning is widely applied to speech recognition, efforts to categorize environmental sounds or music with the same techniques are less common. To the best of my knowledge, the identification of bird songs from field recordings has only been attempted in [one paper](http://ceur-ws.org/Vol-1609/16090560.pdf) from 2016. The authors, Tóth and Czeba, used a standard short-time FFT procedure to convert audio into 2D spectrograms that are fed into a standard convolutional neural network (CNN). Interestingly, they also decided to drop the low-intensity portions of the images (i.e. set them to zero) to emphasize the actual features. This seemed slightly counterproductive to me, since the stark contrast between the zero-set portions and the leftovers produced artificial edges that a CNN would detect as features. Still, their results were solid: their best model, a variant of AlexNet, reached a MAP (mean average precision) of 43% over 999 classes.

For this project, I opted to build my own audio preprocessing pipeline and use the more complex [Inception-v3](https://arxiv.org/abs/1409.4842) model architecture. The code for the audio processing and model training/evaluation steps can be viewed on [GitHub](https://github.com/austinmoehle/birdsongs).

## The Dataset

<img src="/assets/Tufted_Titmouse_1.jpg" width="400">

**Tufted Titmouse**

<img src="/assets/White-throated_Sparrow_1.jpg" width="400">

**White-Throated Sparrow**

<img src="/assets/Tennessee_Warbler_male.jpg" width="400">

**Tennessee Warbler (Male)**


The Cornell Lab of Ornithology provides a Master Set of bird field recordings, consisting of 2.65 GB (~5000 recordings) of bird songs and calls labeled by species and by type (e.g. "whistle call" or "song"). To narrow the focus of this project, I limited the dataset to just songs and included only bird species with at least 6 field recordings. This reduced the number of classes from 326 to 48.

Label        | Bird Species | Recordings
-------------|--------------|-------------
1|Tufted Titmouse|22
2|White-throated Sparrow|19
3|Tennessee Warbler|15
4|Magnolia Warbler|15
5|White-winged Crossbill|13
...|...|...
48|American Tree Sparrow|6


See the [labels file](/samples/labels.txt) for a full list of the 48 included birds.

## Data Preprocessing

The standard approach to feeding audio into a convolutional neural network (CNN) is to first slice the audio into short snippets then process these samples using a short-time Fourier transform (FFT), producing frequency-vs.-time spectrograms. These 2D greyscale "images" can be used as input to any standard CNN once properly scaled and cropped.

The FFT works as follows: a short-time window placed over the audio captures the frequency distribution of the sound at that instant in time. Sliding this window over the length of the data produces a 2D "image" that captures the frequency-vs.-time characteristics of the audio. Usually, a mel-scale (a type of log-scale) is used for the frequency axis.

For this project, I extracted 4-second snippets from each field recording by sliding a triangular window function across the audio and selecting the highest-intensity samples. After applying an FFT (Hamming window, FFT size of 512 samples, overlap of 256 samples), I generated 1857 greyscale images (224x341), each representing 4 seconds of audio.

<img src="/samples/wav/6_5.jpg" width="400">

**Song of a Ruby-Crowned Kinglet**


In the TensorFlow graph, the parsing function takes a random 3.75-second crop of each 4-second spectrogram then rescales the resulting image to 299x299 to match the input of the Inception-v3 network. The random crop in the time dimension serves as a data augmentation step; each time a specific spectrogram is sampled by the model, it is slightly different. I chose not to apply a random crop in the frequency dimension to keep absolute pitch unperturbed.

```python
def _parse_function(filename, label):
    # (1) Decode the image from jpg format.
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
    image.set_shape([224, 341, 3])

    # (2) Resize the image 299x319.
    new_height = tf.to_int32(299.0)
    new_width = tf.to_int32(319.0)
    crop_height = tf.to_int32(299.0)
    crop_width = tf.to_int32(299.0)
    image = tf.image.resize_images(image, [new_height, new_width])

    # (3) Take a random 299x299 crop of the image (random time slice).
    max_offset_height = tf.reshape(new_height - crop_height + 1, [])
    max_offset_width = tf.reshape(new_width - crop_width + 1, [])
    offset_height = tf.constant(0, dtype=tf.int32)
    offset_width = tf.random_uniform([], maxval=max_offset_width, dtype=tf.int32)
    original_shape = tf.shape(image)
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])
    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))
    image = tf.slice(image, offsets, cropped_shape)

    # (4) Standard preprocessing for Inception-v3 net:
    #     Scale `0 -> 1` range for each pixel to `-1 -> 1`.
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image, label
```

To balance the training dataset, images from less-populated classes are resampled multiple times per epoch to keep the model’s exposure to each class relatively even. Note that the images are slightly different each time they are sampled due to random cropping.


## Model Architecture

Since I was limited by my personal GPU (GTX 960 with 4GB RAM), training a large CNN from scratch was not feasible. Instead, I initialized a pre-trained model then fine-tuned the learned weights on my dataset, a technique called transfer learning. I used the Inception-v3 CNN model architecture, as specified in [the paper](https://arxiv.org/abs/1409.4842) and provided by the TensorFlow Slim model library. Inception-v3 achieved 000000 accuracy on the [ImageNet](http://www.image-net.org/) training set of 000000 images from 000000 classes.

![Inception-v3 Architecture](/assets/inception_v3_architecture.png)

The Inception architecture is based on the "Inception module", a parallel stack of convolutional filters (1x1, 3x3, 5x5) alongside a pooling layer. Because large convolutions (3x3, 5x5) are computationally expensive when applied to a layer with a large number of filters, 1x1 convolutions (convolutions over the depth dimension) are placed before these larger convolutions to reduce the dimensionality of the input. This was first done in 2014 for the original 22-layer Inception network, [GoogLeNet](https://arxiv.org/abs/1409.4842).

For Inception-v3 (2016), Google improved the modules further by factorizing large convolutions: replacing 5x5 convolutions with two 3v3 convolutions, replacing 7x7 convolutions with 1x7 and 7x1 convolutions in series, etc. The final net, which contained these changes and numerous other improvements, was 42 layers deep but with a computational cost only 2.5 times higher than GoogLeNet; see [the paper](https://arxiv.org/abs/1512.00567) for a detailed description of the model architecture.

Although Inception-v3 was trained on an image classification task, I expected the model to transfer fairly well to audio classification. The visual features in bird song spectrograms (e.g. patterned lines and squiggles) are easily distinguishable by a human observer, so they should also be picked up by the early layers of CNNs, which are generally responsible for detecting low-level visual features.

I initially froze these lower layers with the ImageNet-trained weights and trained only the final fully-connected layer used for classification. Then, I "fine-tuned" the model by unfreezing the rest of the layers and training the entire network over many epochs.

## Training the Final Fully-Connected Layer

I set up training for only the final (fully-connected) layer of Inception-v3.

```python
# Restore only the layers before Logits/AuxLogits.
layers_exclude = ['InceptionV3/Logits', 'InceptionV3/AuxLogits']
variables_to_restore = tf.contrib.framework.get_variables_to_restore(
    exclude=layers_exclude)
init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
    args['init_path'], variables_to_restore)

logits_variables = tf.contrib.framework.get_variables('InceptionV3/Logits')
logits_variables += tf.contrib.framework.get_variables('InceptionV3/AuxLogits')
logits_init = tf.variables_initializer(logits_variables)

tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                       logits=logits,
                                       weights=1.0)
tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                       logits=end_points['AuxLogits'],
                                       weights=0.4)
loss = tf.losses.get_total_loss()

# Use this optimizer to train only the re-initialized final (FC) layer.
logits_optimizer = tf.train.AdamOptimizer(learning_rate=args['learning_rate'],
                                          epsilon=args['epsilon'])
logits_train_op = logits_optimizer.minimize(loss, var_list=logits_variables)
```

Good values of `learning rate` and `epsilon` for the Adam optimizer were determined by a random search, as illustrated below.

![Hyperparameter search](/assets/hp_logits.png)


After training the final layer for 200 epochs with `learning_rate=6.1e-3` and `epsilon=0.93`, the model achieved 27.7% classification accuracy on the train set and 24.0% on the validation set.

-------------|--------------
Loss|4.541
Train Accuracy|27.7%
Validation Accuracy|24.0%



## Training the Full Inception-v3 Network


Continuing from the partially-trained model above, I modified the graph to allow the optimizer to train all layers of the network. As before, I conducted a hyperparameter search and settled on `learning_rate=7.8e-4` and `epsilon=0.67` for this phase of training.

```python
all_variables = tf.contrib.framework.get_variables_to_restore()
restore_dir_fn = tf.contrib.framework.assign_from_checkpoint_fn(
    tf.train.latest_checkpoint(restore_dir), all_variables)
tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                       logits=logits,
                                       weights=1.0)
tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                       logits=end_points['AuxLogits'],
                                       weights=0.4)
loss = tf.losses.get_total_loss()
tf.summary.scalar('loss', loss)

# Use this optimizer to train all model layers.
full_optimizer = tf.train.AdamOptimizer(
    learning_rate=args['learning_rate'], epsilon=args['epsilon'])
full_train_op = full_optimizer.minimize(loss)
```

The results after training for 750 epochs are given below.

-------------|--------------
Loss|3.213
Train Accuracy|46.4%
Validation Accuracy|41.7%

Loss and accuracy both appeared to plateau after the first 250 epochs. While it may be feasible to squeeze more performance out of the network with different settings, it's also possible that the dataset is too small or too inherently noisy/random to improve substantially. I plan to test alternative image processing settings (e.g. sample length or FFT parameters) and model hyperparameters (weight decay, learning rate, etc.) to see if I can make slight improvements.

Finally, I evaluated the model on the held-out test set, which consists of 286 spectrograms generated from distinct recordings not shared by the training data. The resulting accuracy, 27.3%, is somewhat lower than the training/validation numbers. I suspect that the model is overfitting due to the small size of the dataset: only 408 recordings (1571 training images) over 48 classes.

-------------|--------------
Top-1 Test Accuracy|27.3%
Top-5 Test Accuracy|59.3%


## Examples of Correct Predictions

Let's take a look at some spectrograms and associated model predictions. For each prediction, I've provided a random sample from the true class and a random sample from the predicted class. Comparing the two might give you an idea of the visual similarities that are "confusing" the model when it makes a classification error.

### Example 1 - [\#48 - American Tree Sparrow]
<img src="/samples/sample_5.png" width="355">
![Samples 5](/samples/duo_5.png)
[Audio - Left Image](/samples/wav/48_4.wav)

[Audio - Right Image](/samples/wav/48_5.wav)


### Example 2 - [\#44 - Blue Grosbeak]
<img src="/samples/sample_13.png" width="355">
![Samples 13](/samples/duo_13.png)
[Audio - Left Image](/samples/wav/44_4.wav)

[Audio - Right Image](/samples/wav/44_1.wav)


### Example 3 - [\#17 - Virginia's Warbler]
<img src="/samples/sample_14.png" width="355">
![Samples 14](/samples/duo_14.png)
[Audio - Left Image](/samples/wav/17_2.wav)

[Audio - Right Image](/samples/wav/17_1.wav)


### Example 4 - [\#33 - Ovenbird]
<img src="/samples/sample_23.png" width="355">
![Samples 23](/samples/duo_23.png)
[Audio - Left Image](/samples/wav/33_5.wav)

[Audio - Right Image](/samples/wav/33_0.wav)


## Examples of Incorrect Predictions


### Example 5 - [\#34 - Orange-Crowned Warbler (Lutescens)]
<img src="/samples/sample_3.png" width="355">
![Samples 3](/samples/duo_3.png)
[Audio - Left Image](/samples/wav/34_2.wav)

[Audio - Right Image](/samples/wav/24_1.wav)


### Example 6 - [\#13 - Lincoln's Sparrow]
<img src="/samples/sample_31.png" width="355">
![Samples 31](/samples/duo_31.png)
[Audio - Left Image](/samples/wav/13_8.wav)

[Audio - Right Image](/samples/wav/6_5.wav)


### Example 7 - [\#32 - Purple Finch (Eastern)]
<img src="/samples/sample_18.png" width="355">
![Samples 18](/samples/duo_18.png)
[Audio - Left Image](/samples/wav/32_4.wav)

[Audio - Right Image](/samples/wav/23_1.wav)




## Conclusions

By finetuning the Inception-v3 network on a birdsong classification problem, I was able to achieve 27.3% top-1 and 59.3% top-5 accuracy across 48 classes. Further experiments will be necessary to determine whether I can wring more performance out of this model by improving the  image processing pipeline or by optimizing hyperparameters (weight decay, choice of optimizer, etc.). Of course, it's also possible that the dataset is too small (1800 distinct training images) or too inherently noisy/random (the same bird song can vary from sample to sample) to improve much further.

What can we do with bird songs and other interesting sounds besides classification? At one point I wanted to try my hand at musical style transfer ala [neural-style](https://github.com/jcjohnson/fast-neural-style) or generate a sonic "visualization" of intermediate model layers as is done with image-based nets. Sadly, there is a catch associated with the spectrogram approach: the spectrogram image is only the real part of the Fourier transform. The phase information, i.e. the imaginary part of the FFT, is "lost", or at least not directly moldable by the network.

In other words, if we try to use a deep neural network to produce original sounds, audio reconstruction based on the model-generated spectrograms will be lossy without the phase information. Direct audio style-transfer (e.g. "genre swaps" for music) might therefore require training on raw (1D) audio rather than 2D spectrograms, which would necessitate a different model architecture. One attempt in the literature to apply a time-convolutional neural network to raw audio was unable to outperform a spectrogram-based approach (see ["End-to-end learning for music audio"](http://benanne.github.io/research/)).

I am not an expert on the theory behind FFTs and phase reconstruction, so if you are aware of a better solution, please let me know!

The code for audio processing and model training/evaluation can be found on [GitHub](https://github.com/austinmoehle/birdsongs).
