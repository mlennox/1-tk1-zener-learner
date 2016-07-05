# Zener learner
Simple neural net to classify zener cards. Tutorial 1 in a series of some.

## Zener cards?
In the 1930s psychologist Karl Zener designed a set of five cards to be used in experiments conducted with paraphyschologist Joseph Rhine.

![Zener Cards Image - Circle, Cross, Wavy Lines, Square, Star](https://github.com/mlennox/1-tk1-zener-learner/blob/master/data/Zener_cards_color.svg.png)

Image by Mikhail Ryazanov (talk) 01:30, 1 April 2014 (UTC) (File:Cartas Zener.svg + File:Zenerkarten c.jpg) [GFDL (http://www.gnu.org/copyleft/fdl.html) or CC-BY-SA-3.0 (http://creativecommons.org/licenses/by-sa/3.0/), via Wikimedia Commons]

The experimenter would choose a card from a pack of 25, which contained five sets of the five cards shown in the image above - Circle, Cross, Wavy Lines, Square and Star. If the experimental subject guessed correctly more than 20% of the time then it could be possible that they are endowed with supernatural abilities. Or not, the experiments never conclusively proved the existence of ESP.

I hope the parallel with machine learning classification is obvious and in this repo I will attempt to train a Theano based neural net (non-convolutional) running on a Jetson TK1 to properly classify Zener Cards.

## Goal
The point of this project - besides teaching me machine learning - is to produce a neural net that can recognise any of the Zener symbols whether that input comes from a high-quality photoshop rendering or a badly-lit phone camera shot of a crudely-drawn Zener symbol.

This will probably not be achieved with such a simple network...

## Prerequisites
You will need to have Python 2.7 installed. Earlier versions *may* work, but that is what I have installed, so. Also you'll need to install [Pillow](https://pillow.readthedocs.io/en/3.0.0/installation.html) for loading and mucking about with image files.

## Test data
We will attempt to generate a large data set starting only with the symbols taken from the image above.
To achieve this I will use Python to distort, scale and transpose the initial data.
It is likely I will add more starting data to the examples, but for now these will suffice as it is a simple neural net the risk of over-fitting is somewhat lower.

### Data augmentation
After a cursory search I couldn't find any tools that would help me generate extra data from an existing data set. The first part of this project will require the creation of some Python scripts to fold, spindle and mutilate the starting data set.

#### The symbols after distortion
Below you can see an example of what the data expansion script generates. I think these look pretty good for a start. The script currently applies a perspective distortion and then rotates the image before cropping it down to the symbol and resizing the image to the chosen 32 x 32 pixel data sample size.

![Circle](https://github.com/mlennox/1-tk1-torch-zener-learner/blob/master/content/circle7.png)
![Cross](https://github.com/mlennox/1-tk1-torch-zener-learner/blob/master/content/cross1.png)
![Wavy](https://github.com/mlennox/1-tk1-torch-zener-learner/blob/master/content/wavy9.png)
![Square](https://github.com/mlennox/1-tk1-torch-zener-learner/blob/master/content/square6.png)
![Star](https://github.com/mlennox/1-tk1-torch-zener-learner/blob/master/content/star8.png)

#### Additional work
I may add some other type of distortion - pincushion, skew or whatever, but we'll see what the training evaluation tells us.

An obvious enhancement would be to use a larger set of starting images. It would be straight-forward enough to find (creative commons of course!) images of Zener cards, and even add some hand-drawn versions. 

Another simple augmentation is to add colour to the training examples, overlay with a vignette, and add random patterns to the background of each training image.

### Loading Data

W.I.P.

## The network
### Construction
This will be a simple neural network with two hidden layers and five output nodes, one for each of the 'categories' we are going to train the network to classify - circle, cross, wavy lines, square, star.

We'll use a [Rectified Linear unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) instead of a more standard Sigmoid. There are two reasons for this - I hear they are faster and also less prone to vanishing gradient. The latter is not such a problem with such a simple network, but any help I get to increase the learning rate and avoid getting stuck in a local minimum, I'll take.

The output layer will use the [Softmax function](https://en.wikipedia.org/wiki/Softmax_function) to produce a set of probabilities on the output nodes. The softmax function just normalises the outputs so that all the probabilities add up to 1. 

### Training
We will use Stochastic Gradient Descent to train the network. We'll use a batch size of 32 so ech training epoch will take 32 steps to complete.



### Metrics
While training the network it will be imperative to know how the training is going - does the learning rate need tweaking? Is the network overfitting? Is it training quickly enough?
To measure the validity of choices for the network parameters, I'll need to hold aside some of the data to use as a validation set to evaluate different learning rates etc.
