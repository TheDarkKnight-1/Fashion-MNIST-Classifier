## Fashion-MNIST-Classifier
A convolution neural network (CNN) to classify the fashion items in 10 respective classes in the Fashion MNIST dataset.

I have downloaded the fashion_MNIST dataset locally which are in the form of "idx3-ubyte" files

## Files downloaded from:  "https://www.kaggle.com/zalando-research/fashionmnist"

<ul>
<li>train-images-idx3-ubyte.gz</li>
<li>train-labels-idx1-ubyte.gz</li>
<li>t10k-images-idx3-ubyte.gz</li>
<li>t10k-labels-idx1-ubyte.gz</li>
</ul>

These files are also available in my repo

## We need to convert the idx files to numpy arrays inorder to process the images:

<p>trainX = idx2numpy.convert_from_file(train_path)</p>
<p>trainY = idx2numpy.convert_from_file(train_label_path)</p>

<p>testX = idx2numpy.convert_from_file(test_path)</p>
<p>testY = idx2numpy.convert_from_file(test_label_path)</p>

### The step by step explaination of how to design the classifier is in the "DNN_fashion_image_classifier.ipynb" jupyter notebook of my repository.
