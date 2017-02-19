**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
 

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

In this repository, there are also:
* model_hkawanishi_final.ipynb I used Jupyter notebook throughout this project.  This was later converted to model.py
* model_hkawanishi_final.py This is same as model.py  I converted model_hkawanishi_final.ipynb to model_hkawnaishi_final.py and then copy it to model.py

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
model.h5 is included in this repository.

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network (model.py lines 102-127)
* 5 x 5 filter, depth of 24
* 5 x 5 filter, depth of 36
* 3 x 3 filter, depth of 48
* 3 x 3 filter, depth of 64
and
three fully connected layers.

The model includes RELU layers to introduce nonlinearity (lines 102-127), and the data is normalized in the model using a Keras lambda layer (code line 100). 
At each conv2d layer, Max pooling is also added.  

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 102-127). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 47, lines 87-88). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 144).

####4. Appropriate training data

When I first downloaded the simulation, I had a hard time driving with my computer using arrow keys and a mouse and can't seem to be able to produce good training data.  So I decided to use the training data which were downloaded in Udacity site.  I used all center image, left and right camera images.
(note: at the end of my project, I think the newly improved simulation became available which seemed to be easier to drive even using arrow keys.  However, I was already training with the current data so I decided to stick with them).

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start a simple model and then try to improve the model by adding more layers.

My first step was to add one fully connected layer to see if the code worked.  The vehicle immediately went outside the path but the process seemed to work.  
Then I started to add more layers.  I used the NVIDIA paper (https://arxiv.org/pdf/1604.07316v1.pdf) as a guideline.  But rather than added layers and tried, I added one layer at a time and tested each change.
For each layer, I added RELU.  Then I added dropout to prevent overfitting.  
I tried BatchNormalization but this made my model output a constant steering angle. I am not sure if I was not using it correctly or not but decided not to use it.

I first used only center images and tested.  

I noticed the vehicle just veered to left no matter how many layers I added.  I added flipped images (more details in "Creation of the Training Set & Training Process" section) and then added left and right images and that improved the vehicle behaviour.  The clopping the images also helped the vehicle behavior.  

After adding all available training set and improving the model, there were still a few places where teh vehicle fell off the track.  I had to use a minimum batch size (4) to be able to make the vehicle successfully ran without leaving the road.  

####2. Final Model Architecture

My model consists of a convolution neural network (model.py lines 102-127)
* 5 x 5 filter, depth of 24
* 5 x 5 filter, depth of 36
* 3 x 3 filter, depth of 48
* 3 x 3 filter, depth of 64
and
three fully connected layers..

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

As I wrote above, I had a hard time driving a vehicle using a mouse/keyboard initially. Later, it looks like the improved simulation became available but by then I already decided to use the training data which were provided.  

The example image is below.  Each image had center, left, and right camera iamges.

![alt text][image2]

Then I repeated this process on track two in order to get more data points.

I noticed the vehicle always veered to left since the training data only contained the left turn.  To augment the data set, I also flipped images and multiplied the steering angle to -1.0.  For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had X number of data points. I then preprocessed this data by


I andomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I chose the number of epochs to be 5 because when I used more it didn't seem to improve the accuracy.  I used an adam optimizer so that manually training the learning rate wasn't necessary.

