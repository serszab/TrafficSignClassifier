# **Traffic Sign Recognition** 

Author: Szabolcs Sergyan

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

Your are reading my writeup. My project code and generated html is available in my workspace or you can find it in my github [repo](https://github.com/serszab/TrafficSignClassifier/tree/master).

### Data Set Summary & Exploration

#### 1. Summary of the data sets

You can find my data set summary in code cell 2.

* The size of training set is 34,799.
* The size of the validation set is 4,410.
* The size of test set is 12,630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Exploratory visualization of the dataset.

You will find twelve examples of training set images in code cell 3. The title of each image contains its label and description.

The distribution regarding labels of each dataset is shown in cell 4. You can see the distribution is not exactly the same for the three datasets.

### Design and Test a Model Architecture

#### 1. Preprocessing of images

As preprocessing I only made two steps. First I converted the images to grayscale and then normalized them into range [-1, 1].

I would have made other preprocessing steps like randomly rotating or translating images. Other option would have been considering color information using a well selected color space.
I did not apply them because the final results meet the requirements currently and I did not have too much time to improve my results.

An example of grayscale conversion and normalization can be seen in code cell 5.


#### 2. Model architecture

I used the LeNet architecture with some modifications.

My final model consisted of the following layers:

| Layer         		    |     Description	        					| 
|:-------------------------:|:---------------------------------------------:| 
| Input         		    | 32x32x1 grayscale image   					| 
| Convolution 5x5 	        | 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					    |												|
| Max pooling	      	    | 2x2 stride, outputs 14x14x32  				|
| Convolution 5x5	        | 1x1 stride, valid paddig, outputs 10x10x16	|
| RELU		                |                                               |
| Max pooling			    | 2x2 stride, outputs 5x5x16					|
| Flatten       		    | outputs 400									|
| Droppout				    | keeping probability 0.65						|
| Fully connected		    | outputs 120            						|
| RELU	    			    |                       						|
| Droppout				    | keeping probability 0.65						|
| Fully connected		    | outputs 84            						|
| RELU	    			    |                       						|
| Fully connected		    | outputs 43            						|
 
For the implementation see code cell 6.

#### 3. Training the model

In training phase I used Adam optimizer with learning rate 0.00075. The batch size was 128 and the number of epochs was 30. Mean was set to 0 and standard deviation to 0.1.

In the code you can see it in cell 7.

#### 4. Approach taken to find a solution

My final model results were:
* training set accuracy of 99.8%.
* validation set accuracy of 97.7%
* test set accuracy of 96.0%

The code and results can be seen in cells 8-11.

I chose LeNet architecture. I think convolutional neural networks are effective for computer vision projects. The convolutional layers can implement many image processing methods. In order to improve the effectiveness of the network I used more deep convolutional layers what was suggested in the videos.

Based on the accuracy results I think the network was not over- or underfitted. The test results a bit worse than the validation results, but they look fine. (See cell 11)


### Test a Model on New Images

#### 1. 11 images selected from the Internet

I selected 11 German traffic signs from the Internet. The network worked well for some images, but for some images it gave not so good result.

In cell 12 you will find the selected images.

The 4th image had such bad quality that was challenging to identify the traffic sign for me as well.

#### 2. Model's prediction for the selected images

The overall prediction for the selected images was only 72.7%. It is a little bit surprising. In cell 14 and 15 you can see the results.

The model was able to identify correctly 8 images of 11.

#### 3. Certainty of prediction

In cell 16 you can see the first five "closest" images.

That is surprising for the fourth image the a bad result was given with 100% certainty. I guess its reason would be the bad quality of the image itself.

Another weird case is the 10th image (speed limit 100). The network predicted a result with 100% certainty, but I could not find any similarity between the two images.

