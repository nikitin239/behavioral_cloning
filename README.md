

**Behavioral Cloning Project**



[//]: # (Image References)


[image2]: /examples/2.png "Training results with dropout and added reverse driving data"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model
### Model Architecture and Training Strategy
It was very difficult and interesting process with a lot of iterations.

#### 1. Solution Design Approach

1) At first I used simple LeNet architecture and about 7000 images of driving from central camera, I've trained that network for 7 epochs and result was not good, my car was driving in a circle, and also, i've got overfitting after 3 epochs, for normalization I used:

```sh
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
```

2) Then I've applied augmentation strategy with flipping images from central camera and training set becomes to be about 14k of images following openCV method flip:

```sh
cv2.flip(image, 1)
```

3) After increasing data set i've desided to increase number of epochs too
And results was better then in previous steps, but the car was driving terrible
4) I've decided to train car on the second track, and after 2 laps there, i've retrained my model, it was better, but there was problems on the corners and when car lost track it was driving in a circle
5) I've cropped images from the bottom for 50 rows and from the top for 20 rows for better network training results, trees, sky and etc doesn't matter for us, and also I've added images from left and right camera with correction value=0.2, there was triple count of data, which could help me train network better. I've avoided overfiting, have increased count of epochs to 10 but after checking progress I've found problems on turns.
6) Then I've changed network architecture to NVIDIA SELF DRIVING ARCHITECTURE:
<p> * CONVOLUTIONAL LAYER
<p> * MAXPOOL LAYER
<p> * CONVOLUTIONAL LAYER
<p> * MAXPOOL LAYER
<p> * FLATTEN LAYER
<p> * DENSE LAYER
<p> * DENSE LAYER
<p> * DENSE LAYER 
<p>And then little bit tuned it, new Architecture was:


```sh
model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```
This arch. was good solution, but sometimes I've found that it checks markup bad and sometimes goes to the curbs. Also I saw overfitting after 3rd epoch. 

7) I've decided collect more data and increased dataset with values to passing turns, it was about 57000 of samples, training time was terrible, but results not so good. 
8) I thought, I should reset all my training data and i followed your advice recording just 2 laps centered ride, one lap reverse ride, 1 lap with going slowly on turns, Also I've added dropout layer to avoid overfitting and increased count of epochs to 5, and changed cropping to 70,25 and it helped me cut not matter data.
9) Final architecture is: 
```sh
model=Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```
I used Adam optimizer and MSE loss algorithm, also to avoid memory problems I've tuned code with generator and used fit_generator method for Keras.
```sh
model.compile(loss='mse',optimizer='adam')
history_object=model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/32,
  validation_data=validation_generator, validation_steps=len(validation_samples),
  epochs=5,verbose=1)

```

#### 2. Training results.
![alt text][image2]

