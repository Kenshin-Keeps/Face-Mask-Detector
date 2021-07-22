# Face Mask Detector
 This system can detect if there is a mask on a face in any video footage.

There are many Face mask detector out there. I have build this one to get a good understanding of how to train, test and implement an ML model. For this, I have followed a tutorial of [**pyimagesearch**](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/). 

The dataset that I have used, is collected form [**Kaggle**](https://www.kaggle.com/prithwirajmitra/covid-face-mask-detection-dataset). 

There are two main parts:
1. Train the model.
2. Implement the model.

The first is done in ```model_train.py``` file and the later one is done in ```implementation.py``` file. ```MobileNetV2``` model is used in this code as the base model and then  the head created as per interest. 

Steps to run this model:
1. Donwload this Github repository and unzip it in you desired location.
2. Keep the virtual environment active in the terminal
3. Traverse to the file localiton where "implementation.py" file exists.
4. Run that python file. 
   - Command : ```python implementation.py```
5. Your webcam will start and try to play with and without mask on your face. 
