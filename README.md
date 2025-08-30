# Sign-Language-To-Text-and-Speech-Conversion ✋ 🗣️


This project presents a real-time method for recognizing American Sign Language (ASL) finger spelling using Convolutional Neural Networks (CNNs). Hand gestures captured from a camera are processed, classified, and used to train the CNN for accurate recognition of hand positions and orientations.


## 📖 Introduction

American Sign Language (ASL) is the primary means of communication for people with hearing and speech impairments. Since they cannot rely on spoken language, they use hand gestures to express ideas and exchange messages. These visual, nonverbal gestures form the basis of sign language.

In our project, we focus on producing a model which can recognize Fingerspelling based hand gestures in order to form a complete word by combining each gesture. The gestures we aim to train are as given in the image below. 

<p align="left">
     <img src="https://user-images.githubusercontent.com/99630855/201489493-585ffe5c-f460-402a-b558-0d03370b4f92.jpg" alt="Image 1" width="500" height="400"/>
</p>

More than 70 million deaf people around the world use sign languages to communicate. Sign language allows them to learn, work, access services, and be included in the communities. It is hard to make everybody learn the use of sign language with the goal of ensuring that people with disabilities can enjoy their rights on an equal basis with others so, the aim is to develop a user-friendly human computer interface (HCI) where the computer understands the American sign language. This Project will help the dumb and mute people by making their life easy. 

### 🎯 Objective
- Create a computer software model using CNN to recognize hand gestures of American Sign Language (ASL).  
- Convert recognized gestures into both text and audio output using a text-to-speech system.

### 🔭 Scope
This system helps bridge communication between deaf/mute individuals and those unfamiliar with sign language by recognizing gestures and providing output in both text and speech.

## 📦 Modules

## A. Data Acquisition 📷
The different approaches to acquire data about the hand gesture can be done in the following ways: 

**1. Glove-based** methods use electromechanical devices to capture hand configuration and position, but they are costly and less user-friendly.

**2. Vision-based** methods use a simple camera to detect hands and fingers, offering a natural and low-cost solution.

The main challenge of vision-based hand detection ranges from coping with the large variability of the human hand’s appearance due to a huge number of hand movements, to different skin-color possibilities as well as to the variations in viewpoints, scales, and speed of the camera capturing the scene.

<p align="left">
     <img src="https://github.com/user-attachments/assets/bbb6bbb6-3b90-4def-b3d1-289d347dde08" alt="Image 1" width="500" height="500"/>
</p>

## B. Data pre-processing and Feature extraction 🔎
- Hand detection using MediaPipe.  
- ROI cropped, converted to grayscale, blurred with Gaussian filters, and converted to binary using thresholding.  
- Collected ASL letters (A–Z) from different angles.  

**Limitation:** Requires clean background and proper lighting.  

**Improvement:** MediaPipe landmarks drawn on plain white images for robustness against background noise and lighting variations.

**Mediapipe Landmark System:** 
<p align="left">
     <img src="https://user-images.githubusercontent.com/99630855/201489741-3649959e-df4d-4c32-898a-8f994be92ca2.png" alt="Image 1" width="500" height="500"/>
</p>

<p align="left">
  <img src="https://user-images.githubusercontent.com/99630855/201490095-96402d48-b289-4ff3-9738-ed99ffcffca6.jpg" alt="Image 2" width="250"/>
  <img src="https://user-images.githubusercontent.com/99630855/201490105-87b17583-45c5-4e3b-82d1-0c9a6f98fc55.jpg" alt="Image 3" width="250"/>
  <img src="https://github.com/user-attachments/assets/859ca7a9-49e3-4802-85dc-25e779408390" alt="Image 4" width="250"/>
</p>

Now we get this landmark points and draw it in plain white background using opencv library 

-By doing this we tackle the situation of background and lighting conditions because the mediapipe library will give us landmark points in any background and mostly in any lightning conditions. We have collected 180 skeleton images of Alphabets from A to Z 

## C. Gesture Classification : ✨

**C.1 Convolutional Layer:** In convolution layer I have taken a small window size [typically of length 5*5] that extends to the depth of the input matrix. 
The layer consists of learnable filters of window size. During every iteration we slid the window by stride size [typically 1], and compute the dot product of filter entries and input values at a given position. 

As I continue this process we'll create a 2-Dimensional activation matrix that gives the response of that matrix at every spatial position, i.e, the network will learn filters that activate when they see some type of visual feature such as an edge of some orientation or a blotch of some colour.

<p align="left">
     <img src="https://user-images.githubusercontent.com/99630855/201490154-1416d8ad-c7df-42a2-a296-5d56bad1d5c5.png" alt="Image 1" width="500" height="500"/>
</p>

**C.2 Pooling Layer:** We use pooling layer to decrease the size of activation matrix and ultimately reduce the learnable parameters. 
There are two types of pooling: 

   **C.2.1 Max Pooling:** In max pooling we take a window size [for example window of size 2*2], and only taken the maximum of 4 values. 
    Well slide this window and continue this process, so well finally get an activation matrix half of its original Size. 

  **C.2.2 Average Pooling:**  In average pooling we take average of all Values in a window.
     <p align="left">
        <img src="https://user-images.githubusercontent.com/99630855/201490158-22a8a043-c2fe-4082-8fb5-a6c173061b58.jpg" alt="Image 1" width="500" height="500"/>
    </p>

## D. Fully Connected Layer: 🔗
In convolution layer neurons are connected only to a local region, while in a fully connected region, we connect the all the inputs to neurons.

   <p align="left">
        <img src="https://user-images.githubusercontent.com/99630855/201490169-00b17306-e355-4d2e-88e5-3fbd4c7b3f17.png" alt="Image 1" width="250" height="250"/>
    </p>

-The preprocessed 180 images/alphabet will feed the keras CNN model.  
-Because we got bad accuracy in 26 different classes thus, we divided whole 26 different alphabets into 8 classes in which every class contains similar alphabets: 
[y,j] 
[c,o] 
[g,h] 
[b,d,f,I,u,v,k,r,w] 
[p,q,z] 
[a,e,m,n,s,t] 

-All the gesture labels will be assigned with a probability. The label with the highest probability will treated to be the predicted label, so when model will classify [aemnst] in one single class using mathematical operation on hand landmarks we will classify further into single alphabet a or e or m or n or s or t. 

-Finally, we got **97%** Accuracy (with and without clean background and proper lightning conditions) through our method. And if the background is clear and there is good lightning condition then we got even **99%** accurate results!

**Text To Speech Translation:** The model translates known gestures into words. we have used pyttsx3 library to convert the recognized words into the appropriate speech. The text-to-speech output is a simple workaround, but it's a useful feature because it simulates a real-life dialogue. 

## System Flowchart

<p align="center">
 <img src="https://user-images.githubusercontent.com/99630855/201490238-224f65aa-071f-473a-8c23-a9d60e0a47d8.png" alt="Image 1" width="400" height="500"/>
 </p>

## 🛠 Requirements
1. Python 3.9
2. opencv-python==4.7.0.72
3. mediapipe==0.10.21
4. tensorflow==2.10.0
5. protobuf==3.20.3
6. pyttsx3
7. cvzone
8. Hardware : Webcam
9. Operating System: Windows 8 and Above 
  
## ⚙️ Setup Instructions

**1. Clone or Download the Project**
Place the project folder on your system (e.g., Desktop).

**2. Create a Virtual Environment (only once)**
```bash
python -m venv slenv39
```

**3. Activate the Environment** (every time before running)
- Windows (PowerShell):
```bash
.\slenv39\Scriptsctivate
```
- Windows (Command Prompt):
```bash
slenv39\Scripts\activate
```
You should now see `(slenv39)` at the start of your terminal.

**4. Install Dependencies**(only once)
```bash
pip install opencv-python==4.7.0.72 mediapipe==0.10.21 tensorflow==2.10.0 protobuf==3.19.6 pyttsx3 cvzone
```

If `requirements.txt` is missing, install manually:
```bash
pip install opencv-python mediapipe tensorflow pyttsx3 cvzone
```

## 🚀 Running the Project

**Option 1: Run from PowerShell/Command Prompt**
```bash
python final_pred.py
```

**Option 2: Run from VS Code**
1. Open VS Code in the project folder.  
2. Select the **Python Interpreter** → `slenv39`.  
   *(Bottom-right corner → click → choose `.\slenv39\Scripts\python.exe`)*  
3. Open `final_pred.py`.  
4. Press **Run** (or `Ctrl + F5`).


## 🔧 Common Fixes

1. If you see `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` → reinstall protobuf:
```bash
pip install --upgrade protobuf==3.20.3
```

2. If nothing happens in VS Code but works in PowerShell → check interpreter is set to `slenv39`.
3. Always **activate the environment before running**.


## ✔️ Done!
Now when you run `final_pred.py`, the camera will open, detect hand signs, and speak out the prediction.

 

