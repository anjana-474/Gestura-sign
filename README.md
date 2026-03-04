<h1 align="center">🤟 GESTURA</h1>
<p align="center">
Sign Language Translation using Computer Vision & Deep Learning
</p>

<p align="center">

<img src="https://img.shields.io/badge/Python-3.x-blue" />
<img src="https://img.shields.io/badge/PyTorch-DeepLearning-red" />
<img src="https://img.shields.io/badge/MediaPipe-HandTracking-green" />
<img src="https://img.shields.io/badge/Flask-WebApp-lightgrey" />
<img src="https://img.shields.io/badge/Status-Active-success" />

</p>

---

## 📌 Overview

**GESTURA** is a real-time sign language translation system that converts hand gestures into meaningful text and multilingual speech using computer vision and deep learning.

The system works using a standard webcam and recognizes both:

* ✋ Static alphabet gestures
* 🔄 Dynamic word-level gestures

Recognized gestures are combined into meaningful sentences and converted into speech, enabling smoother communication between sign language users and non-signers.

---

## ✨ Highlights

* Lightweight models (runs on CPU)
* Real-time inference using MediaPipe landmarks
* Static + dynamic gesture recognition
* Conversation module for sentence generation
* Multilingual text-to-speech output
* Flask-based interactive web interface

---

## 🧠 Models Used

### 🔹 Alphabet Recognition

* Model: Multi-Layer Perceptron (MLP)
* Input: 63-dimensional hand landmark features
* Accuracy: **99.54%**

### 🔹 Dynamic Gesture Recognition

* Model: Bidirectional LSTM (BiLSTM)
* Input: 30-frame sequence (126 features)
* Accuracy: **98.56%**

---

## ⚙️ Tech Stack

* Python
* PyTorch
* MediaPipe
* OpenCV
* Flask
* HTML / CSS / JavaScript
* Text-to-Speech Libraries

---

## 🏗️ System Architecture

1. Webcam Input  
2. MediaPipe Hand Landmark Extraction  
3. Alphabet Recognition (MLP)  
4. Dynamic Gesture Recognition (BiLSTM)  
5. Conversation Module (Sentence Formation)  
6. Multilingual Text-to-Speech Output
---

## ▶️ Installation & Run

### Clone Repository

```bash
git clone https://github.com/anjana-474/Gestura-sign.git
cd Gestura-sign
```

### Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open browser:

```
http://127.0.0.1:5000
```

---

## 🎯 Goal

To build an accessible real-time sign language translation system that works without expensive hardware and helps bridge communication gaps in everyday situations.

---

## 🔮 Future Improvements

* Larger multi-user dataset
* Facial expression & pose integration
* Mobile deployment
* Support for regional sign languages

---

## 🎥 Demo

Gestura - Sign Language Translation using webcam input.

⚠️ This project uses real-time webcam input, so the full system runs locally.

🎬 Demo video: 

[![Watch the demo](https://img.youtube.com/vi/vif9UEqb45w/0.jpg)](https://youtu.be/vif9UEqb45w) 

---
⭐ If you like this project, consider giving it a star!



