
# 👟 Pix2Pix Shoe Image Generator

A web application that uses **Generative Adversarial Networks (GANs)** to generate realistic shoe images based on user-drawn sketches. This project leverages the **Pix2Pix** model, allowing users to sketch shoe outlines using a free-hand drawing tool, then transforms these sketches into colorful, detailed shoe images. 🎨

## 📝 Introduction

This project demonstrates the power of **Generative Adversarial Networks (GANs)** and **deep learning** in image-to-image translation. By implementing the Pix2Pix model, the web app can transform user-generated shoe outlines into realistic shoe images, showcasing the creative potential of generative AI.

## 💻 Description

The Pix2Pix Shoe Image Generator is a machine learning application built using **React** on the front end and **Flask** on the back end. Users can draw a shoe outline using a free-hand drawing tool and submit it to the model, which processes the sketch and generates a corresponding realistic shoe image using a pre-trained **Pix2Pix GAN** model. The project highlights the exciting capabilities of neural networks in generating detailed and coherent images from simple sketches.

## Screenshots

<p align="center">
  <img src="https://github.com/sufiyanpatel27/Pix2Pix/blob/main//assets/HomePage1.png?raw=true" alt="Alt Text" width="700">
  <img src="https://github.com/sufiyanpatel27/Pix2Pix/blob/main//assets/HomePage2.png?raw=true" alt="Alt Text" width="700">
  <img src="https://github.com/sufiyanpatel27/Pix2Pix/blob/main//assets/HomePage3.png?raw=true" alt="Alt Text" width="700">
  <img src="https://github.com/sufiyanpatel27/Pix2Pix/blob/main//assets/HomePage4.png?raw=true" alt="Alt Text" width="700">
  <img src="https://github.com/sufiyanpatel27/Pix2Pix/blob/main//assets/HomePage5.png?raw=true" alt="Alt Text" width="700">
  <img src="https://github.com/sufiyanpatel27/Pix2Pix/blob/main//assets/HomePage6.png?raw=true" alt="Alt Text" width="700">
</p>

### 🌟 Features

- ✏️ Free-hand drawing tool for shoe sketches.
- ⚡ Real-time image generation based on sketches using GANs.
- 🚀 Fast and efficient deep learning model integration.
- 💻 Frontend built with **React** for a seamless user experience.
- 🛠️ Backend built with **Flask** to handle model inference and image generation.

### 🛠️ Tech Stack

- **Frontend:** React, HTML5 Canvas (for free-hand drawing)
- **Backend:** Flask
- **Machine Learning Frameworks:** TensorFlow, Keras
- **Model:** Pix2Pix Generative Adversarial Network (GAN)
- **Languages:** JavaScript, Python
- **Deployment:** Flask server for backend, web app frontend

## 🚧 Challenges Faced

- **Model Training:** One of the key challenges was training the Pix2Pix GAN model. The process involved a lot of trial and error, tuning hyperparameters to achieve high-quality shoe image generation.
- **Integration of Free-hand Drawing Tool:** Implementing a smooth and responsive drawing tool on the web that integrates well with the model’s input format was a challenge.
- **Performance Optimization:** Ensuring that the model inference runs quickly enough for a real-time user experience on both local and deployed environments.

## 🚀 Future Enhancements

- **Model Improvement:** Further fine-tuning of the Pix2Pix model to generate even more detailed shoe images.
- **Support for Multiple Types of Footwear:** Extend the functionality to allow users to sketch other types of footwear such as sandals or boots. 👡👢
- **Mobile-Friendly Interface:** Enhancing the UI for mobile devices to provide a seamless experience across all platforms. 📱
- **User Accounts and Gallery:** Allow users to create accounts and save their generated shoe designs in a gallery for later use. 🖼️
- **Cloud Deployment:** Deploy the app on a cloud platform like AWS or GCP for scalability and performance improvements. ☁️

## 🛠️ Installation

### Prerequisites

- Python 3.x
- Node.js
- TensorFlow and Keras
- Flask

### Project Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/sufiyanpatel27/Pix2Pix.git
   cd Pix2Pix/
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the Flask server:
   ```bash
   python server.py
   ```


### 🚀 Usage

1. Open your browser and go to `http://localhost:5000`. 🌐
2. Use the free-hand drawing tool to sketch a shoe outline.
3. Click "Submit" to generate the shoe image. 🎨
4. The Flask server processes the input, and within a few moments, the generated shoe image based on your sketch will be displayed. 👟

## 📚 Resources

- [Pix2Pix Paper](https://arxiv.org/abs/1611.07004) – The original paper describing the Pix2Pix GAN model.
- [TensorFlow Documentation](https://www.tensorflow.org/) – Documentation for TensorFlow and Keras.
- [Flask Documentation](https://flask.palletsprojects.com/en/2.0.x/) – Documentation for Flask.
- [Dataset](https://www.kaggle.com/datasets/balraj98/edges2shoes-dataset) - Dataset contains around 50k shoe images.
