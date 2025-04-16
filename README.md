# Facial Emotion Recognition System

A deep learning-based facial emotion recognition system that can detect and classify seven different emotions from facial expressions. Built with Streamlit and TensorFlow, this application provides a user-friendly interface for real-time emotion detection.

## Features

- Real-time emotion detection from facial images
- Support for seven emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral
- User-friendly web interface built with Streamlit
- Pre-trained deep learning model using CNN architecture
- Support for both custom image uploads and sample images
- Confidence score for predictions

## Project Structure

```
facial-recognition-system/
├── data/
│   ├── test/
│   └── train/
├── main.py
├── model.weights.h5
├── dataset-cover.png
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.x
- Required Python packages (versions specified in requirements.txt):
  - numpy==2.2.4
  - opencv-python==4.10.0.84
  - Pillow==11.2.1
  - streamlit==1.42.2
  - tensorflow==2.17.0
  - tensorflow-intel==2.17.0

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/facial-recognition-system.git
cd facial-recognition-system
```

2. Install the required packages using the requirements.txt file:
```bash
pip install -r requirements.txt
```

### Dataset Setup

1. Download the FER2013 dataset from Kaggle: [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
2. Unzip the downloaded dataset
3. Place the dataset in the `data` folder with the following structure:
```
data/
├── test/
└── train/
```

## Running the Application

1. Make sure you have all the prerequisites installed and the dataset properly set up
2. Run the Streamlit application:
```bash
streamlit run main.py
```
3. The application will be available at http://localhost:8501

## Usage

The application provides three main sections:

1. **Home**: Overview of the project and emotion categories
2. **About Project**: Detailed information about the dataset, model architecture, and implementation
3. **Prediction**: Interface for emotion detection
   - Upload custom images
   - Select from sample images
   - View prediction results with confidence scores

## Model Architecture

The system uses a Convolutional Neural Network (CNN) with the following architecture:
- Multiple convolutional layers with ReLU activation
- MaxPooling layers for dimension reduction
- Dropout layers for regularization
- Dense layers for classification
- Softmax activation for final emotion prediction

## License

This project is licensed under the MIT License - see the LICENSE file for details. 

