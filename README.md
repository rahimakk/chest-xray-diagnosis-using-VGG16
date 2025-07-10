# Chest X-Ray Pneumonia Classification using VGG16

This project demonstrates a binary image classification system for detecting **Pneumonia** from chest X-ray images using a pre-trained **VGG16** model and transfer learning. The model was trained and evaluated on the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset in a Kaggle Notebook.

## Dataset

The dataset contains chest X-ray images categorized into two classes: **NORMAL** and **PNEUMONIA**. It is divided into:
- `train/` - for training the model
- `val/` - for validation
- `test/` - for final evaluation

Dataset used:  
🔗 [Chest X-Ray Images (Pneumonia) on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Model Architecture

- **Base model**: VGG16 (pre-trained on ImageNet)
- **Added layers**:
  - Flatten
  - Dense(128, ReLU)
  - Dropout(0.5)
  - Dense(1, Sigmoid) for binary classification

## Techniques Used

- Transfer Learning with frozen base layers
- Binary Crossentropy loss function
- Adam optimizer
- Evaluation using Confusion Matrix and Classification Report
- Visual prediction results shown using Matplotlib

## Model Performance

The model was trained and tested in multiple runs. It showed consistent behavior in correctly identifying most pneumonia cases, with fewer false negatives — making it useful as a diagnostic assist tool.

> Note: Since the dataset is small and dropout is used, results (accuracy, precision, recall) can vary slightly on each run.

## Sample Predictions

Visualizations are provided showing model predictions on:
- 8 images from the `NORMAL` class
- 8 images from the `PNEUMONIA` class

Each image shows the predicted label and confidence score.

##  Trained Model (.h5)

The trained model is available here:  
 [Download from Kaggle](https://www.kaggle.com/datasets/rahimak/vgg16-pneumonia-model)

To use it in a notebook:

```python
from tensorflow.keras.models import load_model
model = load_model('/kaggle/input/vgg16-pneumonia-model/vgg16_pneumonia_model.h5')
```
##  How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/rahimakk/chest-xray-diagnosis-using-VGG16.git
   cd chest-xray-diagnosis-using-VGG16
2. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook

3. Make sure the following Python packages are installed:
   - tensorflow
   - keras
   - numpy
   - matplotlib
   - seaborn
   - sklearn


---

##  Acknowledgements

- [VGG16 – Keras Applications](https://keras.io/api/applications/vgg/)
- [Chest X-Ray Images (Pneumonia) Dataset on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- This project was built and trained in the **Kaggle notebook environment** using **TensorFlow** and **Keras**.
