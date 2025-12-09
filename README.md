**Medical X-Ray Classification: Normal vs. Pneumonia**

This project builds a deep-learning model to classify chest X-ray images as Normal or Pneumonia using ResNet50 transfer learning. It includes preprocessing, training, evaluation, and Grad-CAM explainability to highlight lung regions influencing predictions.

**Features**

ResNet50 transfer learning (freeze → fine-tune)

Binary image classification

Accuracy/Loss curves

Confusion matrix

Grad-CAM visual explanations

Early stopping + model checkpoints

**Dataset**

Kaggle Chest X-Ray dataset with two classes:

Normal

Pneumonia

Images are resized, normalized, and split into train/val/test.

**How to Run**
pip install -r requirements.txt
python train.py
python evaluate.py


Or run the provided .ipynb notebook sequentially.

**Results**

High accuracy on validation/test sets

Clear separation between classes

Grad-CAM highlights infected lung regions

**Project Structure**
/data
/models
/outputs
train.py
evaluate.py
utils.py
README.md

**Future Work**

Add more chest pathologies

Deploy via Streamlit/Flask

Use EfficientNet or DenseNet
