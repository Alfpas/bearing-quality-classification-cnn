# Bearing Quality Classification System (CNN)

An end-to-end system for automated bearing quality inspection using 
computer vision and deep learning.

---

## 🚀 Overview
This project aims to classify bearing conditions into multiple categories 
using a Convolutional Neural Network (CNN). The system is designed as a 
prototype for industrial quality control applications.

---

## 🧠 Features
- Image classification using CNN (ResNet50 / MobileNetV2)
- Multi-class classification (e.g., Healthy, Defect, No Bearing)
- Scalable for real-time inspection system
- Can be integrated with conveyor-based hardware system

---

## 🛠️ Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

---

## 📂 Project Structure
- src/ → training & prediction scripts
- results/ → evaluation results (confusion matrix, plots)
- sample_data/ → example images (not full dataset)
 
---

## 📊 Dataset
The dataset was collected and annotated using Roboflow.

- Classes: Healthy, Defect, No Bearing
- Augmentation: rotation, flipping, brightness adjustment

⚠️ Note: Only sample data is included in this repository.  
Full dataset is available externally.

---

## ▶️ How to Run
1. Install dependencies: pip install -r requirements.txt
2. Train model:python src/train.py
3. Run prediction:python src/predict.py


---

## 📈 Results
- Model: ResNet50 (Fine-tuned)
- Achieved high accuracy on controlled dataset
- Performance may vary in real-world environment (e.g., different background)

---

## ⚠️ Challenges & Notes
- Dataset captured with controlled background (black)
- Deployment may involve different environments (e.g., green conveyor)
- Potential domain shift needs to be handled

---

## 📌 Future Work
- Real-time deployment on conveyor system
- Integration with embedded systems (ESP32 / IoT)
- Model optimization for edge devices

---

## 👨‍💻 Author
Alfredo Bona Gabe Pasaribu  
Robotics | IoT | Computer Vision | Embedded Systems
