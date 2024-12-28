# ✋ **Gesture-Based Numeric Drawing and Real-Time Calculation System** ✨

This project revolutionizes how we interact with technology by enabling users to perform numeric inputs and calculations through **hand gestures**. Designed for accessibility, education, and interactive systems, this solution leverages **AI** and **Computer Vision** to provide an intuitive, real-time interface for drawing digits and symbols. Say goodbye to traditional keyboards and hello to a seamless interaction experience!

---

## 🛠️ **Features**

- 🎥 **Real-Time Gesture Tracking:** Utilizes **Mediapipe** to track hand gestures and landmarks with high precision.  
- 🖼️ **Interactive Drawing Interface:** Built with **OpenCV**, providing a canvas for users to draw numeric inputs and symbols like `+`, `-`, `*`, and `/`.  
- 🧠 **Custom Trained Model:** A CNN model trained from scratch on a **10,000-image dataset** of hand-drawn digits and mathematical symbols ensures accuracy and adaptability.  
- ⚡ **Real-Time Calculation:** Dynamically parses and computes user-drawn equations using Python’s `eval`.  
- 💻 **Accessible and Adaptive:** Designed for diverse environments, including **smart classrooms**, **assistive technologies**, and **digital whiteboards**.  
- 🔄 **Data Augmentation:** Enhances training data through techniques like rotation, flipping, and noise addition for better generalization.  

---

## 📂 **Project Overview**

This project bridges the gap between humans and machines by introducing a gesture-based interface. The system allows users to draw digits and symbols naturally with their hands, which are then recognized and processed for real-time calculations.  

### **Core Components**  
1. **Hand Tracking:** Detects and tracks hand gestures using **Mediapipe**, ensuring smooth and accurate drawing.  
2. **Drawing Interface:** Creates a virtual canvas using **OpenCV**, capturing user input dynamically.  
3. **Model Training:** A custom **Convolutional Neural Network (CNN)** model trained on **10,000 hand-drawn images** from Kaggle to recognize digits (`0-9`) and symbols (`+`, `-`, `*`, `/`).  
4. **Real-Time Computation:** Converts recognized symbols into mathematical equations and evaluates them instantly.  

---

## 🛠️ **Tools & Technologies Used**

### **Programming Language**  
- 🐍 **Python**  

### **Libraries & Frameworks**  
- 📸 **Mediapipe:** For hand gesture tracking.  
- 🖼️ **OpenCV:** For virtual drawing interface.  
- 🧠 **TensorFlow/Keras:** For training and deploying the CNN model.  
- 📊 **NumPy:** For numerical computations.  

### **Software & Editors**  
- 💻 **VS Code** / **PyCharm:** For development.  
- 📦 **Kaggle Datasets:** A curated dataset of hand-drawn images for training.  

### **Hardware Requirements**  
- 🎥 Webcam  
- 🖥️ Laptop or PC with at least **8GB RAM** and **2GB VRAM**  

---

## 📈 **System Workflow**

1. **Data Capture:**  
   - Captures real-time video using a webcam.  
   - Tracks hand landmarks to create a drawing on the canvas.  

2. **Preprocessing:**  
   - Converts drawings to grayscale, resizes them to 28x28 pixels, and normalizes them for model input.  

3. **Recognition:**  
   - A custom-trained CNN model processes the drawings to identify digits and symbols.  

4. **Equation Parsing:**  
   - Recognized symbols are converted into a mathematical equation.  

5. **Calculation:**  
   - The equation is evaluated dynamically, and the result is displayed on the interface.  

---

## 🔑 **Key Highlights**

- **Custom Model:** Our CNN model, trained on **10,000 images**, achieves high accuracy for digit and symbol recognition.  
- **Low Latency:** Processes inputs and computations in real-time, ensuring smooth user interaction.  
- **Robust Design:** Handles varying lighting conditions and drawing styles effectively.  
- **Scalability:** Designed to adapt to more complex symbols and larger datasets in the future.  

---

## 📚 **Applications**

- 🎓 **Education:**  
  Enhance teaching and learning experiences in smart classrooms.  

- ♿ **Accessibility:**  
  Assistive tools for individuals with motor disabilities, providing alternative input methods.  

- 💼 **Professional Use:**  
  Ideal for collaborative whiteboarding and brainstorming in digital workspaces.  

- 🎨 **Creative Spaces:**  
  Integrate gesture-based interactions for immersive artistic and interactive exhibits.  

---

## 🚀 **Future Scope**

1. Expand the model to include more complex mathematical symbols like parentheses and trigonometric functions.  
2. Integrate with **Augmented Reality (AR)** for immersive experiences.  
3. Develop **mobile and web-based versions** for cross-platform compatibility.  
4. Add **voice and haptic feedback** for enhanced accessibility.  
5. Train the model on diverse datasets to support multilingual symbols and inputs.  

---

## 🤝 **Contributions**

Contributions are welcome! If you’d like to improve the project or explore new features, feel free to fork the repository and submit a pull request.  

---

## 📜 **License**

This project is licensed under the **MIT License** – you’re free to use, modify, and distribute it as long as proper credit is given.  

---

## 📨 **Contact**

For questions or collaborations, reach out to:  
- **Prabhat Kumar**  
  - LinkedIn: [Prabhat Kumar](https://www.linkedin.com/in/prabhat-kumar-1260a5259)  
  - Email: [prabhatsharma84226@gmail.com](mailto:prabhatsharma84226@gmail.com)  
