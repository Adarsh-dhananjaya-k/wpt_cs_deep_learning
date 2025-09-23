# 📘 Introduction to Deep Learning  

Deep Learning is a **subset of Machine Learning** where algorithms learn patterns from data using **neural networks with many layers**.  
It is inspired by how the **human brain processes information** using neurons.  

---

## 🟢 1. Why Deep Learning?  

- Traditional **Machine Learning** requires manual feature extraction.  
  Example: In image recognition, you need to manually extract edges, shapes, etc.  
- Deep Learning automatically learns **hierarchical features** from raw data.  

👉 Applications:  
- Face unlock on smartphones  
- Self-driving cars  
- Chatbots & voice assistants  
- Healthcare (X-ray, MRI scan analysis)  

---

## 🟢 2. Biological Neuron vs Artificial Neuron  

A **biological neuron** has:  
- **Dendrites** (inputs)  
- **Soma** (cell body – processes inputs)  
- **Axon** (output signal)  

An **artificial neuron (perceptron)** has:  
- **Inputs (x)** → features from dataset  
- **Weights (w)** → importance of each input  
- **Summation (Σ w·x + b)**  
- **Activation function** → decides output  

📌 Diagram (Neuron analogy):  
![Neuron Analogy](https://miro.medium.com/v2/resize:fit:720/format:webp/1*ZDKuwZQzQ5d6gd0dfzpOwA.png)  

---

## 🟢 3. Neural Network Architecture  

- **Input Layer** → receives data  
- **Hidden Layers** → extract features  
- **Output Layer** → gives predictions  

📌 Example: Handwritten Digit Recognition (MNIST)  

![Neural Network](https://www.researchgate.net/profile/Dimitrios-Kolovos/publication/324532215/figure/fig1/AS:618216982552577@1523641469850/A-simple-artificial-neural-network-with-two-hidden-layers.png)  

---

## 🟢 4. Forward Propagation  

1. Inputs are multiplied with weights.  
2. Added with a bias.  
3. Passed through an **activation function**.  
4. Output flows forward until the final prediction.  

📌 Formula:  
\[ y = f(\sum (w_i \cdot x_i) + b) \]  

---

## 🟢 5. Activation Functions  

Activation functions decide how signals move forward.  

### (a) Sigmoid  
\[ f(x) = \frac{1}{1+e^{-x}} \]  
Range: (0,1) → Good for probabilities.  

### (b) Tanh  
Range: (-1,1) → Better than sigmoid for centered data.  

### (c) ReLU (Rectified Linear Unit)  
\[ f(x) = \max(0, x) \]  
Most commonly used in Deep Learning.  

### (d) Leaky ReLU  
Fixes “dead neurons” problem by allowing small negative slope.  

🔹 **Code: Visualize Activation Functions**  
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)

def sigmoid(x): return 1/(1+np.exp(-x))
def tanh(x): return np.tanh(x)
def relu(x): return np.maximum(0,x)
def leaky_relu(x): return np.where(x > 0, x, 0.1*x)

plt.plot(x, sigmoid(x), label="Sigmoid")
plt.plot(x, tanh(x), label="Tanh")
plt.plot(x, relu(x), label="ReLU")
plt.plot(x, leaky_relu(x), label="Leaky ReLU")
plt.legend()
plt.title("Activation Functions")
plt.grid()
plt.show()
```

📌 Output graph will show **different activation functions**.  

---

## 🟢 6. Cost Function (Error Function)  

- Measures difference between **predicted output vs actual output**.  
- Example: **Mean Squared Error (MSE)**  

\[ MSE = \frac{1}{n}\sum(y_{true} - y_{pred})^2 \]  

---

## 🟢 7. Gradient Descent (Learning Process)  

- Optimizer that reduces cost function step by step.  
- Updates weights by moving in the direction of **steepest descent**.  

📌 Formula:  
\[ w = w - \eta \cdot \frac{\partial L}{\partial w} \]  

Where:  
- \( w \) = weights  
- \( \eta \) = learning rate  
- \( L \) = loss function  

🔹 **Code: Simple Gradient Descent Simulation**  
```python
import numpy as np
import matplotlib.pyplot as plt

# Loss function: y = x^2
x = np.linspace(-10, 10, 100)
y = x**2
plt.plot(x, y, label="Loss Function")

# Gradient descent simulation
cur_x = 8
lr = 0.1
epochs = 15
x_history = [cur_x]
for i in range(epochs):
    grad = 2*cur_x   # derivative of x^2
    cur_x = cur_x - lr * grad
    x_history.append(cur_x)

plt.scatter(x_history, [i**2 for i in x_history], color="red")
plt.legend()
plt.title("Gradient Descent Steps")
plt.show()
```

📌 The red dots will show how the optimizer moves towards the minimum.  

---

## 🟢 8. Hands-on Example: Simple Neural Network with Keras  

```python
import tensorflow as tf
from tensorflow.keras import layers

# Simple model
model = tf.keras.Sequential([
    layers.Dense(4, activation="relu", input_shape=(2,)),  # input = 2 features
    layers.Dense(1, activation="sigmoid")                  # output = probability
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Dummy data (XOR problem)
import numpy as np
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])   # XOR output

model.fit(X, y, epochs=100, verbose=0)
print("Predictions:", model.predict(X))
```

📌 This trains a neural network to learn the **XOR problem**.  

---

# ✅ Summary  

- Deep Learning uses **multi-layer neural networks**.  
- **Neurons** take weighted inputs, apply activation, and pass outputs.  
- Training uses **loss functions + gradient descent**.  
- **Keras/TensorFlow** makes it easy to build models.  
