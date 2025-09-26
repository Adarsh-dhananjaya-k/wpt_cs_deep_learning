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
<img width="955" height="630" alt="image" src="https://github.com/user-attachments/assets/65067539-beb4-4682-82a0-41a4320874ce" />


---

## 🟢 3. Neural Network Architecture  

- **Input Layer** → receives data  
- **Hidden Layers** → extract features  
- **Output Layer** → gives predictions  
 

A Neural Network is made up of layers of neurons connected together. Each layer has a specific role.

 - ### 1. Input Layer
---
This is the entry point of data into the network.
Each neuron in the input layer represents one feature of the dataset.

📌 Example: In digit recognition (MNIST):
Input = 28×28 pixel image = 784 features.
Input layer will have 784 neurons, one for each pixel.

- ### 2. Hidden Layers
---
Layers between input and output.
They apply weights + activation functions to transform inputs into useful patterns.
Each layer extracts higher-level features:
First hidden layer might detect edges in an image.
Next layers might detect shapes.
Final layers might detect digits or objects.
👉 More hidden layers = deeper network = Deep Learning.

- ### 3. Output Layer
---
Produces the final prediction.
Number of neurons = number of classes/outputs.

📌 Example:

For MNIST digit recognition → 10 neurons (digits 0–9).
Each neuron’s value = probability of that digit.

 - ### 📊 Diagram
---
Here’s a simple 3-layer neural network (Input → Hidden → Output):

         Input Layer (784 neurons for MNIST)
                ●   ●   ●   ●   ● ...  
                 \  |   / \   |  /
                  \ |  /   \  | /
                  Hidden Layer (128 neurons, ReLU)
                    ● ● ● ● ● ● ● ● ...
                       \   |   /
                        \  |  /
                        Output Layer (10 neurons, Softmax)
                        [0][1][2]...[9]


Input Layer → 784 pixels

Hidden Layer → 128 neurons with ReLU activation

Output Layer → 10 neurons with Softmax activation (probabilities for each digit)

🔹 Example in Keras (MNIST)
```python
# --- Setup & Training ---
import tensorflow as tf
from tensorflow.keras import layers, models

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Flatten (28x28 -> 784), cast to float32, normalize 0..1
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 784).astype("float32") / 255.0

# Simple MLP model
model = models.Sequential([
    tf.keras.Input(shape=(784,)),         # explicit Input layer (avoids warnings)
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(
    x_train, y_train,
    epochs=3, batch_size=128,
    validation_data=(x_test, y_test)
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"✅ Test accuracy: {test_acc:.4f}")

```


#### Quick Visual Check (Predict a few test images)
```python

import numpy as np
import matplotlib.pyplot as plt

preds = model.predict(x_test[:12], verbose=0)
y_pred = np.argmax(preds, axis=1)

plt.figure(figsize=(10,3))
for i in range(12):
    plt.subplot(2,6,i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap="gray")
    plt.title(f"Pred: {y_pred[i]}")
    plt.axis("off")
plt.suptitle("Sanity check on MNIST test samples")
plt.show()
````

📌 Output:

The network learns to recognize digits 0–9.

Accuracy after training (3 epochs) ≈ 97%.
📌 Example: Handwritten Digit Recognition (MNIST)  

 <img width="1400" height="925" alt="image" src="https://github.com/user-attachments/assets/13966d68-5aa3-44ce-8cc5-01f846eac8e7" />


---

## 🟢 4. Forward Propagation  

1. Inputs are multiplied with weights.  
2. Added with a bias.  
3. Passed through an **activation function**.  
4. Output flows forward until the final prediction.  


![forward_propagation](https://github.com/user-attachments/assets/1e2608e1-1874-4621-be2b-50d6ec307ab3)

📌 Formula:  
<img width="249" height="69" alt="image" src="https://github.com/user-attachments/assets/13d8ff00-d17e-4a23-9b5e-ff5bc8d05138" />

 

---

## 🟢 5. Activation Functions  

Activation functions decide how signals move forward.  
An activation function decides whether a neuron should “fire” (pass its signal forward) or not.
Without activation functions, neural networks would just be linear equations and could not learn complex patterns like images, speech, or language.

🔹 Why Do We Need Activation Functions?

Introduce non-linearity → allows the network to learn complex decision boundaries.

Normalize neuron outputs into a useful range (like 0–1).

Help in gradient flow during backpropagation.

### (a) Sigmoid  

<img width="208" height="90" alt="image" src="https://github.com/user-attachments/assets/476ab3f8-a821-4ce5-ab16-b4269bb8298a" />

\[ f(x) = \frac{1}{1+e^{-x}} \]  
Range: (0,1) → Good for probabilities.  

### (b) Tanh  
Range: (-1,1) → Better than sigmoid for centered data.  
<img width="211" height="67" alt="image" src="https://github.com/user-attachments/assets/da1a366b-c096-4056-80e5-b4dc90446f43" />


### (c) ReLU (Rectified Linear Unit)  
<img width="169" height="54" alt="image" src="https://github.com/user-attachments/assets/c6637ec9-cd61-4927-890e-890ce9e115b9" />

\[ f(x) = \max(0, x) \]  
Most commonly used in Deep Learning.  

### (d) Leaky ReLU  
Fixes “dead neurons” problem by allowing small negative slope.  
<img width="290" height="111" alt="image" src="https://github.com/user-attachments/assets/b268bc5c-399e-426b-8523-9df78874945f" />


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
A Cost Function (also called Loss Function) measures how far off the model’s predictions are from the actual (true) values.

<img width="304" height="93" alt="image" src="https://github.com/user-attachments/assets/c68cc64a-f59c-4d25-aa9f-ccfcfade3c12" />

```python
import numpy as np

y_true = np.array([1.5, 2.0, 3.5])
y_pred = np.array([1.4, 2.3, 3.0])

mse = np.mean((y_true - y_pred)**2)
print("MSE:", mse)
```

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



🔹 Popular Deep Learning Frameworks
---
### 1. TensorFlow

Developed by Google.

Supports both low-level operations (TensorFlow Core) and high-level APIs (Keras).

Built-in support for TensorBoard (visualization).

Strong for production deployment (e.g., TensorFlow Lite for mobile, TensorFlow Serving for cloud).

📌 Example:
```python
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
print("TensorFlow Addition:", a+b)
```
### 2. Keras

High-level API built on top of TensorFlow.
Focused on simplicity and rapid prototyping.
Ideal for beginners and quick model development.

📌 Example:
```pyhton
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Dense(64, activation="relu", input_shape=(100,)),
    layers.Dense(10, activation="softmax")
])
```
### 3. PyTorch

Developed by Facebook (Meta AI).
Dynamic computation graph → more flexible and Pythonic.
Very popular in research.
Strong ecosystem: torchvision, torchaudio, torchtext.

📌 Example:
``` python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3*x + 2
y.backward()
print("Gradient:", x.grad)
```
### 4. MXNet

Backed by Apache.
Scalable, supports multiple languages (Python, R, Scala).
Used by Amazon (AWS Deep Learning AMIs).

### 5. JAX

Developed by Google Research.
Combines NumPy syntax + automatic differentiation + XLA compiler.
Very fast, often used for cutting-edge ML research.

📌 Example:
```python
import jax.numpy as jnp
from jax import grad

f = lambda x: x**2 + 3*x + 2
df = grad(f)
print("Gradient at x=2:", df(2.0))
```

🟢 Keras Layers and APIs
---
Keras is a high-level deep learning API that makes building neural networks easy.
It runs on top of TensorFlow and provides clean abstractions to define and train models.

### 🔹 Types of APIs in Keras
#### 1. Sequential API (Beginner-Friendly)

The simplest way to build a model.

Layers are stacked one after another.

Good for feedforward networks.

📌 Example:
``` python
from tensorflow.keras import Sequential, layers

# Simple feedforward model
model = Sequential([
    layers.Dense(128, activation="relu", input_shape=(784,)),  # Hidden layer
    layers.Dense(64, activation="relu"),                       # Hidden layer
    layers.Dense(10, activation="softmax")                     # Output layer
])
```
#### 2. Functional API (Flexible)

Used for more complex models (multi-input, multi-output, branching).
Layers are treated as functions applied to tensors.

📌 Example:
``` python
from tensorflow.keras import Model, Input, layers

inputs = Input(shape=(784,))
x = layers.Dense(128, activation="relu")(inputs)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)
```

### 3. Model Subclassing (Advanced)

Allows full control by subclassing the Model class.
Useful for research and custom training loops.

📌 Example:
```python
from tensorflow.keras import Model, layers

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = layers.Dense(128, activation="relu")
        self.d2 = layers.Dense(64, activation="relu")
        self.out = layers.Dense(10, activation="softmax")

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.out(x)

model = MyModel()
```
###🔹 Commonly Used Keras Layers

- Dense (Fully Connected Layer)
```python 
layers.Dense(64, activation="relu")

```
Fully connected neurons, most common layer.

- Dropout (reduce overfitting)
```
layers.Dropout(0.5)
```

- Conv2D (for images)
```
layers.Conv2D(32, (3,3), activation="relu")
```

LSTM / GRU (for sequences, NLP)
```
layers.LSTM(64)
```

Flatten (convert 2D → 1D)
```python
layers.Flatten()
```
🔹 Model Compilation & Training
---
```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

```



🟢 Building Neural Networks with Keras
---

Now that we know about Keras APIs and layers, let’s build a complete neural network step by step using the MNIST dataset (handwritten digits).

-🔹 Step 1: Import Libraries

```python
import tensorflow as tf
from tensorflow.keras import Sequential, layers, datasets
```

-🔹 Step 2: Load the Dataset

We’ll use the MNIST dataset (60,000 training images + 10,000 test images).

```python
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)

```

-🔹 Step 3: Preprocess the Data
Flatten images (28×28 → 784).
Normalize pixel values (0–255 → 0–1).

```python
x_train = x_train.reshape(-1, 784).astype("float32") / 255
x_test  = x_test.reshape(-1, 784).astype("float32") / 255
```
-🔹 Step 4: Build the Model

We’ll use a Sequential model with:
Input Layer → 784 neurons
Hidden Layers → 128 + 64 neurons (ReLU)
Output Layer → 10 neurons (Softmax)

```python
model = Sequential([
    layers.Dense(128, activation="relu", input_shape=(784,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

```
-🔹 Step 5: Compile the Model

Choose:
Optimizer → Adam (fast + popular)
Loss Function → Sparse Categorical Crossentropy (multi-class classification)

Metrics → Accuracy

```pyhton
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

```
-🔹 Step 6: Train the Model

```python
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_data=(x_test, y_test)
)

```

Output will show training accuracy and validation accuracy improving over epochs.

-🔹 Step 7: Evaluate the Model

``` python
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("✅ Test Accuracy:", test_acc)

```

Typical accuracy ~ 97–98% after 5 epochs.

-🔹 Step 8: Make Predictions

``` python
import numpy as np

predictions = model.predict(x_test[:5])
for i, pred in enumerate(predictions):
    print("Image", i, "→ Predicted:", np.argmax(pred), "| True:", y_test[i])

```
-🔹 Visualization of Predictions
``` python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,3))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap="gray")
    plt.title(f"Pred: {np.argmax(predictions[i])}")
    plt.axis("off")
plt.show()

```

🟢 TensorFlow & TensorBoard Setup
---
TensorFlow is the engine that runs deep learning models, and TensorBoard is its built-in tool to visualize training progress (loss, accuracy, graphs).

-🔹 Installing TensorFlow

In Google Colab, TensorFlow is already installed.
If running locally:
``` python
pip install tensorflow
```
-🔹 Verifying Installation

```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices("GPU"))
```
🔹 Training a Model with TensorBoard Logging

TensorBoard records training metrics using a log directory.
``` python
import datetime
from tensorflow.keras import datasets, layers, Sequential

# Load dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255
x_test  = x_test.reshape(-1, 784).astype("float32") / 255

# Build model
model = Sequential([
    layers.Dense(128, activation="relu", input_shape=(784,)),
    layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Create log directory
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train model with TensorBoard
model.fit(x_train, y_train, epochs=5,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback])
```
🔹 Launch TensorBoard in Colab
```
%load_ext tensorboard
%tensorboard --logdir logs/fit
```


🟢 Natural Language Processing (NLP) Basics
---

Natural Language Processing (NLP) is a field of AI that enables computers to understand, interpret, and generate human language.
It is widely used in chatbots, translation, sentiment analysis, and search engines.

### 🔹 Key Preprocessing Steps in NLP

Before text can be used in a neural network, it must be cleaned and converted into numerical form.

- 1. Tokenization
Breaking text into smaller units (words or sentences).

📌 Example:
``` python
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download("punkt")

text = "Deep Learning is amazing. NLP makes machines understand language."
print("Word Tokenization:", word_tokenize(text))
print("Sentence Tokenization:", sent_tokenize(text))
```

- 2. Normalization

Standardizing text into a consistent form.
 - Lowercasing
 - Removing punctuation/numbers
 - Lemmatization/Stemming

📌 Example (Lemmatization):
``` python
from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
words = ["running", "flies", "better"]

print([lemmatizer.lemmatize(w, pos="v") for w in words])
```

Output:
```
['run', 'fly', 'better']
```
- 3. Stop Word Removal

Stop words = very common words (the, is, in, and…) that don’t carry much meaning.

📌 Example:
``` python
from nltk.corpus import stopwords
nltk.download("stopwords")

text = "Deep learning is a part of machine learning"
words = word_tokenize(text)
filtered = [w for w in words if w.lower() not in stopwords.words("english")]

print("Without stopwords:", filtered)
```

Output:
```
['Deep', 'learning', 'part', 'machine', 'learning']
```
- 4. Vectorization

Convert text into numbers so ML models can understand.
 - Bag of Words (BoW)
 - TF-IDF (Term Frequency–Inverse Document Frequency)
 - Word Embeddings (Word2Vec, GloVe, BERT)

📌 Example: TF-IDF
```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "Deep learning is amazing",
    "Natural language processing is powerful",
    "Deep learning powers NLP"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("TF-IDF Matrix:\n", X.toarray())
```


🟢 NLP Application: Sentiment Analysis (Text Classification)
---
Now that we know how to preprocess text (tokenization, normalization, stop word removal, vectorization), let’s apply NLP to a real task → Sentiment Analysis.

- 🔹 What is Sentiment Analysis?

A classification problem where we determine if a piece of text is positive, negative, or neutral.
Widely used in:
Customer reviews (Amazon, Flipkart, etc.)
Social media analysis (Twitter, Facebook)
Chatbots & feedback systems

- 🔹 Approach 1: Using TF-IDF + Logistic Regression
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Sample dataset
texts = [
    "I love this product, it is amazing!",
    "Worst purchase ever, very disappointed.",
    "Absolutely fantastic service and quality",
    "Terrible experience, I hate it",
    "Not bad, could be better"
]
labels = [1, 0, 1, 0, 1]  # 1 = Positive, 0 = Negative

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Test prediction
print("Prediction:", clf.predict(X_test))
print("True Labels:", y_test)
```

- 🔹 Approach 2: Using Pretrained Transformer (HuggingFace)
```python
!pip install transformers --quiet
from transformers import pipeline

# Load pretrained sentiment-analysis model
classifier = pipeline("sentiment-analysis")

print(classifier("I absolutely loved this movie!"))
print(classifier("This was the worst experience ever."))
```

Output:

```
[{'label': 'POSITIVE', 'score': 0.9998}]
[{'label': 'NEGATIVE', 'score': 0.9997}]
```

- 🔹 Visualizing Predictions

```python
reviews = [
    "The phone battery life is great",
    "The screen cracked within a week",
    "Customer support was very helpful",
    "I will never buy from here again"
]

results = classifier(reviews)

for review, res in zip(reviews, results):
    print(f"{review} → {res['label']} ({res['score']:.2f})")
```
