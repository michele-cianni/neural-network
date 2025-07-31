# Neural Network Implementation in Java

A simple, educational implementation of a multi-layer neural network from scratch in Java, featuring forward propagation, backpropagation, and training capabilities.

## 🚀 Features

- **Multi-layer Neural Network**: Configurable architecture with customizable layers and neurons
- **Forward Propagation**: Complete forward pass through all layers
- **Backpropagation**: Gradient descent with error backpropagation for training
- **Softmax Activation**: Output layer uses softmax for multi-class classification
- **Xavier Weight Initialization**: Proper weight initialization for better training
- **Cross-entropy Loss**: Suitable for classification tasks
- **Training Loop**: Complete training implementation with epoch-based learning

## 🏗️ Architecture

The neural network consists of several key components:

- **NeuralNetwork**: Main orchestrator that manages layers and training
- **Layer**: Contains multiple neurons and handles layer-wise operations  
- **Neuron**: Individual processing unit with weights and bias
- **NeuralComponent**: Interface defining the contract for neural components

## 📋 Requirements

- Java 11 or higher
- Gradle 8.9 or higher

## 🛠️ Installation & Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/michele-cianni/neural-network.git
   cd neural-network
   ```

2. **Build the project**:

   ```bash
   ./gradlew build
   ```

3. **Run the application**:

   ```bash
   ./gradlew run
   ```

## 🎯 Usage

### Basic Example

The main application demonstrates training a neural network on a simple XOR-like problem:

```java
// Create a neural network with 2 input neurons, 4 hidden neurons, and 3 output neurons
int[] neuronsPerLayer = { 4, 3 };
int inputSize = 2;
NeuralNetwork neuralNetwork = new NeuralNetwork(neuronsPerLayer, inputSize);

// Training data (XOR-like problem)
List<List<Double>> trainingData = Arrays.asList(
    Arrays.asList(0.0, 0.0),
    Arrays.asList(0.0, 1.0),
    Arrays.asList(1.0, 0.0),
    Arrays.asList(1.0, 1.0)
);

List<Integer> trainingLabels = Arrays.asList(0, 1, 1, 0);

// Train the network
double learningRate = 0.1;
int epochs = 1000;

for (int epoch = 0; epoch < epochs; epoch++) {
    for (int i = 0; i < trainingData.size(); i++) {
        List<Double> input = trainingData.get(i);
        int actualLabel = trainingLabels.get(i);
        
        // Forward pass
        List<Double> predicted = neuralNetwork.feedForward(input);
        
        // Backward pass
        neuralNetwork.backpropagate(input, predicted, actualLabel, learningRate);
    }
}
```

### Creating Custom Networks

You can easily create networks with different architectures:

```java
// Create a network with multiple hidden layers
int[] neuronsPerLayer = { 8, 6, 4, 3 }; // 3 hidden layers + output layer
int inputSize = 5; // 5 input features
NeuralNetwork customNetwork = new NeuralNetwork(neuronsPerLayer, inputSize);
```

## 🧪 Testing

Run the test suite to verify the implementation:

```bash
./gradlew test
```

The project includes comprehensive tests for:

- Individual neuron functionality
- Layer operations
- Complete neural network training and prediction

## 📊 Example Output

When you run the application, you'll see training progress and test predictions:

```
=== Testing Neural Network ===
Training the neural network...
Epoch 0, Average Loss: 1.0986
Epoch 100, Average Loss: 0.8234
Epoch 200, Average Loss: 0.6123
...
Training completed.

=== Testing Predictions ===
Input: [0.0, 0.0], Predicted: Class 0 (0.892), Actual: Class 0, Correct: V
Input: [0.0, 1.0], Predicted: Class 1 (0.834), Actual: Class 1, Correct: V
Input: [1.0, 0.0], Predicted: Class 1 (0.798), Actual: Class 1, Correct: V
Input: [1.0, 1.0], Predicted: Class 0 (0.756), Actual: Class 0, Correct: V
```

## 🔧 Configuration

### Network Architecture

- Modify `neuronsPerLayer` array to change the number of layers and neurons per layer
- Adjust `inputSize` based on your input data dimensions

### Training Parameters

- `learningRate`: Controls how much weights are updated (default: 0.1)
- `epochs`: Number of training iterations (default: 1000)

### Weight Initialization

The network uses Xavier initialization for better convergence:

```java
double range = Math.sqrt(6.0 / numberOfInputs);
```

## 📁 Project Structure

```
app/
├── src/main/java/org/example/
│   ├── App.java                    # Main application entry point and training example
│   └── api/
│       ├── NeuralNetwork.java      # Neural network core class
│       ├── Layer.java              # Layer abstraction and logic
│       ├── Neuron.java             # Neuron implementation
│       └── NeuralComponent.java    # Interface for neural components
│
└── src/test/java/org/example/
    └── api/
        ├── NeuralNetworkTest.java  # Tests for NeuralNetwork
        ├── LayerTest.java          # Tests for Layer
        └── NeuronTest.java         # Tests for Neuron
```

## 🚧 Limitations

This is an educational implementation with the following limitations:

- Only supports fully connected (dense) layers
- Single activation function (ReLU for hidden layers, softmax for output)
- Basic gradient descent (no momentum or advanced optimizers)
- No regularization techniques implemented
- Limited to classification tasks

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🎓 Educational Purpose

This implementation is designed for educational purposes to demonstrate:

- How neural networks work under the hood
- Forward and backward propagation algorithms
- Gradient descent optimization
- Object-oriented design patterns in ML

Perfect for students and developers wanting to understand neural network fundamentals!
