package org.example;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class NeuralNetworkTest {

    @Test
    public void testNeuralNetworkOutputSize() {
        // Define input
        List<Double> inputs = List.of(1.0, 2.0, 3.0);

        // Create a neural network with 3 neurons in the output layer
        int[] neuronsPerLayer = {4, 3}; // Output layer should have 3 neurons
        NeuralNetwork network = new NeuralNetwork(neuronsPerLayer, inputs.size());

        // Calculate outputs
        List<Double> outputs = network.feedForward(inputs);

        // Assert that the output size matches the number of neurons in the last layer
        assertEquals(3, outputs.size());
    }

    @Test
    public void testNeuralNetworkSingleNeuronOutput() {
        // Define input
        List<Double> inputs = List.of(1.0, 2.0, 3.0);

        // Create a neural network with 1 neuron in the output layer
        int[] neuronsPerLayer = {2, 1}; // Output layer should have 1 neuron
        NeuralNetwork network = new NeuralNetwork(neuronsPerLayer, inputs.size());

        // Calculate outputs
        List<Double> outputs = network.feedForward(inputs);

        // Assert that the network produces a single output
        assertEquals(1, outputs.size());
    }

    @Test
    public void testSoftmaxOutput() {
        // Define input
        List<Double> inputs = List.of(1.0, 2.0, 3.0);

        // Create a neural network with 3 neurons in the output layer
        int[] neuronsPerLayer = {3}; // Output layer should have 3 neurons
        NeuralNetwork network = new NeuralNetwork(neuronsPerLayer, inputs.size());

        // Forward pass through the network
        List<Double> outputs = network.feedForward(inputs);
        double sum = outputs.stream().mapToDouble(Double::doubleValue).sum();

        assertEquals(1.0, sum, 0.001);
    }
}