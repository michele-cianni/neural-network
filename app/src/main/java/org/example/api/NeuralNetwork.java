package org.example.api;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Represents a neural network composed of multiple layers.
 * Each layer can perform feedforward operations and backpropagation.
 * 
 * @author Michele Cianni
 * @version 1.0
 * @see NeuralComponent
 * @see Layer
 */
public final class NeuralNetwork implements NeuralComponent<List<Double>, List<Double>> {

    private final List<Layer> layers;

    /**
     * Constructs a NeuralNetwork with specified layers.
     *
     * @param neuronsPerLayer An array where each element specifies the number of
     *                        neurons in that layer
     * @param inputSize       The number of inputs for the first layer
     */
    public NeuralNetwork(int[] neuronsPerLayer, int inputSize) {
        layers = IntStream.range(0, neuronsPerLayer.length)
                .mapToObj(i -> createLayer(neuronsPerLayer, inputSize, i))
                .toList();
    }

    private Layer createLayer(int[] neuronsPerLayer, int inputSize, int i) {
        int neurons = neuronsPerLayer[i];
        return new Layer(neurons, i == 0 ? inputSize : neuronsPerLayer[i - 1]);
    }

    @Override
    public List<Double> feedForward(List<Double> inputs) {
        List<Double> outputs = inputs;

        for (Layer layer : layers) {
            outputs = layer.feedForward(outputs);
        }

        return softmax(outputs);
    }

    private List<Double> softmax(List<Double> values) {
        return values.stream().mapToDouble(value -> Math.exp(value) / sumAll(values)).boxed().toList();
    }

    private double sumAll(List<Double> values) {
        return values.stream().mapToDouble(Math::exp).sum();
    }

    /**
     * Backpropagates the error through the network.
     * This method updates the weights and biases of each layer based on the error
     * signals and returns the error signals for the previous layer.
     * 
     * @param inputs       The inputs to the network during the forward pass
     * @param predicted    The predicted outputs from the network
     * @param actualLabel  The actual label for the input data, used for calculating
     *                     error
     * @param learningRate The learning rate for updating the weights and biases
     */
    public void backpropagate(List<Double> inputs, List<Double> predicted, int actualLabel, double learningRate) {
        // Calculate error signals for the output layer
        List<Double> errorSignals = new ArrayList<>(predicted.size());
        for (int i = 0; i < predicted.size(); i++) {
            errorSignals.add(i, predicted.get(i) - (i == actualLabel ? 1 : 0)); // Derivative of softmax + cross-entropy
        }

        List<List<Double>> layerOutputs = new ArrayList<>();
        layerOutputs.add(inputs);

        List<Double> currentOutput = inputs;
        for (Layer layer : layers) {
            currentOutput = layer.feedForward(currentOutput);
            layerOutputs.add(currentOutput);
        }

        // Backpropagate through each layer
        for (int i = layers.size() - 1; i >= 0; i--) {
            errorSignals = layers.get(i).backpropagate(layerOutputs.get(i), errorSignals, learningRate);
        }
    }
}
