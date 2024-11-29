package org.example;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class NeuralNetwork implements NeuralComponent<List<Double>, List<Double>> {
    private final List<Layer> layers;

    public NeuralNetwork(int[] neuronsPerLayer, int inputSize) {
       layers = IntStream.range(0, neuronsPerLayer.length)
        .mapToObj(i -> createLayer(neuronsPerLayer, inputSize, i))
        .toList();
    }

    private Layer createLayer(int[] neuronsPerLayer, int inputSize, int i) {
        int neurons = neuronsPerLayer[i];
        return new Layer(neurons, i == 0 ? inputSize : neuronsPerLayer[i - 1]);
    }

    public List<Double> feedForward(List<Double> inputs) {
        List<Double> outputs = inputs;

        for (Layer layer : layers) {
            outputs = layer.feedForward(outputs);
        }

        return softmax(outputs);
    }

    private List<Double> softmax(List<Double> values) {
        return values.stream().mapToDouble(value ->  Math.exp(value) / sumAll(values)).boxed().toList();
    }

    private double sumAll(List<Double> values) {
        return values.stream().mapToDouble(Math::exp).sum();
    }

    public void backpropagate(List<Double> inputs, List<Double> predicted, int actualLabel) {
        // Calculate error signals for the output layer
        List<Double> errorSignals = new ArrayList<>(predicted.size());
        for (int i = 0; i < predicted.size(); i++) {
            errorSignals.add(i,predicted.get(i) - (i == actualLabel ? 1 : 0)); // Derivative of softmax + cross-entropy
        }

        // Backpropagate through each layer
        for (int i = layers.size() - 1; i >= 0; i--) {
            errorSignals = layers.get(i)
                    .backpropagate(i == 0 ? inputs : layers.get(i - 1)
                    .feedForward(inputs), errorSignals, actualLabel);
        }
    }
}
