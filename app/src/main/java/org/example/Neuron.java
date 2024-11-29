package org.example;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class Neuron implements NeuralComponent<List<Double>, Double> {
    private final List<Double> weights;
    private final List<Double> weightsGradients;
    private double bias;
    private double biasGradient;

    public Neuron(List<Double> weights, double bias) {
        this.weights = weights;
        this.bias = bias;
        this.weightsGradients = new ArrayList<>(weights.size());
        this.biasGradient = 0;
    }

    @Override
    public Double feedForward(List<Double> input) {
        return activationReLU(bias + sumWeights(input));
    }

    private double sumWeights(List<Double> input) {
        return IntStream.range(0, weights.size()).mapToDouble(i -> weights.get(i) * input.get(i)).sum();
    }

    private double activationReLU(double sum) {
        return Math.max(0, sum);
    }

    // Backpropagate gradients for the neuron
    public void backpropagate(List<Double> inputs, double errorSignal, double learningRate) {
        // Calculate gradients for weights and bias
        for (int i = 0; i < weights.size(); i++) {
            weightsGradients.add(i,  errorSignal * inputs.get(i));
        }
        biasGradient = errorSignal;

        // Update weights and bias using gradient descent
        for (int i = 0; i < weights.size(); i++) {
            weights.set(i, weights.get(i) - learningRate * weightsGradients.get(i));
        }
        bias -= learningRate * biasGradient;
    }

    public List<Double> getWeights() {
        return weights;
    }
}
