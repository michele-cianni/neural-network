package org.example.api;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Neuron represents a single neuron in a neural network.
 * This class encapsulates the weights, bias, and the methods for
 * forward propagation and backpropagation of errors.
 * 
 * @author Michele Cianni
 * @version 1.0
 * @see NeuralComponent
 */
public final class Neuron implements NeuralComponent<List<Double>, Double> {
    
    private final List<Double> weights;
    private final List<Double> weightsGradients;
    private double bias;
    private double biasGradient;

    /**
     * Constructs a Neuron with the specified weights and bias.
     * The weights are initialized using Xavier initialization.
     * 
     * @param weights The initial weights of the neuron.
     * @param bias The initial bias of the neuron.
     */
    public Neuron(List<Double> weights, double bias) {
        this.weights = weights;
        this.weightsGradients = new ArrayList<>(weights.size());
        this.bias = bias;
        this.biasGradient = 0;
    }

    @Override
    public Double feedForward(List<Double> input) {
        return activationReLU(bias + sumWeights(input));
    }

    private double activationReLU(double sum) {
        return Math.max(0, sum);
    }

    /**
     * Backpropagate the error through this neuron.
     * This method calculates the gradients for the weights and bias
     * and updates them using gradient descent.
     * 
     * @param inputs The inputs to the neuron during the forward pass.
     * @param errorSignal The error signal from the next layer or the output layer.
     * @param learningRate The learning rate for updating the weights and bias.
     */
    public void backpropagate(List<Double> inputs, double errorSignal, double learningRate) {

        double weightedSum = bias + sumWeights(inputs);
        double activatedErrorSignal = errorSignal * activationDerivativeReLU(weightedSum);

        // Clear previous gradients
        weightsGradients.clear();
        
        // Calculate gradients for weights and bias
        for (int i = 0; i < weights.size(); i++) {
            weightsGradients.add(i, activatedErrorSignal * inputs.get(i));
        }
        biasGradient = activatedErrorSignal;

        // Update weights and bias using gradient descent
        for (int i = 0; i < weights.size(); i++) {
            weights.set(i, weights.get(i) - learningRate * weightsGradients.get(i));
        }
        bias -= learningRate * biasGradient;
    }

    private double sumWeights(List<Double> input) {
        return IntStream.range(0, weights.size()).mapToDouble(i -> weights.get(i) * input.get(i)).sum();
    }

    private double activationDerivativeReLU(double value) {
        return value > 0 ? 1 : 0;
    }

    /**
     * Gets the weights of this neuron.
     * 
     * @return The list of weights.
     */
    public List<Double> getWeights() {
        return weights;
    }
}
