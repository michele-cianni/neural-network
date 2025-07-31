package org.example.api;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Represents a layer in a neural network, containing multiple neurons.
 * Each layer can perform feedforward operations and backpropagation.
 * 
 * @author Michele Cianni
 * @version 1.0
 * @see NeuralComponent
 * @see Neuron
 */
public final class Layer implements NeuralComponent<List<Double>, List<Double>> {

    private final List<Neuron> neurons;

    /**
     * Constructs a Layer with a specified number of neurons, each initialized with
     * weights based on the number of inputs.
     *
     * @param numberOfNeurons The number of neurons in this layer
     * @param numberOfInputs  The number of inputs each neuron will receive
     */
    public Layer(int numberOfNeurons, int numberOfInputs) {
        neurons = IntStream.range(0, numberOfNeurons).mapToObj(i -> createNeuron(numberOfInputs)).toList();
    }

    private Neuron createNeuron(int numberOfInputs) {
        double range = Math.sqrt(6.0 / numberOfInputs); // Xavier initialization
        List<Double> weights = IntStream.range(0, numberOfInputs)
                .mapToDouble(j -> (Math.random() * 2 - 1) * range)
                .boxed().collect(Collectors.toList());
        double bias = 0.0;
        return new Neuron(weights, bias);
    }

    @Override
    public List<Double> feedForward(List<Double> input) {
        return neurons.stream().mapToDouble(neuron -> neuron.feedForward(input)).boxed().toList();
    }

    /**
     * Backpropagates the error through the layer.
     * This method updates the weights and biases of each neuron based on the error
     * signals and returns the error signals for the previous layer.
     * 
     * @param inputs       The inputs to the layer during the forward pass
     * @param errorSignals The error signals from the next layer or the output layer
     * @param learningRate The learning rate for updating the weights and biases
     * @return A list of error signals for the previous layer
     */
    public List<Double> backpropagate(List<Double> inputs, List<Double> errorSignals, double learningRate) {
        
        List<Double> nextErrorSignals = new ArrayList<>(inputs.size()); 

        // Initialize with zeros
        for (int i = 0; i < inputs.size(); i++) {
            nextErrorSignals.add(0.0);
        }

        // Backpropagate through each neuron in the layer
        for (int i = 0; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);
            double errorSignal = errorSignals.get(i);
            neuron.backpropagate(inputs, errorSignal, learningRate);

            // Accumulate the error signals for the previous layer
            for (int j = 0; j < inputs.size(); j++) {
                double currentError = nextErrorSignals.get(j);
                nextErrorSignals.set(j, currentError + neuron.getWeights().get(j) * errorSignal);
            }
        }

        return nextErrorSignals;
    }

}
