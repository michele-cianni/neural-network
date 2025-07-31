package org.example;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Layer implements NeuralComponent<List<Double>, List<Double>> {

    private final List<Neuron> neurons;

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

    // Backpropagate the error through this layer
    public List<Double> backpropagate(List<Double> inputs, List<Double> errorSignals, double learningRate) {
        List<Double> nextErrorSignals = new ArrayList<>(inputs.size()); // Error signal for the previous layer

        // Initialize with zeros
        for (int i = 0; i < inputs.size(); i++) {
            nextErrorSignals.add(0.0);
        }

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
