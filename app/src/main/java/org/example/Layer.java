package org.example;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class Layer implements NeuralComponent<List<Double>, List<Double>> {
    protected final List<Neuron> neurons;

    public Layer(int numberOfNeurons, int numberOfInputs) {
        neurons = IntStream.range(0, numberOfNeurons).mapToObj(i -> createNeuron(numberOfInputs)).toList();
    }

    private Neuron createNeuron(int numberOfInputs) {
        List<Double> weights = IntStream.range(0, numberOfInputs)
                .mapToDouble(j -> Math.random())
                .boxed().toList();
        double bias = Math.random();
        return new Neuron(weights, bias);
    }


    @Override
    public List<Double> feedForward(List<Double> input) {
        return neurons.stream()
                .mapToDouble(neuron -> neuron.feedForward(input))
                .boxed().toList();
    }

    // Backpropagate the error through this layer
    public List<Double> backpropagate(List<Double> inputs, List<Double> errorSignals, double learningRate) {
        List<Double> nextErrorSignals = new ArrayList<>(inputs.size()); // Error signal for the previous layer

        for (int i = 0; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);
            double errorSignal = errorSignals.get(i);
            neuron.backpropagate(inputs, errorSignal, learningRate);

            // Calculate error signals for the previous layer
            for (int j = 0; j < inputs.size(); j++) {
                nextErrorSignals.add(j, neuron.getWeights().get(j) * errorSignal);
            }
        }

        return nextErrorSignals;
    }

    public List<Double> getPrevLayerErrors(Layer prevLayer) {
        // Implement logic to propagate errors backward to previous layer
        return new ArrayList<>(prevLayer.neurons.size());
    }

}
