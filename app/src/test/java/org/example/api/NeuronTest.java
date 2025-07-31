package org.example.api;

import org.example.api.Neuron;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class NeuronTest {

    @Test
    void shouldFeedPositive() {
        // Define inputs, weights, and bias
        List<Double> inputs = List.of(1.0, 2.0, 3.0);
        List<Double> weights = List.of(0.2, 0.8, -0.5);
        double bias = 2.0;

        // Create a Neuron
        Neuron neuron = new Neuron(weights, bias);

        // Calculate output
        double output = neuron.feedForward(inputs);

        // Assert that the output is correct
        assertEquals(2.3, output, 0.001);
    }

    @Test
    public void testNegativeOutputReLU() {
        // Define inputs, weights, and bias
        List<Double> inputs = List.of(1.0, 2.0, 3.0);
        List<Double> weights = List.of(-0.2, -0.8, -0.5);
        double bias = -1.0;

        // Create a Neuron
        Neuron neuron = new Neuron(weights, bias);

        // Calculate output
        double output = neuron.feedForward(inputs);

        // Assert that the output is zero (because of ReLU)
        assertEquals(0.0, output, 0.001);
    }

    @Test
    public void testZeroBias() {
        // Define inputs, weights, and bias
        List<Double> inputs = List.of(1.0, 1.0);
        List<Double> weights = List.of(0.5, 0.5);
        double bias = 0.0;

        // Create a Neuron
        Neuron neuron = new Neuron(weights, bias);

        // Calculate output
        double output = neuron.feedForward(inputs);

        // Assert that the output is correct
        assertEquals(1.0, output, 0.001);
    }

}