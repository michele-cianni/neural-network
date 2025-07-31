package org.example.api;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class LayerTest {

    @Test
    public void testLayerOutputSize() {
        // Define inputs
        List<Double> inputs = List.of(1.0, 2.0, 3.0);

        // Create a layer with 3 neurons, each taking 3 inputs
        Layer layer = new Layer(3, inputs.size());

        // Calculate outputs
        List<Double> outputs = layer.feedForward(inputs);

        // Assert that the layer produces 3 outputs
        assertEquals(3, outputs.size());
    }

    @Test
    public void testLayerDifferentNeurons() {
        // Define inputs
        List<Double> inputs = List.of(1.0, 2.0, 3.0);

        // Create a layer with 2 neurons, each taking 3 inputs
        Layer layer = new Layer(2, inputs.size());

        // Calculate outputs
        List<Double> outputs = layer.feedForward(inputs);

        // Assert that each neuron produces a unique output (not identical)
        assertNotEquals(outputs.getFirst(), outputs.getLast());
    }
}