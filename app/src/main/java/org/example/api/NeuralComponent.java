package org.example.api;

/**
 * NeuralComponent is a functional interface that defines a contract for components
 * in a neural network that can process inputs and produce outputs.
 * It is used to create a uniform interface for different components like Neuron, Layer,
 * and NeuralNetwork.
 * 
 * @param <I> The type of input to the neural component.
 * @param <O> The type of output produced by the neural component.
 * 
 * @author Michele Cianni
 * @version 1.0
 */
@FunctionalInterface
public interface NeuralComponent<I, O> {
    
    /**
     * Processes the given input and produces an output.
     * 
     * @param input The input to the neural component.
     * @return The output produced by the neural component.
     */
    O feedForward(I input);
    
}
