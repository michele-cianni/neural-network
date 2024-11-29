package org.example;

@FunctionalInterface
public interface NeuralComponent<I, O> {
    O feedForward(I input);
}
