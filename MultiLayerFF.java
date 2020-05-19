/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural;

import neural.ActivationFunctions.ActivationFunction;
import neural.ActivationFunctions.LinearFunction;
import neural.ActivationFunctions.SoftMax;
import neural.Recycled.DataPoint;

/**
 *
 * @author ARZavier
 */
public class MultiLayerFF extends Network {
    
    public MultiLayerFF(int inputs, int[] hiddenLayers, int outputs, ActivationFunction function, double learningRate) {
        super(inputs, hiddenLayers, outputs, function, (outputs == 1) ? new LinearFunction() : new SoftMax(), learningRate);
        /**
         * (outputs == 1) ? new LinearFunction() : new SoftMax() is utilized
         * if regression utilizes linear activation
         */
        
        // Adds a bias node to every layer except the output layer
        for(int i = 0; i < net.length - 1; i++){addBiasNode(i);}
    }
    
    @Override
    public void train(DataPoint[] data){
        for (DataPoint DP : data){
            Node[] inputNodes = net[0];
            for (Node[] layer : net){
                for (Node node : layer){node.reset();}
            }
            for (Node node : inputNodes){node.backprop(MultiLayerFF.learningRate, DP);}
        }
    }
}
