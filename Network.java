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
import java.util.Arrays;

/**
 *
 * @author ARZavier
 */
public abstract class Network {
    
    static double learningRate;
    public Node[][] net;
    
    public Network (int inputs, int[] hiddenLayers, int outputs, 
            ActivationFunction function, ActivationFunction outputFunction, double learningRate) {
        Node[][] net = new Node[hiddenLayers.length + 2][];
        net[0] = addLayer(inputs, function); // Input Layer
        for (int i = 0; i < hiddenLayers.length; i++){net[i+1] = addLayer(hiddenLayers[i], function);}
        net[net.length - 1] = addLayer(outputs, outputFunction); // Output Layer
        
        for (int i = 0; i < net.length - 1; i++) {
            for (Node n : net[i]) {
                double[] weight = new double[net[i+1].length];
                Arrays.fill(weight, 0.1);
                n.addConnections(net[i+1], weight);
            }
        }
        this.net = net;
        Network.learningRate = learningRate;
        
    }
    
    public Network (int inputs, int[] hiddenLayers, int outputs, ActivationFunction function, double learningRate){
        this(inputs, hiddenLayers, outputs, function, (outputs == 1) ? new LinearFunction() : new SoftMax(), learningRate);
        Network.learningRate = learningRate;
    }
    
    public Network(int[] layers, ActivationFunction activationFunction, double LearningRate){
        this(layers[0], Arrays.copyOfRange(layers, 1, layers.length-1), layers[layers.length-1], activationFunction, learningRate);
    }
    
    /**
     * Trains the model on the data, until every weight changes by less than the threshold
     * @param data the dataset to be trained on
     * @param threshold the threshold weight value
     */
    
    public void train(DataPoint[] data, double threshold) {
        boolean flag = true;
        int cnt = 0;
        while (flag && cnt < 100){
            flag = false;
            cnt++;
            train(data);
            for(int i = 0; i < net.length; i++){
                for(int j = 0; j < net[i].length; j++){
                    for (double delta : net[i][j].getDelta()){
                        if(Math.abs(delta) > threshold)
                            flag = true;
                    }
                }
            }
        }
    }
    
    public abstract void train(DataPoint[] data);
    
    /**
     * Generates an expect output based on an unknown data point
     * @param dataPoint
     * @return
     */
    
    public double[] predict (DataPoint dataPoint) {
        double[] outputs = new double[net[net.length - 1].length];
        for (int i = 0; i < outputs.length; i++){outputs[i] = net[net.length-1][i].genOutput(dataPoint);}
        return outputs;
    }
    
    /**
     * Finds the index of an output node
     * @param n
     * @return
     */
    
    public int getOutputIndex(Node n){
        for (int i = 0; i < net[net.length-1].length; i++){
            if (net[net.length-1][i].equals(n))
                return i;
        }
        return -1;
    }
    
    /**
     * Find the index of an input node
     * @param n
     * @return 
     */
    
    public int getInputIndex(Node n){
        for (int i = 0; i < net[0].length; i++){
            if (net[0][i].equals(n))
                return i;
        }
        return -1;
    }
    
    /**
     * Utility function for building the network
     * @param length
     * @param function
     * @return
     */
    
    protected Node[] addLayer(int length, ActivationFunction function){
        Node[] layer = new Node[length];
        for(int j = 0; j < layer.length; j++){layer[j] = new Node(function, this);}
        return layer;
    }
    
    protected void addBiasNode(int layer){
        if (layer >= net.length-1)
            return;
        Node[] newLayer = new Node[net[layer].length + 1];
        for(int i = 0; i < net[layer].length; i++){newLayer[i] = net[layer][i];}
        // Creates the bias node and connects it to downstream layers
        BiasNode b = new BiasNode(this);
        double[] weight = new double[net[layer+1].length];
        Arrays.fill(weight, 0.1);
        b.addConnections(net[layer+1], weight);
        newLayer[newLayer.length-1] = b;
    }
}
