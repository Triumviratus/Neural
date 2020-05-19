/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural;

import neural.ActivationFunctions.GaussianFunction;
import neural.ActivationFunctions.LinearFunction;
import neural.ActivationFunctions.LogisticFunction;
import neural.ActivationFunctions.SoftMax;
import neural.Recycled.Centroid;
import neural.Recycled.ClusteringModel;
import neural.Recycled.DataPoint;

/**
 *
 * @author ARZavier
 */
public class RBF extends Network {
    
    public RBF (ClusteringModel model, double learningRate){
        this(model.obtainCentroids(), calculateSigmas(model), model.obtainData()[0].obtainNumberOfFeatures(),
                model.obtainData()[0].obtainTarget().length, learningRate);
    }
    
    public RBF(DataPoint[] centers, double[] sigmas, int inputs, int outputs, double learningRate) {
        super(inputs, new int[]{sigmas.length}, outputs, new GaussianFunction(),
                (outputs == 1) ? new LinearFunction() : new SoftMax(), learningRate);
        
        // Sets it so that there are no weights on the input layer
        Node[] inputLayer = net[0];
        for (Node n : inputLayer){for (int i = 0; i < n.weights.length; i++){n.weights[i] = 1.0;}}
        
        // Sets the Gaussians in the hidden layer
        Node[] hiddenLayer = net[1];
        for (int i = 0; i < hiddenLayer.length; i++){
            hiddenLayer[i].setFunction(new GaussianFunction(centers[i], sigmas[i]));
        }
        addBiasNode(1); // Adds a bias node to the hidden layer
    }
    
    private static double[] calculateSigmas(ClusteringModel model) {
        Centroid[] centroids = model.obtainCentroids();
        double[] sigmas = new double[centroids.length];
        for (int i = 0; i < centroids.length; i++) {
            sigmas[i] = centroids[i].deriveDeviation();
            if(sigmas[i] == 0)
                System.out.println("PROBLEM");
        }
        return sigmas;
    }
    
    @Override
    public void train(DataPoint[] data) {
        for (DataPoint DP : data){
            Node[] inputNodes = net[1];
            /**
             * Start backpropagation at the hidden layer
             * so not to train input nodes.
             */
            for (Node[] layer : net){for (Node node : layer){node.reset();}}
            for (Node node : inputNodes){node.backprop(RBF.learningRate, DP);}
        }
    }
}
