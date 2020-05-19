/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural;

import neural.ActivationFunctions.ActivationFunction;
import neural.Recycled.DataPoint;
import neural.Recycled.Utilities;

import java.util.ArrayList;
import java.util.Arrays;

/**
 *
 * @author ARZavier
 */
public class Tuning {
    public static String retMe = "";
    
    /**
     * Tunes a Feed Forward Network
     * This infers the number of inputs and outputs from the dataset
     * @param nodeSearchParams the first parameter is the maximum number of
     * layers, second is the lower bound of nodes, the third is the upper bound
     * of nodes, and the fourth is the increment between
     * @param learningRates the learning rates to search
     * @param trainData data to train the model
     * @param testData data to test the model
     * @param function the activation function utilized for all the nodes
     * @return the tuned model
     */
    
    public static MultiLayerFF tuneFFNetwork (int[] nodeSearchParams, double[] learningRates,
                                              DataPoint[] trainData, DataPoint[] testData, ActivationFunction function){
        
        int inputs = trainData[0].obtainNumberOfFeatures();
        int outputs = trainData[0].obtainTarget().length;
        
        int[][][] options = buildLayers(nodeSearchParams[0], nodeSearchParams[1], nodeSearchParams[2], nodeSearchParams[3]);
        double best = Double.NEGATIVE_INFINITY;
        MultiLayerFF tunedModel = null;
        
        for (int layer = 0; layer < nodeSearchParams[0]; layer++) {
            for (double rate: learningRates) {
                for (int i = 0; i < options[layer].length; i++) {
                    MultiLayerFF model = new MultiLayerFF (inputs, options[layer][i], outputs, function, rate);
                    model.train(trainData, .01);
                    double value = 0;
                    for (DataPoint d : testData){
                        if (d.obtainTarget().length == 1){
                            // Regression (Utilize Squared Error)
                            value -= Math.pow((d.obtainTarget()[0] - model.predict(d)[0]), 2);
                            // Subtract because we dsire to maximize negative error
                        } else {
                            // Classification (Utilize Accuracy)
                            if (Utilities.argMax(model.predict(d)) == Utilities.argMax(d.obtainTarget()))
                                value += 1;
                        }
                    }
                    
                    if (value > best) {
                        // Set the new model to this one because it is better
                        System.out.printf("Layers: %d, Learning Rate: %f, Layer Plan: %s \n", layer, rate, 
                                Arrays.toString(options[layer][i]));
                        System.out.println("Accuracy: " + value/testData.length);
                        retMe = "" + layer + "," + Arrays.toString(options[layer][i]) + ",";
                        tunedModel = model;
                        best = value;
                    }
                }
            }
        }
        return tunedModel;
    }
    
    /**
     * Utility function that builds an array of all 
     * possible hidden nodes and layer configurations.
     * @param layers
     * @param lower
     * @param upper
     * @param inc
     * @return
     */
    
    private static int[][][] buildLayers(int layers, int lower, int upper, int inc){
        int[] base = new int[(upper - lower)/inc];
        for (int i = 0; i < base.length; i++){base[i] = lower + i * inc;}
        int[][][] allOptions = new int[layers + 1][][];
        // First index is layer number, second is a counter, and the last array stores the configuration
        allOptions[0] = new int[][] {{}}; // No Hidden Layer
        allOptions[1] = new int[base.length][1]; // One Hidden Layer
        for(int i = 0; i < base.length; i++){allOptions[1][i][0] = base[i];}
        for (int i = 2; i < layers; i++){
            ArrayList<int[]> layer = new ArrayList<>();
            int[][] prev = allOptions[i-1];
            for (int j = 0; j < prev.length; j++){
                for (int k = 0; k < base.length; k++){layer.add(append(base[k], prev[j]));}
            }
            allOptions[i] = flatten(layer);
        }
        return allOptions;
    }
    
    private static int[][] flatten(ArrayList<int[]> l) {
        int[][] retMe = new int[l.size()][];
        for (int i = 0; i < retMe.length; i++){retMe[i] = l.get(i);}
        return retMe;
    }
    
    private static int[] append(int value, int[] arr) {
        int[] retMe = new int[arr.length + 1];
        for (int i = 0; i < arr.length; i++){retMe[i] = arr[i];}
        retMe[retMe.length - 1] = value;
        return retMe;
    }
}
