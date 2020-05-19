/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural.Recycled;

import neural.Network;
import java.util.ArrayList;
import java.util.Arrays;

/**
 *
 * @author ARZavier
 */
public class Validator {
    
    /**
     * Tests the data on accuracy
     * @param model the model to be tested
     * @param testData the data to be tested
     * @return the accuracy of the model
     */
    
    public static double accuracy(Network model, DataPoint[] testData) {
        int correctClassification = 0; // Counters
        for (int i = 0; i < testData.length; i++) {
            if (Utilities.argMax(model.predict(testData[i])) == Utilities.argMax(testData[i].obtainTarget()))
                correctClassification++;
        }
        return ((double) correctClassification)/((double) testData.length);
    }
    
    public static double squaredError(Network model, DataPoint[] testData) {
        double error = 0;
        for (int i = 0; i < testData.length; i++) {
            error += Math.pow((model.predict(testData[i])[0] - testData[i].obtainTarget()[0]), 2);
        }
        return error/testData.length;
    }
}
