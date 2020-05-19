/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural.ActivationFunctions;

import neural.Node;
import neural.Recycled.DataPoint;

/**
 *
 * @author ARZavier
 */
public abstract class ActivationFunction {
    
    public ActivationFunction(){}
    
    /**
     * Calculate the activation of a neuron
     * @param in
     * @param n
     * @param dataPoint
     * @return
     */
    
    public abstract double value(double in, Node n, DataPoint dataPoint);
    
    /**
     * Calculates the derivative of the neuron activation
     * @param in
     * @param n
     * @param dataPoint
     * @return
     */
    public abstract double derivative (double in, Node n, DataPoint dataPoint);
    
    public double dot (Node n, DataPoint d){
        double sum = 0;
        for (int i = 0; i < n.upstreamNodes.length; i++){
            /**
             * Obtains the weight corresponding to
             * this node from the upstream node.
             */
            sum += n.upstreamNodes[i].weights[n.upstreamNodes[i].getIndex(n)] * n.upstreamNodes[i].genOutput(d);
        }
        return sum;
    }
}
