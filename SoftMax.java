/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural.ActivationFunctions;

import neural.Network;
import neural.Node;
import neural.Recycled.DataPoint;

/**
 *
 * @author ARZavier
 */
public class SoftMax extends ActivationFunction {
    
    @Override
    public double value(double in, Node n, DataPoint d){
        double numerator = Math.exp(dot(n, d));
        double denominator = 0;
        
        Node[][] net = n.network.net;
        for (Node x : net[net.length - 1]){denominator += Math.exp(dot(x, d));}
        return numerator / denominator;
    }
    
    @Override
    public double derivative(double in, Node n, DataPoint d){
        return in * (1 - in); // Should not be utilized but just in case utilize Logistic derivative
    }
}
