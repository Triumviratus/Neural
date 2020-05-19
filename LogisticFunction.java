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
public class LogisticFunction extends ActivationFunction {
    
    @Override
    public double value (double in, Node n, DataPoint d){return (1 / (1 + Math.exp(-in)));}
    @Override
    public double derivative (double in, Node n, DataPoint dataPoint){return in * (1 - in);}
}
