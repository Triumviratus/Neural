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
public class LinearFunction extends ActivationFunction {
    
    @Override
    public double value(double in, Node n, DataPoint dataPoint){return in;}
    @Override
    public double derivative (double in, Node n, DataPoint dataPoint){return in;}
}
