/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural.ActivationFunctions;

import neural.Node;
import neural.Recycled.DataPoint;
import neural.Recycled.Utilities;

/**
 *
 * @author ARZavier
 */
public class GaussianFunction extends ActivationFunction {
    
    DataPoint center;
    double sigma;
    
    public GaussianFunction(){} // Just a placeholdr (it should always be replaced later)
    public GaussianFunction(DataPoint center, double sigma) {
        this.center = center;
        this.sigma = sigma;
    }
    
    @Override
    public double value (double in, Node n, DataPoint d) {
        if (center == null)
            System.out.println("Null Center");
        return Math.exp(-(Math.pow(Utilities.obtainDistance(center, d), 2) / (2 * Math.pow(sigma, 2))));
    }
    
    @Override
    public double derivative (double in, Node n, DataPoint dataPoint) {return value (in, n, dataPoint);}
}
