/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural;

import neural.ActivationFunctions.LinearFunction;
import neural.Recycled.DataPoint;

/**
 *
 * @author ARZavier
 */
public class BiasNode extends Node {
    public BiasNode(Network network){super(new LinearFunction(), network);}
    @Override
    public double genOutput(DataPoint d){return 1;}
}
