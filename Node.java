/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural;

import neural.ActivationFunctions.ActivationFunction;
import neural.Recycled.DataPoint;

import java.util.ArrayList;
import java.util.Arrays;

/**
 *
 * @author ARZavier
 */
public class Node {
    private Node[] downstreamNodes;
    public Node[] upstreamNodes;
    public Double[] weights;
    private ActivationFunction function;
    public Network network;
    
    private boolean ranBackProp = false;
    private double prevChange[];
    private double alpha = 0.5;
    private double delta[];
    double sum = 0;
    
    public Node (ActivationFunction func, Network network) {
        downstreamNodes = new Node[0];
        weights = new Double[0];
        upstreamNodes = new Node[0];
        function = func;
        this.network = network;
    }
    
    /**
     * Adds connection from this node to downstream node
     * @param n
     * @param w
     */
    
    public void addConnection(Node n, double w) {
        ArrayList<Node> tempDownstream = new ArrayList<>();
        ArrayList<Double> tempWeights = new ArrayList<>();
        
        tempDownstream.addAll(Arrays.asList(downstreamNodes));
        tempWeights.addAll(Arrays.asList(weights));
        // Adds the new node and weight
        tempDownstream.add(n);
        tempWeights.add(w);
        // Tell new node that this now is upstream
        n.addUpstreamNode(this);
        downstreamNodes = tempDownstream.toArray(new Node[tempDownstream.size()]);
        weights = tempWeights.toArray(new Double[tempWeights.size()]);
    }
    
    /**
     * Connects a downstream node to its upstream node
     * @param n
     */
    
    private void addUpstreamNode(Node n) {
        ArrayList<Node> tempUpstream = new ArrayList<>();
        tempUpstream.addAll(Arrays.asList(upstreamNodes));
        tempUpstream.add(n);
        this.upstreamNodes = tempUpstream.toArray(new Node[tempUpstream.size()]);
    }
    
    /**
     * Adds connections from this node to all other nodes
     * @param nodes nodes to be connected to
     * @param weights weights of each connection
     */
    public void addConnections(Node[] nodes, double[] weights) {
        for (int i = 0; i < nodes.length; i++){addConnection(nodes[i], weights[i]);}
    }
    
    /**
     * Run backpropagation to train weights (this is a recursive function
     * to train all the weights in the network).
     * @param learningRate learning rate
     * @param d data point being trained on
     * @return the change in weight of this node
     */
    
    public double backprop (double learningRate, DataPoint d) {
        if (this.isOutputNode()){
            // Base Case (This is an Output Node)
            double nodeOutput = this.genOutput(d);
            double target;
            if(d.obtainTarget().length == 1)
                target = d.obtainTarget()[0];
            else {
                // Find the output of the classification node
                int index = network.getOutputIndex(this);
                target = d.obtainTarget()[index];
            }
            double changeToWeights = (target-nodeOutput) * function.derivative(nodeOutput, this, d);
            ranBackProp = true;
            return changeToWeights;
        } else {
            // Recursive Case (This is a Hidden Node)
            if(!ranBackProp){
                // We already called backprop from this node
                for (int i = 0; i < downstreamNodes.length; i++) {
                    double backpropResult = downstreamNodes[i].backprop(learningRate, d);
                    sum += backpropResult * weights[i];
                    // Change Weights
                    double change = (learningRate * backpropResult * this.genOutput(d) + alpha * prevChange[i]);
                    weights[i] = weights[i] + change;
                    delta[i] = change - prevChange[i];
                    prevChange[i] = change;
                }
                ranBackProp = true; // For memorization so we do not call this many times
            }
            // oj(1-oj)(SUM())
            return function.derivative(this.genOutput(d), this, d) * sum;
        }
    }
    
    private Double remOutput = null; // For Memorization
    
    /**
     * First implementation for Base Node (i.e., takes values 
     * directly and not as a combination of previous nodes).
     * @param d
     * @return the linear combination put through a sigmoid function -> sig(W.x)
     */
    
    public double genOutput(DataPoint d){
        /**
         * Memorizes previous value so that the new value does not
         * have to be computed recursively through the network again
         */
        if (remOutput != null)
            return remOutput;
        if (this.isInputNode()) {
            // Return the value from the data point
            int index = network.getInputIndex(this);
            return d.obtainFeatureAt(index).obtainContinuousPayload();
        }
        double sum = 0;
        for (int i = 0; i < upstreamNodes.length; i++) {
            // Obtains the weight corresponding to this node from the upstream node
            double upOut = upstreamNodes[i].genOutput(d);
            sum += upstreamNodes[i].weights[upstreamNodes[i].getIndex(this)] * upstreamNodes[i].genOutput(d);
        }
        remOutput = function.value(sum, this, d);
        return remOutput;
    }
    
    /**
     * Causes the node to forget memorized values
     */
    public void reset(){
        remOutput = null;
        ranBackProp = false;
        prevChange = new double[weights.length];
        delta = new double[weights.length];
        
        for (int i = 0; i < weights.length; i++) {
            prevChange[i] = 0;
            delta[i] = 0;
        }
    }
    
    public boolean isInputNode(){return this.upstreamNodes.length == 0;}
    public boolean isOutputNode(){return this.downstreamNodes.length == 0;}
    
    public int getIndex(Node n){
        for (int i = 0; i < downstreamNodes.length; i++){
            if (downstreamNodes[i].equals(n))
                return i;
        }
        return -1;
    }
    
    public void setFunction(ActivationFunction function){this.function = function;}
    public void setWeights(Double[] weights){this.weights = weights;}
    public double[] getDelta(){return delta;}
}
