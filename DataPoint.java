/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural.Recycled;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 *
 * @author ARZavier
 */
public class DataPoint {
    // Form of { label: payload }
    protected Feature[] features;
    protected double[] target;
    protected String classMembership;
    
    // Default Constructor
    public DataPoint(Feature[] features){this.features = features;}
    
    // Constructor With Class Membership
    public DataPoint(Feature[] features, double[] target) {
        this.features = features;
        this.target = target;
    }
    
    public DataPoint(Feature[] features, String classMembership) {
        this(features);
        this.classMembership = classMembership;
    }
    
    public String obtainClassMembership(){return this.classMembership;}
    
    public void setClassMembership(String newClassMembership) {this.classMembership = newClassMembership;}
    
    /**
     * @return an array of target values
     */
    
    public double[] obtainTarget(){return this.target;}
    
    /**
     * @return the number of features that this point contains
     */
    
    public int obtainNumberOfFeatures(){return this.features.length;}
    
    /**
     * Obtains the feature value based on the label
     * @param label String identifier of the feature
     * @return the feature value
     */
    
    public Feature obtainFeatureByLabel(String label) {
        for (Feature feat: this.features){
            if (feat.obtainLabel().equals(label))
                return feat;
        }
        // If there is no feature with that label
        return null;
    }
    
    /**
     * Obtains the ith feature
     * @param i index of the feature
     * @return the feature
     */
    
    public Feature obtainFeatureAt(int i) {
        if (i >= features.length)
            System.out.println(this.toString());
        return features[i];
    }
    
    /**
     * @return all of the features of this point
     */
    
    public Feature[] obtainFeatures(){return features;}
    
    @Override
    public String toString(){
        List<String> featureStrings = new ArrayList<>();
        for (Feature feat: features) {featureStrings.add(feat.toString());}
        return "Features: {" + String.join(",", featureStrings) + "} | Target: " + Arrays.toString(target);
    }
    
    @Override
    public boolean equals(Object obj) {
        DataPoint other = (DataPoint) obj;
        for (int i = 0; i < features.length; i++) {
            if (!features[i].equals(other.obtainFeatureAt(i)))
                return false;
        }
        return true;
    }
}
