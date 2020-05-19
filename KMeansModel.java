/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural.Recycled;

import java.util.Random;

/**
 *
 * @author ARZavier
 */
public class KMeansModel extends ClusteringModel {
    
    /**
     * Constructor
     * @param data data to be clustered
     */
    public KMeansModel(DataPoint[] data){super(data);}
    
    @Override
    public void cluster(int k){
        /**
         * Initialize random centroids and be more
         * strategic about where we initialize points.
         */
        centroids = new Centroid[k];
        DataPoint[] randArray = Utilities.scramble(data);
        
        for (int i = 0; i < k; i++) {
            centroids[i] = new Centroid(randArray[i]); // Puts the centroid at random point
            for (int j = 0; j < centroids[i].obtainFeatures().length; j++) {
                centroids[i].obtainFeatureAt(j).setContinuousPayload(
                        centroids[i].obtainFeatureAt(j).obtainContinuousPayload() * (2 * Math.random() - 1));
            }
        }
        /**
         * Find when to stop in a better way
         * Run 100 iterations to train
         */
        for (int i = 0; i < 100; i++){this.runIteration();}
    }
    
    /**
     * Runs an iteration of the algorithm, which assigns points
     * to centroids and moves the centroids to their averages.
     */
    
    private void runIteration() {
        // Step 1: Evaluate Group Membership
        this.clearCentroids(); // Empties out old data points for new iteration
        
        for (DataPoint point : data) {
            // Find distances from all current centroids to this point
            Double[] distances = Utilities.getDistancesToAllPoints(point, this.centroids);
            // Find the centroid this point is closest to
            Centroid closestCentroid = this.centroids[Utilities.findIndexOfMinimum(distances)];
            // Set the group of this point to be the group of the centroid
            closestCentroid.addPoint(point);
        }
        // Step 2: Compute New Centroids
        this.moveCentroids();
        this.assignClasses(); // So the classes on centroid remain up-to-date
    }
}
