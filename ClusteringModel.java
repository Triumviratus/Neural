/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural.Recycled;

/**
 *
 * @author ARZavier
 */
public abstract class ClusteringModel {
    protected DataPoint[] data;
    protected Centroid[] centroids;
    
    /**
     * Initialize the clustering model with some data to be clustered
     * @param data the dataset
     */
    public ClusteringModel(DataPoint[] data){this.data = data;}
    
    /**
     * Creates k clusters on data
     * @param k the number of clusters
     */
    
    public abstract void cluster(int k);
    
    /**
     * Obtains the distortion value of the clusters
     * We must have called cluster() before obtainDistortion
     * @return the distortion of the clusters
     * @throws Exception if the number of features is not the same between points being compared
     */
    
    public double obtainDistortion() throws Exception {
        double ans = 0.0;
        // Cycle Through All points
        for (DataPoint point : this.data){
            // Obtain the group average for the group this point belongs to
            DataPoint average = obtainCentroid(point);
            // Check to ascertain that we are comparing points of the same dimensionality
            if (average.obtainNumberOfFeatures() != point.obtainNumberOfFeatures())
                throw new Exception("Attempting to Compare 2 Vetors of Different Dimensions. Probable Error in Dataset.");
            ans += Math.pow(Utilities.obtainDistance(point, average), 2);
        }
        return ans;
    }
    
    /**
     * Moves all the centroids to their centroids
     */
    public void moveCentroids(){for (Centroid centroid : centroids){centroid.moveToAvg();}}
    
    /**
     * Returns the centroid to which the point belongs
     * @param p point being queried
     * @return the centroid to which the point belongs
     */
    
    private Centroid obtainCentroid(DataPoint p){
        for (Centroid centroid : centroids){
            if (centroid.contains(p))
                return centroid;
        }
        // Error catching in case the point does not belong to any centroid
        Double[] dists = Utilities.getDistancesToPoints(p, centroids);
        Centroid c = centroids[Utilities.findIndexOfMinimum(dists)];
        c.addPoint(p);
        return c;
    }
    
    /**
     * Obtains the centroids
     * @return the centroids
     */
    public Centroid[] obtainCentroids(){return this.centroids;}
    
    /**
     * Obtains the data being clustered
     * @return the data
     */
    public DataPoint[] obtainData(){return this.data;}
    
    /**
     * Sets the centroids to a new Centroid[]
     * @param newPoints the new centroids
     */
    protected void setCentroids(Centroid[] newPoints){this.centroids = newPoints;}
    
    /**
     * Sets the class membership of the centroid based on the majority class of its points.
     */
    public void assignClasses(){
        for (Centroid centroid : centroids){
            if (centroid.obtainPoints().length > 0)
                centroid.setClassMembership(Utilities.vote(centroid.obtainPoints()));
        }
    }
    
    /**
     * Empties the centroids of all their points (i.e.,
     * removes all point-centroid associations).
     */
    
    public void clearCentroids(){for (Centroid centroid : centroids){centroid.clearPoints();}}
}
