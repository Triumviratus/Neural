/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural.Recycled;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author ARZavier
 */
public class Centroid extends DataPoint {
    private List<DataPoint> dataPoints; // Stores the data points that belong to this centroid
    
    /**
     * Creates a centroid located at the features given as a parameter
     * @param features the features identify the location of this object in the hyperspace
     */
    
    public Centroid (Feature[] features) {
        super(features);
        dataPoints = new ArrayList<>();
        this.target = new double[]{};
    }
    /**
     * Create a centroid located at a given data point
     * @param dataPoint the location at where to put the centroid
     */
    public Centroid(DataPoint dataPoint){this(dataPoint.features);}
    
    /**
     * Obtains the points that belong to the centroid
     * @return the data points that belong to this centroid
     */
    
    public DataPoint[] obtainPoints(){return Utilities.listToArr(dataPoints);}
    
    /**
     * Adds a point to the centroid
     * @param d the point to be added
     */
    
    public void addPoint(DataPoint d){dataPoints.add(d);}
    
    /**
     * Moves the centroid to the average of its points
     */
    
    public void moveToAvg(){
        if (dataPoints.size() > 0){
            // Checks if any points belong to the centroid
            DataPoint average = Utilities.findAverageOfPoints(this.obtainPoints());
            features = average.obtainFeatures();
        }
    }
    
    /**
     * Moves the centroid to the given point
     * @param point point at which to place the centroid
     */
    public void moveToPoint(DataPoint point) {features = point.obtainFeatures();}
    
    public double deriveDeviation(){
        double deviation = 0;
        for (int i = 0; i < dataPoints.size(); i++) {deviation += Utilities.obtainDistance(this, dataPoints.get(i));}
        if (dataPoints.size() > 0){
            double value = Math.sqrt(deviation/dataPoints.size());
            return (value >= 0.3) ? value : 0.3;
        } else
            return 0.3;
    }
    
    /**
     * Checks to see whether a point belongs to the centroid
     * @param point the point being queried
     * @return true if the point is associated with the centroid
     */
    public boolean contains(DataPoint point){return dataPoints.contains(point);}
    
    /**
     * Removes all associations with data points (i.e., this
     * centroid no longer has any points that belong to it).
     */
    public void clearPoints(){dataPoints = new ArrayList<>();}
}
