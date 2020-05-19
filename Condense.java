/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural.Recycled;

import java.util.ArrayList;

/**
 *
 * @author ARZavier
 */
public class Condense {
    
    /**
     * Condenses the dataset
     * @param data data to be condensed
     * @return the condensed dataset
     */
    
    public static Centroid[] condense (neural.Recycled.DataPoint[] data) {
        ArrayList<DataPoint> newData = new ArrayList<>();
        data = Utilities.scramble(data);
        newData.add(data[0]); // Inserts the first item
        
        int size = newData.size();
        int delta = 1;
        
        while (Math.abs(newData.size() - size) < delta) {
            // until it does not change that much
            for (neural.Recycled.DataPoint x : data){
                Double[] dists = Utilities.getDistancesToPoints(x, newData); // Gets dist to each point in new set
                int indx = Utilities.findIndexOfMinimum(dists);
                if (!categoricalEquals(newData.get(indx).obtainTarget(), x.obtainTarget())){
                    // Checks if the closest one is different from this one
                    newData.add(x);
                }
            }
        }
        Centroid[] retMe = convertToCentroid(newData);
        for (DataPoint point : data){
            // Find distances from all current centroids to this point
            Double[] distances = Utilities.getDistancesToPoints(point, retMe);
            // Find the centroid this point is closest to
            Centroid closestCentroid = retMe[Utilities.findIndexOfMinimum(distances)];
            // Set the group of this point to be the group of the centroid
            closestCentroid.addPoint(point);
        }
        return retMe; // Condensed Dataset
    }
    
    private static Centroid[] convertToCentroid(ArrayList<DataPoint> data) {
        Centroid[] retMe = new Centroid[data.size()];
        for (int i = 0; i < data.size(); i++){retMe[i] = new Centroid(data.get(i));}
        return retMe;
    }
    
    private static boolean categoricalEquals(double[] a, double[] b) {
        if (a.length == b.length){
            for (int i = 0; i < a.length; i++){
                if (a[i] != b[i])
                        return false;
            }
            return true;
        } else
            return false;
    }
}
