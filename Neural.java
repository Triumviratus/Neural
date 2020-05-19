/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural;

import neural.ActivationFunctions.LogisticFunction;
import neural.Recycled.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 *
 * @author ARZavier
 */
public class Neural {
    
    public static ArrayList<double[]> classes;
    public static String[] classificationFiles = {"abalone", "car", "segmentation"};
    public static String[] regressionFiles = {"wine", "forestfires", "machine"};
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        String[] allFiles = {"abalone", "car", "segmentation", "forestfires", "wine", "machine"};
        String[] CATEGORICAL_FEATURES = {"sex", "buying", "maintenance", "doors", "persons", 
                                         "safety", "lug_boot", "month", "day", "model"};
        String[] trial = {"machine"};
        String[] empty = {};
        
        for (String path : trial) {
            // Reads the File
            String header = "fold, Multi Hidden Layers, Multi Hidden Nodes, Multi METRIC, Condense Size, Condense METRIC, "
                    + "K-Means METRIC, K-Metroids METRIC\n";
            File outputFile = createNewFile("Data/Assignment3/outputs" + path);
            appendToFile(header, outputFile);
            
            System.out.println();
            System.out.println();
            System.out.println("Processing: " + path);
            String file = readEntireFile("Data/Assignment3/" + path + "_preprocessed.data"); // Reads in the data
            
            String[] lines = file.split("\r\n");
            DataPoint[] data = new DataPoint[lines.length]; // First line is feature labels
            String[] featureLabels = lines[0].split(",");
            classes = new ArrayList<>();
            
            // Generates the data points
            for (int i = 1; i < lines.length; i++) {
                data[i] = genDataPoint(lines[i], featureLabels, CATEGORICAL_FEATURES, path);
                if (!classes.contains(data[i].obtainTarget()))
                    classes.add(data[i].obtainTarget());
            }
            
            DataPoint[][] folds = fold(data); // Splits the dat into 10 roughly equal folds
            data = trim(data);
            
            for (int i = 0; i < folds.length; i++){
                DataPoint[] testData = folds[i];
                DataPoint[] trainData = new DataPoint[0];
                for (int j = 0; j < folds.length; j++){
                    if (j != i)
                        trainData = concat(folds[j], trainData);
                }
                String fold = "" + i + ",";
                int inputs = data[0].obtainNumberOfFeatures();
                int outputs = data[0].obtainTarget().length;
                
                MultiLayerFF tunedFF = Tuning.tuneFFNetwork(new int[]{3, 5, inputs, (inputs - 5)/10 + 1}, 
                        new double[]{0.1, 0.01, 0.001}, folds[0], folds[1], new LogisticFunction());
                fold = fold + Tuning.retMe;
                
                boolean classification = Utilities.contains(path, classificationFiles);
                if (classification){
                    System.out.println("Multilayer FF Accuracy: " + Validator.accuracy(tunedFF, testData));
                    fold = fold + Validator.accuracy(tunedFF, testData) + ",";
                } else {
                    System.out.println("Multilayer FF Error: " + Validator.squaredError(tunedFF, testData));
                    fold = fold + Validator.squaredError(tunedFF, testData) + ",";
                }
                Centroid[] condense = Condense.condense(trainData);
                double[] sigmas = new double[condense.length];
                for (int k = 0; k < condense.length; k++){sigmas[k] = condense[k].deriveDeviation();}
                fold = fold + condense.length + ",";
                System.out.println("Condense Cluster Done");
                
                KMeansModel means = new KMeansModel(trainData);
                means.cluster(15);
                System.out.println("K Means Cluster Done");
                KMetroidsModel metroids = new KMetroidsModel(trainData) {};
                metroids.cluster(15);
                System.out.println("K Metroids Cluster Done");
                
                RBF rbfCondense = new RBF (condense, sigmas, inputs, outputs, 0.01);
                RBF rbfMeans = new RBF (means, 0.01);
                RBF rbfMet = new RBF (metroids, 0.01);
                
                fold = runRBF (rbfCondense, trainData, testData, classification, fold) + ",";
                fold = fold + "0,";
                fold = runRBF(rbfMeans, trainData, testData, classification, fold) + ",";
                fold = runRBF(rbfMet, trainData, testData, classification, fold) + "\n";
                appendToFile(fold, outputFile);
            }
        }
    }
    
    private static String runRBF(RBF model, DataPoint[] trainData, DataPoint[] testData, boolean classification, String f){
        model.train(trainData, 0.01);
        for (Node[] layer : model.net){for (Node n: layer){System.out.println(Arrays.toString(n.weights));}}
        
        if (classification) {
            System.out.println("RBF Accuracy: " + Validator.accuracy(model, testData));
            f = f + Validator.accuracy(model, testData);
        } else {
            System.out.println("RBF Error: " + Validator.squaredError(model, testData));
            f = f + Validator.squaredError(model, testData);
        }
        return f;
    }
    
    /**
     * Creates a data point based on a string input
     * @param featureString String defining the data point
     * @return data point generated
     */
    
    private static DataPoint genDataPoint(String featureString, String[] featureLabels, 
            String[] categoricalFeatures, String filename) {
        String[] splice = featureString.split(",");
        ArrayList<Feature> features = new ArrayList<>(); // Assume Last Value is Class
        OneHot hot = new OneHot(filename);
        
        for (int i = 0; i < splice.length - 1; i++) {
            if (Arrays.asList(categoricalFeatures).contains(featureLabels[i])) {
                // Is Categorical
                String value = splice[i];
                double[] hotValues = hot.obtainOneHot(i, value);
                int instance = 0;
                for (double hv : hotValues) {
                    features.add(new Feature(hv, featureLabels[i] + instance));
                    instance++;
                }
            } else {
                // Is Continuous
                try {
                    features.add(new Feature(Double.parseDouble(splice[i]), featureLabels[i]));
                } catch (NumberFormatException e){
                    // Is Categorical
                    String value = splice[i];
                    double[] hotValues = hot.obtainOneHot(i, value);
                    int instance = 0;
                    for (double hv : hotValues){
                        features.add(new Feature (hv, featureLabels[i] + instance));
                        instance++;
                    }
                }
            }
        }
        if (Utilities.contains(filename, classificationFiles)) {
            // Classification
            String classMembership = splice[splice.length - 1];
            double[] classOneHot = hot.obtainOneHot(splice.length - 1, classMembership);
            System.out.println(Arrays.toString(classOneHot));
            DataPoint d = new DataPoint(features.toArray(new Feature[features.size()]), classOneHot);
            return d;
        } else {
            // Regression
            double[] regressionTarget = {Double.parseDouble(splice[splice.length - 1])};
            DataPoint d = new DataPoint(features.toArray(new Feature[features.size()]), regressionTarget);
            return d;
        }
    }
    
    private static String readEntireFile(String filePath) {
        // Reads the File
        File file = new File(filePath);
        String retString = "";
        if (file.exists()) {
            try {
                Scanner scan = new Scanner(file);
                scan.useDelimiter("\\Z");
                if (scan.hasNext())
                    retString = scan.next();
                scan.close();
            } catch (FileNotFoundException ignored){
                return "File Not Found for Path: " + file;
            }
        } else 
            System.out.println("File Does Not Exist");
        return retString;
    }
    
    /**
     * Adds the string to the end of a file
     * @param line string to be added
     * @param file the file to be added to
     */
    
    public static void appendToFile(String line, File file) {
        // Adds to File
        try {
            FileWriter writer = new FileWriter(file, true);
            writer.append(line);
            writer.close();
        } catch (IOException ignored){}
    }
    
    /**
     * Creates a file if there does not exist one already,
     * then return the file at the file path.
     * @param filePath file path
     * @return the file (either old or newly created) 
     */
    
    public static File createNewFile(String filePath) {
        // Creates a File
        String newPath = filePath;
        File file = new File(newPath + ".csv");
        int i = 2;
        
        while (file.exists()) {
            newPath = filePath + "-" + i;
            file = new File(newPath + ".csv");
            i += 1;
        }
        try {file.createNewFile();}
        catch (IOException ignored){ignored.printStackTrace();}
        return file;
    }
    
    /**
     * Folds the data into 10 relatively equal folds
     * @param points the data to be folded
     * @return a folded list
     */
    
    private static DataPoint[][] fold (DataPoint[] points) {
        points = Utilities.scramble(points); // Scrambles the data
        DataPoint[][] data = new DataPoint[10][points.length];
        int[] counters = new int[10];
        /**
         * So the elements go into the array in order (i.e.,
         * all the null values are at the end).
         */
        Random rand = new Random();
        for (int i = 0; i < 10; i++){
            // Ascertains all folds have one data point at minimum
            data[i][counters[i]++] = points[i];
        }
        
        for (int i = 10; i < points.length; i++) {
            int random = rand.nextInt(10);
            data[random][counters[random]++] = points[i];
            /**
             * Places the points into the folds in order
             * so as to avoid null values.
             */
        }
        
        for (int i = 0; i < data.length; i++) {
            // Eliminates trailing null values
            data[i] = trim(data[i]);
        }
        return data;
    }
    
    /**
     * Removes the null values from the end of the array
     * @param points the array
     * @return the values in the same order as they appear in points without trailing null values
     */
    
    private static DataPoint[] trim (DataPoint[] points) {
        int i = 0;
        DataPoint point = points[i++];
        while(point != null){point = points[i++];}
        return (DataPoint[]) Arrays.copyOf(points, i-1);
    }
    
    private static DataPoint[] concat(DataPoint[] a, DataPoint[] b){
        DataPoint[] retMe = new DataPoint[a.length + b.length];
        for (int i = 0; i < a.length; i++){retMe[i] = a[i];}
        for (int i = 0; i < b.length; i++){retMe[i + a.length] = b[i];}
        return retMe;
    }
}

