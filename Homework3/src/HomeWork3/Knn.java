package HomeWork3;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.InstanceComparator;

class DistanceCalculator {
    /**
    * We leave it up to you wheter you want the distance method to get all relevant
    * parameters(lp, efficient, etc..) or have it has a class variables.
    */
    public double distance (Instance one, Instance two, double lpDistance) {
        if (Double.isInfinite(lpDistance)){
            return lInfinityDistance(one, two);
        } else {
            return lpDistance(one, two, lpDistance);
        }
    }

    /**
     * Returns the Lp distance between 2 instances.
     * @param one
     * @param two
     */
    private double lpDistance(Instance one, Instance two, double lpDistance) {
        double result = 0.0;
        for (int i = 0; i< one.numAttributes() - 1; i++){
            result += Math.pow(Math.abs(one.value(i) - two.value(i)), lpDistance);
        }
        return Math.pow(result, (1 / lpDistance));
    }

    /**
     * Returns the L infinity distance between 2 instances.
     * @param one
     * @param two
     * @return
     */
    private double lInfinityDistance(Instance one, Instance two) {
        double maxResult = 0.0;
        for (int i = 0; i < one.numAttributes() - 1; i++){
            if (maxResult < Math.abs(one.value(i) - two.value(i))){
                maxResult = Math.abs(one.value(i) - two.value(i));
            }
        }
        return maxResult;
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     * @param one
     * @param two
     * @return
     */
    private double efficientLpDistance(Instance one, Instance two) {
        return 0.0;
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     * @param one
     * @param two
     * @return
     */
    private double efficientLInfinityDistance(Instance one, Instance two) {
        return 0.0;
    }
}

public class Knn implements Classifier {

    public enum DistanceCheck{Regular, Efficient}
    private Instances m_trainingInstances;
    private double k;
    private double lpDistance;
    private String weightingScheme;


    // setter for the data
    public void setData(Instances instances) {
        m_trainingInstances = instances; }

    // setter for k
    public void setK(double k1) {
        this.k = k1;    }

    // setter for the lpDistance1
    public void setLpDistance(double lpDistance1) {
        this.lpDistance = lpDistance1;    }

    // setter for the weightingScheme1
    public void setWeightingScheme(String weightingScheme1) {
        this.weightingScheme = weightingScheme1; }

    // setter for the distance check
    public void setDistanceCheck(String  instances) throws Exception {
        // how the fuck do i use enums
    }

    @Override
    /**
     * Build the knn classifier. In our case, simply stores the given instances for 
     * later use in the prediction.
     * @param instances
     */
    public void buildClassifier(Instances instances) throws Exception {

    }

    /**
     * Returns the knn prediction on the given instance.
     * @param instance
     * @return The instance predicted value.
     */
    public double regressionPrediction(Instance instance, Instances data) {
        // get the prediction using a calculated average
        if (weightingScheme.equals("weighted")) {
            return getWeightedAverageValue(data, instance);
        } else {
            return getAverageValue(data, instance);
        }
    }


    /**
     * Caclcualtes the average error on a give set of instances.
     * The average error is the average absolute error between the target value and the predicted
     * value across all insatnces.
     * @return
     */
    public double calcAvgError (Instances trainingData, Instances validationData){
        double sumErrors = 0.0;
        for (int i = 0; i < validationData.numInstances(); i++){
            Instance instance = validationData.instance(i);
            sumErrors += Math.abs(regressionPrediction(instance, trainingData) - instance.classValue());
        }
        return sumErrors / (double) validationData.numInstances();
    }

    /**
     * Calculates the cross validation error, the average error on all folds.
     * @return The cross validation error.
     */
    public double crossValidationError(double lpDistance, double k, String weightingScheme){
        // set stuff up
        setLpDistance(lpDistance);
        setWeightingScheme(weightingScheme);
        setK(k);
        double sumAvgErrors = 0.0;
        // create 10-fold instances array
        Instances[][] tenFoldInstances = createTenFoldInstances();
        for (int i = 0; i < 10; i++){
            Instances trainingData = tenFoldInstances[0][i];
            Instances validationData = tenFoldInstances[1][i];
            sumAvgErrors += calcAvgError(trainingData, validationData);
        }
        return sumAvgErrors / 10.0;
    }


    /**
     * Finds the k nearest neighbors.
     * @param instance
     */
    /* Collection of your choice */
    public Instances findNearestNeighbors(Instance instance, Instances data) {
        Instances neighbours = new Instances(data, 0);
        Instances tempData = new Instances(data);
        // get the closes neighbour from data, add it to neighbours and remove it from data, k times
        for (int i = 0; i < k; i++){
            // itterate on all instances and find the closes one
            int closesNeighbour = Integer.MAX_VALUE;
            double closesNeighbourDistance = Double.MAX_VALUE;
            for (int j = 0; j < tempData.numInstances(); j++){
                DistanceCalculator distance = new DistanceCalculator();
                double tempNeighbourDistance =  distance.distance(instance, tempData.instance(j), lpDistance);
                if (tempNeighbourDistance < closesNeighbourDistance){
                    closesNeighbour = j;
                    closesNeighbourDistance = tempNeighbourDistance;
                }
            }
            neighbours.add(tempData.instance(closesNeighbour));
            tempData.remove(closesNeighbour);
        }
        return neighbours;
    }

    /**
     * Cacluates the average value of the given elements in the collection.
     * @param
     * @return
     */
    public double getAverageValue (Instances data, Instance instance) {
        double sumNeighbourValues = 0.0;
        Instances neighbours = findNearestNeighbors(instance, data);
        for (int i = 0; i < neighbours.numInstances(); i++) {
            sumNeighbourValues += (neighbours.instance(i).classValue());
        }
        return (1.0 / k) * sumNeighbourValues;
    }

    /**
     * Calculates the weighted average of the target values of all the elements in the collection
     * with respect to their distance from a specific instance.
     * @return
     */
    public double getWeightedAverageValue(Instances data, Instance instance) {
        double sumNeighbourValues = 0.0;
        double wieght;
        double sumWieght = 0.0;
        Instances neighbours = findNearestNeighbors(instance, data);
        for (int i = 0; i < neighbours.numInstances(); i++) {
                // the weighting scheme is weighted, calc the weight of the instance
                DistanceCalculator distanceCalc = new DistanceCalculator();
                double distance = distanceCalc.distance(instance, neighbours.instance(i), lpDistance);
                wieght = 1.0 / Math.pow(distance, 2);
                if (Double.isInfinite(wieght)) {
                    wieght = instance.classValue();
                }
                sumWieght += wieght;
            sumNeighbourValues += (neighbours.instance(i).classValue()) * wieght;
        }
        return sumNeighbourValues / sumWieght;

    }


    // divide the data in m_trainingInstances to 10 equally sizes Instances objects in an array
    public Instances[][] createTenFoldInstances() {
        Instances[] trainingArr = new Instances[10];
        Instances[] validationArr = new Instances[10];

        // init the instances
        for (int i = 0; i < trainingArr.length; i++){
            trainingArr[i] = new Instances(m_trainingInstances, 0);
            validationArr[i] = new Instances(m_trainingInstances, 0);
        }
        // populate the instances
        for (int i = 0; i < trainingArr.length; i++){
            for (int j = 0; j < m_trainingInstances.numInstances(); j++){
                // use modulu to equally divide 1/10 of the instances
                if (j%10 == i) {
                    validationArr[i].add(m_trainingInstances.instance(j));
                } else {
                    trainingArr[i].add(m_trainingInstances.instance(j));
                }
            }
        }
        return new Instances[][] {trainingArr, validationArr} ;
    }


        @Override
    public double[] distributionForInstance(Instance arg0) throws Exception {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public double classifyInstance(Instance instance) {
        // TODO Auto-generated method stub - You can ignore.
        return 0.0;
    }
}
