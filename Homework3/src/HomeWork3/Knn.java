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
    public double distance (Instance one, Instance two) {
        return 0.0;
    }

    /**
     * Returns the Lp distance between 2 instances.
     * @param one
     * @param two
     */
    private double lpDistance(Instance one, Instance two) {
        return 0.0;
    }

    /**
     * Returns the L infinity distance between 2 instances.
     * @param one
     * @param two
     * @return
     */
    private double lInfinityDistance(Instance one, Instance two) {
        return 0.0;
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


    // setter for the data
    public void setData(Instances instances) {
        m_trainingInstances = instances;
    }

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
    public double regressionPrediction(Instance instance, Instances data, double k) {
        double sumNeighbourValues = 0.0;
        Instances neighbours = findNearestNeighbors(instance, data, k);
        for (int i = 0; i < neighbours.numInstances(); i++) {
            sumNeighbourValues += neighbours.instance(i).classValue();
        }
        return  (1.0 / k) * sumNeighbourValues;
    }

    /**
     * Caclcualtes the average error on a give set of instances.
     * The average error is the average absolute error between the target value and the predicted
     * value across all insatnces.
     * @return
     */
    public double calcAvgError (Instances trainingData, Instances validationData, double k){
        //TODO: we need not to test error across all instances, only in 9/10 of it
        double sumErrors = 0.0;
        for (int i = 0; i < validationData.numInstances(); i++){
            Instance instance = validationData.instance(i);
            //TODO: does "average absolute error" means absolute on each instance or on the sum?
            sumErrors += Math.abs(regressionPrediction(instance, trainingData, k) - instance.classValue());
        }
        return sumErrors / (double) validationData.numInstances();
    }

    /**
     * Calculates the cross validation error, the average error on all folds.
     * @return The cross validation error.
     */
    public double crossValidationError(double lpDistance, double k, String weightingScheme){
        double sumAvgErrors = 0.0;
        // create 10-fold instances array
        Instances[][] tenFoldInstances = createTenFoldInstances();
        for (int i = 0; i < tenFoldInstances.length; i++){
            Instances trainingData = tenFoldInstances[0][i];
            Instances validationData = tenFoldInstances[1][i];
            if (weightingScheme.equals("uniform")) {
                // the weighting scheme is uniform

            } else {
                // the weighting scheme is weighted
            }
            sumAvgErrors += calcAvgError(trainingData, validationData, k);
        }
        return sumAvgErrors;
    }


    /**
     * Finds the k nearest neighbors.
     * @param instance
     */
    /* Collection of your choice */
    public Instances findNearestNeighbors(Instance instance, Instances data, double k) {
        Instances neighbours = new Instances(data, 0);

        return null;
    }

    /**
     * Cacluates the average value of the given elements in the collection.
     * @param
     * @return
     */
    public double getAverageValue (/* Collection of your choice */) {
        return 0.0;
    }

    /**
     * Calculates the weighted average of the target values of all the elements in the collection
     * with respect to their distance from a specific instance.
     * @return
     */
    public double getWeightedAverageValue(/* Collection of your choice */) {
        return 0.0;
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
                    trainingArr[i].add(m_trainingInstances.instance(j));
                } else {
                    validationArr[i].add(m_trainingInstances.instance(j));
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
