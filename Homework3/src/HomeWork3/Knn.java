package HomeWork3;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

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
    public double regressionPrediction(Instance instance) {
        return 0.0;
    }

    /**
     * Caclcualtes the average error on a give set of instances.
     * The average error is the average absolute error between the target value and the predicted
     * value across all insatnces.
     * @param insatnces
     * @return
     */
    public double calcAvgError (Instances insatnces){

        return 0.0;
    }

    /**
     * Calculates the cross validation error, the average error on all folds.
     * @param insancesXFold Insances used for the cross validation
     * @param validationIndex The number of folds to use.
     * @return The cross validation error.
     */
    public double crossValidationError(Instances[] insancesXFold, int validationIndex){

        return 0.0;
    }


    /**
     * Finds the k nearest neighbors.
     * @param instance
     */
    /* Collection of your choice */
    public Instances findNearestNeighbors(Instance instance, int k) {
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
    public Instances[] createTenFoldInstances() {
        Instances[] returnArr = new Instances[10];
        // init the instances
        for (int i = 0; i < returnArr.length; i++){
            returnArr[i] = new Instances(m_trainingInstances, 0);
        }
        // populate the instances
        for (int i = 0; i < m_trainingInstances.numInstances(); i++){
            returnArr[i%10].add(m_trainingInstances.instance(i));
        }
        return returnArr;
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
