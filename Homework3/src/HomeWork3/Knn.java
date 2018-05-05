package HomeWork3;

import java.util.Comparator;
import java.util.Iterator;
import java.util.PriorityQueue;

import HomeWork3.Knn.DistanceCheck;
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
    public double distance (Instance one, Instance two, double lpDistance, DistanceCheck checkType, double threshold) {
    	if (checkType == DistanceCheck.Regular){
	        if (Double.isInfinite(lpDistance)){
	            return lInfinityDistance(one, two);
	        } else {
	            return lpDistance(one, two, lpDistance);
	        }
    	} else {
    		if (Double.isInfinite(lpDistance)){
	            return efficientLInfinityDistance(one, two, threshold);
	        } else {
	            return efficientLpDistance(one, two, lpDistance, threshold);
	        }
    	}
    }

    /**
     * Returns the Lp distance between 2 instances.
     * @param one
     * @param two
     */
    private double lpDistance(Instance one, Instance two, double lpDistance) {
        double result = 0.0;
        for (int i = 0; i < one.numAttributes() - 1; i++){
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
    private double efficientLpDistance(Instance one, Instance two, double lpDistance, double threshold) {
    	double result = 0.0;
        for (int i = 0; i < one.numAttributes() - 1; i++){
            result += Math.pow(Math.abs(one.value(i) - two.value(i)), lpDistance);
            if (result > threshold) break;
        }
        return Math.pow(result, (1 / lpDistance));
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     * @param one
     * @param two
     * @return
     */
    private double efficientLInfinityDistance(Instance one, Instance two, double threshold) {
        double maxResult = 0.0;
        for (int i = 0; i < one.numAttributes() - 1; i++){
            if (maxResult < Math.abs(one.value(i) - two.value(i))){
                maxResult = Math.abs(one.value(i) - two.value(i));
            }
            if (maxResult > threshold) break; 
        }
        return maxResult;
    }
}

public class Knn implements Classifier {

    public enum DistanceCheck{Regular, Efficient}
    private Instances m_trainingInstances;
    private Instances currentTraining;
    private double k;
    private double lpDistance;
    private String weightingScheme;
    private DistanceCheck distanceCheck;
    private long[] predictionTimes;
    private int timeIndex;

    // Setter for k
    public void setK(double k1) {
        this.k = k1;    
    }

    // Setter for the lpDistance1
    public void setLpDistance(double lpDistance1) {
        this.lpDistance = lpDistance1;    
    }

    // Setter for the weightingScheme1
    public void setWeightingScheme(String weightingScheme1) {
        this.weightingScheme = weightingScheme1; 
    }

    // Setter for the distance check
    public void setDistanceCheck(DistanceCheck type) {
    	this.distanceCheck = type;
    }
    
    // Setter for the current training set (when splitting the set)
    public void setCurrentTraining(Instances data) {
    	this.currentTraining = data;
    }
    
    public long getTotalPredictionTime(){
    	long elapsedTime = 0;
    	for (int i = 0; i < predictionTimes.length; i++) {
			elapsedTime += predictionTimes[i];
		}
    	return elapsedTime;
    }

    @Override
    /**
     * Build the knn classifier. In our case, simply stores the given instances for 
     * later use in the prediction.
     * @param instances
     */
    public void buildClassifier(Instances instances) throws Exception {
    	m_trainingInstances = instances;
    }

    /**
     * Returns the knn prediction on the given instance.
     * @param instance
     * @return The instance predicted value.
     */
    public double regressionPrediction(Instance instance) {
    	long currentTime = System.nanoTime();
    	PriorityQueue<Instance> neighbors = findNearestNeighbors(instance);
    	double avgValue;
        // get the prediction using a calculated average
        if (weightingScheme.equals("weighted")) {
            avgValue = getWeightedAverageValue(neighbors);
        } else {
        	avgValue = getAverageValue(neighbors);
        }
        // Measuring the prediction time
        predictionTimes[timeIndex] = System.nanoTime() - currentTime;
        return avgValue;
    }


    /**
     * Caclcualtes the average error on a give set of instances.
     * The average error is the average absolute error between the target value and the predicted
     * value across all insatnces.
     * @return
     */
    public double calcAvgError (Instances validationData){
        double sumErrors = 0.0;
        for (int i = 0; i < validationData.numInstances(); i++){
            Instance instance = validationData.instance(i);
            sumErrors += Math.abs(regressionPrediction(instance) - instance.classValue());
        }
        return sumErrors / (double) validationData.numInstances();
    }

    /**
     * Calculates the cross validation error, the average error on all folds.
     * @return The cross validation error.
     */
    public double crossValidationError(Instances data, int num_of_folds, double lpDistance, double k, String weightingScheme, DistanceCheck checkType){
    	// Setting the parameters
    	setLpDistance(lpDistance);
        setWeightingScheme(weightingScheme);
        setK(k);
        setDistanceCheck(checkType);
    	double sumAvgErrors = 0.0;
        // create X-fold instances array
        Instances[][] XFoldInstances = createXFoldInstances(num_of_folds);
        predictionTimes = new long [num_of_folds];
        timeIndex = 0;
        for (int i = 0; i < num_of_folds; i++){
            Instances trainingData = XFoldInstances[0][i];
            Instances validationData = XFoldInstances[1][i];
            setCurrentTraining(trainingData);
            sumAvgErrors += calcAvgError(validationData);
            timeIndex++;
        }
        return sumAvgErrors / (double)num_of_folds;
    }


    /**
     * Finds the k nearest neighbors.
     * @param instance
     */
    /* Collection of your choice */
    public PriorityQueue<Instance> findNearestNeighbors(Instance instance) {
    	// Creating a max heap for the k nearest neighbors with a custom comparator - by the instance's weight which contains the distance from the instance
    	PriorityQueue<Instance> neighbors = new PriorityQueue<Instance>(new Comparator<Instance>() {
    	    public int compare(Instance x, Instance y) {
    	        if (x.weight() < y.weight()) return 1;
    	        if (x.weight() == (y.weight())) return 0;
    	        return -1;
    	    }
    	});
    	DistanceCalculator distanceCalc = new DistanceCalculator();
    	// Initializing the neighbors max heap with the first k instances of the current training set
		for (int i = 0; i < k; i++) {
			if (i < currentTraining.numInstances()) {
				currentTraining.instance(i).setWeight(distanceCalc.distance(instance, currentTraining.instance(i),
						lpDistance, DistanceCheck.Regular, Double.MAX_VALUE));
				neighbors.add(currentTraining.instance(i));

			}
		}
		// Iterate on all current training instances and find the k closest ones
		for (int j = 0; j < currentTraining.numInstances(); j++) {
			double kthNeighborDistance = neighbors.peek().weight();
			// The threshold is the Kth neighbor's distance to the power of
			// lpDistance (unless lpDistance = Infinity)
			double threshold;
			if (Double.isInfinite(lpDistance)) {
				threshold = kthNeighborDistance;
			} else {
				threshold = Math.pow(kthNeighborDistance, lpDistance);
			}
			distanceCalc = new DistanceCalculator();
			currentTraining.instance(j).setWeight(
					distanceCalc.distance(instance, currentTraining.instance(j), lpDistance, distanceCheck, threshold));
			if (currentTraining.instance(j).weight() < kthNeighborDistance) {
				// Removing the Kth neighbor and adding the closer instance
				neighbors.poll();
				neighbors.add(currentTraining.instance(j));
			}
		}
		return neighbors;
	}

    
    /**
     * Cacluates the average value of the given elements in the collection.
     * @param
     * @return
     */
    public double getAverageValue (PriorityQueue<Instance> neighbors) {
        double sumNeighborValues = 0.0;   
        Iterator<Instance> itr = neighbors.iterator();
        while(itr.hasNext()){
        	sumNeighborValues += (itr.next().classValue());
        }
        return (1.0 / k) * sumNeighborValues;
    }

    /**
     * Calculates the weighted average of the target values of all the elements in the collection
     * with respect to their distance from a specific instance.
     * @return
     */
    public double getWeightedAverageValue(PriorityQueue<Instance> neighbors) {
        double sumNeighborValues = 0.0;
        double distance;
        double weight;
        double sumWeight = 0.0;
       Iterator<Instance> itr = neighbors.iterator();
        while(itr.hasNext()){
        	Instance neighbor = itr.next();
        	// The neighbors' instances hold the distance from the instance as their weights (distance = neighbor.weight())
        	distance = neighbor.weight();
        	if (distance == 0){
        		// As instructed in Piazza
        		return neighbor.classValue();
        	} else {
        		weight = 1.0 / Math.pow(distance, 2);
        		sumWeight += weight;
        		sumNeighborValues += (neighbor.classValue()) * weight;
        	}
        }
        return sumNeighborValues / sumWeight;
    }


    // Divide the data in m_trainingInstances to 10 equally sizes Instances objects in an array
    public Instances[][] createXFoldInstances(int folds) {
        Instances[] trainingArr = new Instances[folds];
        Instances[] validationArr = new Instances[folds];
        for (int i = 0; i < folds; i++){
        	trainingArr[i] = m_trainingInstances.trainCV(folds, i);
        	validationArr[i] = m_trainingInstances.testCV(folds, i);	
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
