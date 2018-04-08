package HomeWork1;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class MainHW1 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	/**
	 * Sets the class index as the last attribute.
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);

		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	private static Instances getThreeFeaturesInstances(Instances data, int[] indices) throws Exception{
		Remove remove = new Remove();
		remove.setAttributeIndicesArray(indices);
		remove.setInvertSelection(true);
		remove.setInputFormat(data);
		Instances newInstances = Filter.useFilter(data, remove);
		newInstances.setClassIndex(newInstances.numAttributes() - 1);
		return newInstances;
	}
	public static void main(String[] args) throws Exception {
		double trainingErr;
		double testingErr;
		double bestTrainingErr;
		double bestTestingErr;
		double minErr = Double.MAX_VALUE;
		double tempErr = 0.0;
		int[] bestIndices = new int[4];
		LinearRegression allFeatures = new LinearRegression();
		LinearRegression threeFeatures = new LinearRegression();
		//load data from correct path
		Instances trainingData = null;
		Instances testingData = null;
		try {
			trainingData = loadData("src\\wind_training.txt");
			testingData = loadData("src\\wind_testing.txt");
		} catch (Exception e){
			trainingData = loadData("wind_training.txt");
			testingData = loadData("wind_testing.txt");
		}
		Instances newThreeInstances;
		Instances bestThreeInstances = null;
		//find best alpha and build classifier with all attributes
		allFeatures.buildClassifier(trainingData);
		double alpha = allFeatures.getAlpha();
		// The same alpha that was found should stay for the three features
		threeFeatures.setAlpha(alpha);
		// Calculating the training and the testing error for all the features
		trainingErr = allFeatures.calculateMSE(trainingData);
		testingErr = allFeatures.calculateMSE(testingData);
		//build classifiers with all 3 attributes combinations
		for (int i = 0; i < trainingData.numAttributes() - 1; i++) {
			for (int j = 0; j < trainingData.numAttributes() - 1; j++) {
				for (int k = 0; k < trainingData.numAttributes() - 1; k++) {
					if ((i != j && j != k && i != k) && (i < j && j < k)){
						// Going over all 3 unique index combinations of features from the training data
						int[] indices = {i, j, k, trainingData.numAttributes() - 1};
						// Getting an Instances object with the selected triplet of features (and the class value)
						newThreeInstances = getThreeFeaturesInstances(trainingData, indices);
						threeFeatures.buildClassifier(newThreeInstances);
						tempErr = threeFeatures.calculateMSE(newThreeInstances);
						// The triplet of features with the smallest error is the best ones
						System.out.println(trainingData.attribute(i) + " "+ trainingData.attribute(j) + " "+ trainingData.attribute(k) + " err " + tempErr);
						if (tempErr < minErr){
							minErr = tempErr;
							bestIndices[0] = i;
							bestIndices[1] = j;
							bestIndices[2] = k;
							bestThreeInstances = newThreeInstances;
						}
					}
				}
			}
		}
		// Setting the last index of the best indices array as the class value
		bestIndices[3] = trainingData.numAttributes() - 1;
		threeFeatures.buildClassifier(bestThreeInstances);
		bestTrainingErr = threeFeatures.calculateMSE(bestThreeInstances);
		// Getting an Instances object with the best triplet of features (and the class value) from the testing data
		bestThreeInstances = getThreeFeaturesInstances(testingData, bestIndices);
		bestTestingErr = threeFeatures.calculateMSE(bestThreeInstances);
		System.out.println("The chosen alpha is: " + alpha);
		System.out.println("Training error with all features is: " + trainingErr);
		System.out.println("Test error with all features is: " + testingErr);
		System.out.println("Training error the features " + bestThreeInstances.attribute(0).name() + " " + bestThreeInstances.attribute(1).name() + " " + bestThreeInstances.attribute(2).name() + " : " + bestTrainingErr);
		System.out.println("Test error the features " + bestThreeInstances.attribute(0).name() + " " + bestThreeInstances.attribute(1).name() + " " + bestThreeInstances.attribute(2).name() + " : " + bestTestingErr);

	}

}
