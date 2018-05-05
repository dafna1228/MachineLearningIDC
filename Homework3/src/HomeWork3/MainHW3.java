package HomeWork3;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import HomeWork3.Knn.DistanceCheck;
import weka.core.Instances;
import weka.filters.unsupervised.instance.Randomize;
public class MainHW3 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static double[] bestHyperParams(Instances data, int folds) throws Exception {
		// store best k, lp distance, weighting scheme
		double bestK = 0.0;
		double bestLpDistance = 0.0;
		double bestCrossValidationErr = Double.MAX_VALUE;
		double bestWeightingScheme = 0.0;
		double[] lpValues = {1.0, 2.0, 3.0, Double.POSITIVE_INFINITY};
		String[] weightingSchemes = {"uniform","weighted"};

		for (int k = 1; k < 21; k++){
			for (int lp = 0; lp < lpValues.length; lp++) {
				double lpDistance = lpValues[lp];
				for (int scheme = 0; scheme < weightingSchemes.length; scheme++) {
					// Getting the validation errors for all parameters combinations and keeping track of the best parameters
					Instances tempData = new Instances (data);
					Knn knn = new Knn();
					knn.buildClassifier(tempData);
					double crossValidationError = knn.crossValidationError(tempData, folds, lpDistance, k, weightingSchemes[scheme], DistanceCheck.Regular);
					if (crossValidationError < bestCrossValidationErr){
						bestCrossValidationErr = crossValidationError;
						bestK = k;
						bestLpDistance = lpDistance;
						bestWeightingScheme = ((weightingSchemes[scheme].equals("weighted")) ? 1.0 : 0.0);
					}
				}
			}
		}

		return new double[] {bestK, bestLpDistance, bestWeightingScheme, bestCrossValidationErr};
	}

	public static void main(String[] args) throws Exception {
        //TODO: complete the Main method
		Instances autoPrice = loadData("src\\HomeWork3\\auto_price.txt");
		// Shuffling the data
		autoPrice.randomize(new Random());
		FeatureScaler scaler = new FeatureScaler();
		// Standardizing the data
		Instances autoPriceScaled = scaler.scaleData(autoPrice);
		int[] foldOptions = {autoPriceScaled.numInstances(), 50, 10, 5, 3};
		DistanceCheck[] checkType = {DistanceCheck.Regular, DistanceCheck.Efficient};
		// Using 10 fold for the Hyper Parameters Search 
		double[] results = bestHyperParams(autoPrice, 10);
		double bestK = results[0];
		double bestLpDistance = results[1];
		String bestWeightingScheme =((results[2] == 1.0) ? "weighted" : "uniform");
		double bestAvgErr = results[3];

		System.out.println("----------------------------");
		System.out.println("Results for original dataset:");
		System.out.println("----------------------------");
		System.out.println("Cross validation error with K = " + (int)bestK + ", lp = "+ bestLpDistance +", majority function = "+ bestWeightingScheme +" for auto_price data is: " + bestAvgErr);
		System.out.println();
		results = bestHyperParams(autoPriceScaled, 10);
		bestK = results[0];
		bestLpDistance = results[1];
		bestWeightingScheme =((results[2] == 1.0) ? "weighted" : "uniform");
		bestAvgErr = results[3];
		System.out.println("----------------------------");
		System.out.println("Results for scaled dataset:");
		System.out.println("----------------------------");
		System.out.println("Cross validation error with K = " + (int)bestK + ", lp = "+ bestLpDistance +", majority function = "+ bestWeightingScheme +" for auto_price data is: " + bestAvgErr);
		System.out.println();
		System.out.println();
		// Getting cross validation error for all number of folds with regular and efficient distance checks
		for (int i = 0; i < foldOptions.length; i++) {
			System.out.println("----------------------------");
			System.out.println("Results for " + foldOptions[i] + " folds:");
			System.out.println("----------------------------");
			for (int j = 0; j < checkType.length; j++) {
				Instances tempData = new Instances (autoPriceScaled);
				Knn knn = new Knn();
				knn.buildClassifier(tempData);
				double crossValidationError = knn.crossValidationError(tempData, foldOptions[i], bestLpDistance, bestK, bestWeightingScheme, checkType[j]);
				long totalElapsedTime = knn.getTotalPredictionTime();
				long avgElapsedTime = totalElapsedTime / foldOptions[i];
				if (checkType[j] == DistanceCheck.Regular){
					System.out.println(checkType[j]);
					System.out.println("Cross validation error of regular knn on auto_price dataset is " + crossValidationError + " and the average elapsed time is " + avgElapsedTime);
				} else {
					System.out.println("Cross validation error of efficient knn on auto_price dataset is " + crossValidationError + " and the average elapsed time is " + avgElapsedTime);
				}
				System.out.println("The total elapsed time is: " + totalElapsedTime);
				System.out.println();
			}
		}
	}
}
