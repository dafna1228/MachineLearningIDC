package HomeWork3;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;

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

	public static double[] bestHyperParams(Instances data) throws IOException {
		// store best k, lp distance, weighting scheme
		double bestK = 0.0;
		double bestLpDistance = 0.0;
		double bestCrossValidationErr = Double.MAX_VALUE;
		double bestWeightingScheme = 0.0;
		double[] lpValues = {1.0, 2.0, 3.0, Double.MAX_VALUE};
		String[] weightingSchemes = {"uniform","weighted"};

		for (int k = 1; k < 21; k++){
			for (int lp = 0; lp < lpValues.length; lp++) {
				double lpDistance = lpValues[lp];
				for (int scheme = 0; scheme < weightingSchemes.length; scheme++) {
					// init knn object
					Knn knn = new Knn();
					knn.setData(data);
					double crossValidationError = knn.crossValidationError(lpDistance, k, weightingSchemes[scheme]);
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
		FeatureScaler scaler = new FeatureScaler();
		Instances autoPriceScaled = scaler.scaleData(autoPrice);
		System.out.println(autoPriceScaled.firstInstance());
		System.out.println(autoPriceScaled.lastInstance());
		// find the best hyper parameters (K – number of neighbors, Lp distance measure, weighting scheme)
		double[] lpValues = {1.0, 2.0, 3.0, Double.MAX_VALUE};
		String[] weightingSchemes = {"uniform","weighted"};

		double[] results = bestHyperParams(autoPrice);
		double bestK = results[0];
		double bestLpDistance = results[1];
		String bestWeightingScheme =((results[2] == 1.0) ? "weighted" : "uniform");
		double bestAvgErr = results[3];

		System.out.println("----------------------------");
		System.out.println("Results for original dataset:");
		System.out.println("----------------------------");
		System.out.println("Cross validation error with K = " + bestK + ", lp = "+ bestLpDistance +", majority function = "+ bestWeightingScheme +" for auto_price data is: "+bestAvgErr);

		results = bestHyperParams(autoPriceScaled);
		bestK = results[0];
		bestLpDistance = results[1];
		bestWeightingScheme =((results[2] == 1.0) ? "weighted" : "uniform");
		bestAvgErr = results[3];
		System.out.println("----------------------------");
		System.out.println("Results for scaled dataset:");
		System.out.println("----------------------------");
		System.out.println("Cross validation error with K = " + bestK + ", lp = "+ bestLpDistance +", majority function = "+ bestWeightingScheme +" for auto_price data is: "+bestAvgErr);


	}

}
