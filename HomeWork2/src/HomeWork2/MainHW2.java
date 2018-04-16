package HomeWork2;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.AttributeStats;

public class MainHW2 {

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

	public static void main(String[] args) throws Exception {
		Instances trainingCancer = loadData("src\\HomeWork2\\cancer_train.txt");
		Instances testingCancer = loadData("src\\HomeWork2\\cancer_test.txt");
		Instances validationCancer = loadData("src\\HomeWork2\\cancer_validation.txt");

		//TODO: complete the Main method

		// Construct a tree with Entropy as the impurity measure using the training set.
		DecisionTree entropyTree = new DecisionTree();
		entropyTree.setGiniImpurity(false);
		entropyTree.buildClassifier(trainingCancer);
		System.out.println("train err: " + entropyTree.calcAvgError(trainingCancer));
		//entropyTree.printTree();

		// Calculate the average error on the validation set.
		double entropyErr = entropyTree.calcAvgError(validationCancer);

		//System.out.println("E Err: " + entropyTree.calcAvgError(trainingCancer));

		System.out.println("Validation error using Entropy: " + entropyErr);


		// Construct a tree with Gini as the impurity measure using the training set.
		DecisionTree giniTree = new DecisionTree();
		giniTree.setGiniImpurity(true);
		giniTree.buildClassifier(trainingCancer);
		//giniTree.printTree();

		// Calculate the average error on the validation set.
		double giniErr = giniTree.calcAvgError(validationCancer);
		//System.out.println("Gini Err: " + giniTree.calcAvgError(trainingCancer));
		System.out.println("Validation error using Gini: " + giniErr);
		System.out.println("----------------------------------------------------");
		// Choose the impurity measure that gave you the lowest validation error. Use this impurity measure
		// for the rest of the tasks.
		boolean GiniImpurity = false;
		if (giniErr < entropyErr){
			GiniImpurity = true;
		}

		int bestPValueIndex = 0;
		double bestError = Double.MAX_VALUE;
		// For each p-value cutoff value {1 (no pruning), 0.75, 0.5, 0.25, 0.05, 0.005} do the following:
		double[] pValues = {1, 0.75, 0.5, 0.25, 0.05, 0.005};
		for (int i = 0; i < pValues.length; i++) {
			// Construct a tree and prune it according to the current cutoff value.
			DecisionTree pruneTree = new DecisionTree();
			pruneTree.setGiniImpurity(GiniImpurity);
			pruneTree.setpValueIndex(i);
			// pruning and building
			pruneTree.buildClassifier(validationCancer);
			// Calculate training & validation errors.
			double trainError = pruneTree.calcAvgError(trainingCancer);
			double validationError = pruneTree.calcAvgError(validationCancer);
			// Select the cutoff that resulted in the best validation error.
			if (validationError < bestError){
				bestError = validationError;
				bestPValueIndex = i;
			}
			// Calculate the tree average & max heights according to the validation set as described above.
			double avgHeight = pruneTree.getAvgHeight();
			double maxHeight = pruneTree.getMaxHeight();
			System.out.println("Decision Tree with p_value of: " + pValues[i]);
			System.out.println("The train error of the decision tree is " + trainError);
			System.out.println("Max height on validation data: " + maxHeight);
			System.out.println("Average height on validation data: " + avgHeight);
			System.out.println("The validation error of the decision tree is " + validationError);
		}

		// Calculate the test error for the tree corresponding to best validation error.
		DecisionTree bestTree = new DecisionTree();
		bestTree.setGiniImpurity(GiniImpurity);
		bestTree.setpValueIndex(bestPValueIndex);
		bestTree.buildClassifier(testingCancer);
		double testError = bestTree.calcAvgError(testingCancer);
		System.out.println("----------------------------------------------------");
		System.out.println("Best validation error at p_value = " + pValues[bestPValueIndex]);
		System.out.println("Test error with best tree: " + testError);
		// Print the corresponding tree to the console as described above.
		//bestTree.printTree();
	}
}
