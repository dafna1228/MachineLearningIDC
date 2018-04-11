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
		Instances trainingCancer = loadData("cancer_train.txt");
		Instances testingCancer = loadData("cancer_test.txt");
		Instances validationCancer = loadData("cancer_validation.txt");
		
        //TODO: complete the Main method

		// Construct a tree with Entropy as the impurity measure using the training set.
		DecisionTree entropyTree = new DecisionTree();
		entropyTree.setGiniImpurity(false);
		entropyTree.buildClassifier(trainingCancer);
		entropyTree.printTree();

		// Calculate the average error on the validation set.
		double entropyErr = entropyTree.calcAvgError(validationCancer);
		System.out.println("Entropy Err: " + entropyErr);

		// Construct a tree with Gini as the impurity measure using the training set.
		DecisionTree giniTree = new DecisionTree();
		entropyTree.setGiniImpurity(true);
		giniTree.buildClassifier(trainingCancer);

		// Calculate the average error on the validation set.
		double giniErr = giniTree.calcAvgError(validationCancer);
		System.out.println("Gini Err: " + giniErr);

		// Choose the impurity measure that gave you the lowest validation error. Use this impurity measure
		// for the rest of the tasks.
		DecisionTree fullTree = null;
		if (giniErr < entropyErr){
			fullTree = giniTree;
			entropyTree.setGiniImpurity(true);
		} else {
			fullTree = entropyTree;
			entropyTree.setGiniImpurity(false);
		}

		// For each p-value cutoff value {1 (no pruning), 0.75, 0.5, 0.25, 0.05, 0.005} do the following:

			// Construct a tree and prune it according to the current cutoff value.
			// Calculate training & validation errors.
			// Calculate the tree average & max heights according to the validation set as described above.

		// Select the cutoff that resulted in the best validation error.

		// Calculate the test error for the tree corresponding to this configuration.

		// Print the corresponding tree to the console as described above.

		// Plot the training and validation error rates vs the p-value cutoff on the same graph
		// (two different lines in two different colors) in Excel
		// using the ‘Scatter with Smooth Lines and Markers’ graphing utility.


	}
}
