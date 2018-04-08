//package HomeWork1;
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
	
	public static void main(String[] args) throws Exception {

		//load data

		//find best alpha and build classifier with all attributes

		//build classifiers with all 3 attributes combinations
//		double trainingErr;
//		double testingErr;
//		double bestTrainingErr;
//		double bestTestingErr;
//		double minErr = Double.MAX_VALUE;
//		double tempErr = 0.0;
//		int[] bestIndices = new int[3];
//		Remove remove = new Remove();
		LinearRegression linearRegression = new LinearRegression();
		// TODO: Remove the path! only name.txt
		Instances trainingData = loadData("C:\\\\Users\\\\dafna\\\\IdeaProjects\\\\HomeWork1\\\\HomeWork1\\\\wind_training.txt");
//		Instances testingData = loadData("C:\\Users\\dafna\\IdeaProjects\\HomeWork1\\HomeWork1\\wind_testing.txt");
//		Instances newInst;
//		Instances bestInst = null;

		//linearRegression.initCoefficients(trainingData.numAttributes());
		linearRegression.findAlpha(trainingData);
//		linearRegression.buildClassifier(trainingData);
//		trainingErr = linearRegression.calculateMSE(trainingData);
//		testingErr = linearRegression.calculateMSE(testingData);
//		remove.setInputFormat(trainingData);
//		remove.setInvertSelection(true);
//		for (int i = 0; i < testingData.numAttributes(); i++) {
//			for (int j = 0; j < testingData.numAttributes(); j++) {
//				for (int k = 0; k < testingData.numAttributes(); k++) {
//					if (i != j && j != k && i != k){
//						System.out.println(i + " " + j + " " + k);
//						// Maybe we need to keep the last attribute as I did here below
//						int[] indices = {i, j, k};
//						remove.setAttributeIndicesArray(indices);
//						remove.setInputFormat(trainingData);
//						newInst = Filter.useFilter(trainingData, remove);
//						// 4 or 3?
//						//linearRegression.initCoefficients(3);
//						System.out.println(newInst.numAttributes());
//						linearRegression.buildClassifier(newInst);
//						tempErr = linearRegression.calculateMSE(newInst);
//						if (tempErr < minErr){
//							minErr = tempErr;
//							bestIndices = indices;
//							bestInst = newInst;
//						}
//						System.out.println("The error of " + i + " " + j + " " + k + " is: " + tempErr);
//					}
//				}
//			}
//		}
//		linearRegression.buildClassifier(bestInst);
//		bestTrainingErr = linearRegression.calculateMSE(trainingData);
//		bestTestingErr = linearRegression.calculateMSE(testingData);
//		//System.out.println("The chosen alpha is: " + linearRegression.getAlpha());
//		System.out.println("Training error with all features is: " + trainingErr);
//		System.out.println("Test error with all features is: " + testingErr);
//		System.out.println("Training error the features " + bestIndices[0] + " " + bestIndices[1] + " " + bestIndices[2] + " is " + bestTrainingErr);
//		System.out.println("Test error the features " + bestIndices[0] + " " + bestIndices[1] + " " + bestIndices[2] + " is " + bestTestingErr);
//
//	}
//
	}
}
		//load data
		
		//find best alpha and build classifier with all attributes

   		//build classifiers with all 3 attributes combinations
//		double trainingErr;
//		double testingErr;
//		double bestTrainingErr;
//		double bestTestingErr;
//		double minErr = Double.MAX_VALUE;
//		double tempErr;
//		int[] bestIndices = new int[3];
//		Remove remove = new Remove();
//		LinearRegression linearRegression = new LinearRegression();
//		Instances trainingData = loadData("C:\\Users\\dafna\\IdeaProjects\\HomeWork1\\HomeWork1\\wind_training.txt");
//		Instances testingData = loadData("C:\\Users\\dafna\\IdeaProjects\\HomeWork1\\HomeWork1\\wind_training.txt");
//		Instances newInst;
//		Instances bestInst = null;
//		linearRegression.initCoefficients(trainingData.numAttributes());
//
//		linearRegression.findAlpha(trainingData);
//		System.out.println("The chosen alpha is: " + linearRegression.getAlpha());
//
//		linearRegression.buildClassifier(trainingData);
//		trainingErr = linearRegression.calculateMSE(trainingData);
//		testingErr = linearRegression.calculateMSE(testingData);
////		remove.setInputFormat(trainingData);
////		remove.setInvertSelection(true);
//		for (int i = 0; i < testingData.numAttributes(); i++) {
//			for (int j = 1; j < testingData.numAttributes(); j++) {
//				for (int k = 2; k < testingData.numAttributes(); k++) {
//					int[] indices = {i, j, k};
////					remove.setAttributeIndicesArray(indices);
////					remove.setInputFormat(trainingData);
////					newInst = Filter.useFilter(trainingData, remove);
//					remove.setAttributeIndicesArray(indices);
//					remove.setInvertSelection(true);
//					remove.setInputFormat(trainingData);
//					newInst = Filter.useFilter(trainingData, remove);
//					linearRegression.buildClassifier(newInst);
//					tempErr = linearRegression.calculateMSE(newInst);
//					if (tempErr < minErr){
//						minErr = tempErr;
//						bestIndices = indices;
//						bestInst = newInst;
//					}
//					System.out.println("The error of " + i + " " + j + " " + k + " is: " + tempErr);
//
//				}
//			}
//		}
//		linearRegression.buildClassifier(bestInst);
//		bestTrainingErr = linearRegression.calculateMSE(trainingData);
//		bestTestingErr = linearRegression.calculateMSE(testingData);
//		System.out.println("The chosen alpha is: " + linearRegression.getAlpha());
//		System.out.println("Training error with all features is: " + trainingErr);
//		System.out.println("Test error with all features is: " + testingErr);
//		System.out.println("Training error the features " + bestIndices[0] + " " + bestIndices[1] + " " + bestIndices[2] + " is " + bestTrainingErr);
//		System.out.println("Test error the features " + bestIndices[0] + " " + bestIndices[1] + " " + bestIndices[2] + " is " + bestTestingErr);
//
//	}
