package HomeWork2;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;

import java.util.LinkedList;
import java.util.Queue;

class Node {
	Node[] children;
	Node parent;
	int attributeIndex;
	String attributeValue;
	double returnValue;
	Instances data;
	//
}

public class DecisionTree implements Classifier {
	private Node rootNode;
	private boolean giniImpurity = false;

	public void setGiniImpurity(boolean isGini) {
		giniImpurity = isGini;
	}

	// Builds a decision tree from the training data. buildClassifier is separated from buildTree in order
	// to allow you to do extra preprocessing before calling buildTree method or post processing after.
	// Input: Instances object.
	@Override
	public void buildClassifier(Instances data) throws Exception {
		rootNode.data = data;
		buildTree(data);
	}

	// Builds the decision tree on given data set using either a recursive or queue algorithm.
	// Input: Instances object (probably the training data set or subset in a recursive method).
	public void buildTree(Instances data) throws Exception {
		Queue<Node> queue = new LinkedList<Node>();
		queue.add(rootNode);
		while ( !queue.isEmpty()){
			Node currNode = queue.peek();
			// If training examples in n are perfectly classified-
			// Then mark node as complete and continue to next node in Q
			if (currNode.data.numInstances() == 0) {
				queue.remove();
			} else {
				// Assign the decision attribute for the node data
				int decisionAttr = getDecisionAttr(currNode.data);
				int attrNumVal = currNode.data.attribute(decisionAttr).numValues();
				Node[] currChildren = new Node[attrNumVal];
				// For each boundary value of decisionAttr, create a new descendant of the node
				for (int i = 0; i < attrNumVal; i++){
					Node childNode = new Node();
					childNode.parent = currNode;
					childNode.attributeIndex = decisionAttr;
					childNode.data = new Instances(data,0);
					childNode.attributeValue = currNode.data.attribute(decisionAttr).value(i);
					childNode.returnValue = ;// the class value with the most instances?
					currChildren[i] = childNode;
				}
				// Distribute training examples to descendant nodes
				for (int i = 0; i < currNode.data.numInstances(); i++){
					Instance instance = currNode.data.instance(i);
					String value = instance.stringValue(decisionAttr);
					for (int j = 0; j < attrNumVal; j++){
						if (value.equals(data.attribute(decisionAttr).value(j))){
							currChildren[i].data.add(instance);
						}
					}
				}
				currNode.children = currChildren;

				// Insert all (non empty) descendant nodes to Q
				for (int i = 0; i < attrNumVal; i++){
					queue.add(currChildren[i]);
				}
			}
		}
	}

	// Return the classification of the instance.
	// Input: Instance object.
	// Output: double number, 0 or 1, represent the classified class
	@Override
	public double classifyInstance(Instance instance) {
		double retClassification = 0.0;
		Node currNode = rootNode;
		// walk down the tree
		while (currNode.children.length != 0 ){
			// set currNode as the child of currNode that has the attribute value same as the instance
			for(int i = 0; i < currNode.children.length; i++) {
				String instanceAttrValue = instance.stringValue(currNode.attributeIndex);
				if (instanceAttrValue.equals(currNode.children[i].attributeValue)){
					currNode = currNode.children[i];
				}
			}
		}
		return currNode.returnValue;
	}

	// Calculate the average error on a given instances set (could be the training, test or validation set).
	// The average error is the total number of classification mistakes on the input instances set divided by
	// the number of instances in the input set.
	// Input: Instances object.
	// Output: Average error (double).
	public double calcAvgError(Instances data) {
		int mistakes = 0;
		for (int i=0; i < data.numInstances(); i++) {
			double prediction = classifyInstance(data.instance(i));
			// if the prediction is different from the real class value
			if ( prediction != data.instance(i).classValue()){
				mistakes++;
			}
		}
			return (double) mistakes / (double) data.numInstances();
    }

	// calculates the gain (giniGain or informationGain depending on the impurity measure) of splitting the input data according to the attribute.
	// Input: Instances object (a subset of the training data), attribute index (int).
	// Output: The gain (double).
	public double calcGain(Instances data, int attrIndex) {
		double sumValues = 0.0;
		double[] S = generateProb(data);
		Attribute attribute = data.attribute(attrIndex);
		for (int i=0; i < attribute.numValues(); i++){
			// create Instances object with instances that have value i in the attribute
			Instances SvInstances = new Instances(data, 0);
			for (int j = 0; j < data.numInstances(); j++) {
				Instance instance = data.instance(i);
				if (instance.stringValue(attrIndex).equals(attribute.value(i))) {
					SvInstances.add(instance);
				}
			}
			// calculate the correct impurity
			if (giniImpurity) {
				sumValues += (SvInstances.numInstances() / data.numInstances()) * calcGini(generateProb(SvInstances));
			} else {
				sumValues += (SvInstances.numInstances() / data.numInstances()) * calcEntropy(generateProb(SvInstances));
			}
		}
		return ((giniImpurity) ? calcGini(S) - sumValues : calcEntropy(S) - sumValues);
	}

	// return an array of probabilities from the data, split by class attribute
	public double[] generateProb(Instances data) {
		// create an array in the length of the number of values the class index has (like 'yes', 'no' -> 2)
		AttributeStats stats = data.attributeStats(data.classIndex());
		// nominalWeights returns an array with the number of instances for each value of the attribute
		double[] prob = stats.nominalWeights;
		for (int i=0; i< prob.length; i++) {
			prob[i] = prob[i] / data.numInstances();
		}
		return prob;
	}


		// Calculates the Entropy of a random variable.
	// Input: A set of probabilities (the fraction of each possible value in the tested set).
	// Output: The Entropy (double).
	public double calcEntropy(double[] prob) {
		double retEntropy = 0.0;
		for (int i=0; i< prob.length; i++){
			retEntropy =- prob[i] * (Math.log(prob[i])/Math.log(2));
		}
		return retEntropy;
	}

	// Calculates the Gini of a random variable.
	// Input: A set of probabilities (the fraction of each possible value in the tested set).
	// Output: The Gini (double).
	public double calcGini(double[] prob) {
		double retGini = 0.0;
		for (int i=0; i< prob.length; i++){
			retGini =+ prob[i] * prob[i];
		}
		return 1.0 - retGini;
	}

	// Calculates the chi square statistic of splitting the data according to the splitting attribute as learned in class.
	// Input: Instances object (a subset of the training data), attribute index (int).
	// Output: The chi square score (double).
	public double calcChiSquare(double[] prob) {
		double retChiSquare = 0.0;

		return retChiSquare;
	}

	// Measure the impurity before, Measure the impurity after,
	// Choose the attribute that provides the largest difference!
	public int getDecisionAttr(Instances data) {
		int retAttr = 0;
		double bestGain = 0;
		// itterate over the attribute
		for (int i=0; i < data.numAttributes() - 1; i++){
			double gain = calcGain(data, i);
			if (gain > bestGain) {
				// the gain for this attribute gives bigger change in impurity
				bestGain = gain;
				retAttr = i;
			}
		}
		return retAttr;
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}

}
