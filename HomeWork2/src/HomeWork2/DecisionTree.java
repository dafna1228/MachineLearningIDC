package HomeWork2;

import weka.classifiers.Classifier;
import weka.core.*;

import java.util.LinkedList;
import java.util.Queue;

class Node {
	Node[] children;
	Node parent;
	int attributeIndex;
	int attributeValueIndex;
	double returnValue;
	Instances data;
	int height;
}

public class DecisionTree implements Classifier {
	private Node rootNode = new Node();
	private boolean giniImpurity = false;

	public void setGiniImpurity(boolean isGini) {
		giniImpurity = isGini;
	}

	// Builds a decision tree from the training data. buildClassifier is separated from buildTree in order
	// to allow you to do extra preprocessing before calling buildTree method or post processing after.
	// Input: Instances object.
	@Override
	public void buildClassifier(Instances data) throws Exception {
		buildTree(data);
	}

	// Builds the decision tree on given data set using either a recursive or queue algorithm.
	// Input: Instances object (probably the training data set or subset in a recursive method).
	public void buildTree(Instances data) throws Exception {
		Queue<Node> queue = new LinkedList<Node>();
		// initialize rootNode
		rootNode.data = data;
		rootNode.returnValue = getReturnValue(rootNode.data);
		rootNode.height = 0;
		queue.add(rootNode);
		int count = 0;
		while ( !queue.isEmpty()){
			//System.out.println("q size: " + queue.size());
			Node currNode = queue.remove();
			// If training examples in n are not perfectly classified-
//			for (int i = 0; i < currNode.data.numInstances(); i++) {
//				System.out.println(currNode.data.instance(i));
//			}
			// check stopping condition- If training examples in n are not perfectly classified,
			// or all the instances in data have the same value fot all attributes
			if ((calcClassVal1Ratio(currNode.data) % 1.0 != 0.0) && (!dataHasSameValForAllAttr(currNode.data)) ) {
				// Assign the decision attribute for the node data
				int decisionAttrIndex = getDecisionAttr(currNode.data);
				currNode.attributeIndex = decisionAttrIndex;
				Attribute decisionAttr = currNode.data.attribute(decisionAttrIndex);
				//System.out.println("decision attr: " + decisionAttr);
				int attrNumVal = decisionAttr.numValues();
				Node[] currChildren = new Node[attrNumVal];
				// For each value of decisionAttr, create a new descendant of the node
				for (int i = 0; i < attrNumVal; i++){
					Node childNode = new Node();
					childNode.parent = currNode;
					// childNode.attributeIndex = decisionAttr;
					childNode.data = new Instances(data,0);
					String attributeValue = decisionAttr.value(i);
					childNode.attributeValueIndex = decisionAttr.indexOfValue(attributeValue);
					childNode.height = currNode.height + 1;
					childNode.children = new Node[0];
					currChildren[i] = childNode;
				}
				// Distribute training examples to descendant nodes
				for (int i = 0; i < currNode.data.numInstances(); i++){
					Instance instance = currNode.data.instance(i);
					String value = instance.stringValue(decisionAttr);
					for (int j = 0; j < attrNumVal; j++){
						// if the index of this instance's decisionAttr value is equal to the attributeValueIndex of this child
						if (currChildren[j].attributeValueIndex == decisionAttr.indexOfValue(value)){
							currChildren[j].data.add(instance);
						}
					}
				}
				int insert = 0;
				// Insert all (non empty) descendant nodes to Q
				for (int i = 0; i < attrNumVal; i++){
					if (currChildren[i].data.numInstances() > 0) {
						currChildren[i].returnValue = getReturnValue(currChildren[i].data);
						queue.add(currChildren[i]);
						insert++;
					}
				}
				currNode.children = currChildren;
			}
		}
	}

	// Return the classification of the instance.
	// Input: Instance object.
	// Output: double number, 0 or 1, represent the classified class
	@Override
	public double classifyInstance(Instance instance) {
		Node currNode = rootNode;
		// walk down the tree
		while (currNode.children.length == 0 ){
			// set currNode as the child of currNode that has the attribute value same as the instance
			for(int i = 0; i < currNode.children.length; i++) {
				String instanceAttrValue = instance.stringValue(currNode.attributeIndex);
				int instanceAttrValueIndex = currNode.data.attribute(currNode.attributeIndex).indexOfValue(instanceAttrValue);
				if (instanceAttrValueIndex == currNode.children[i].attributeValueIndex){
					currNode = currNode.children[i];
					break;
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
		double[] Sprob = generateProb(data);
		Attribute attribute = data.attribute(attrIndex);
		for (int i=0; i < attribute.numValues(); i++){
			// create Instances object with instances that have value i in the attribute
			Instances SvInstances = new Instances(data, 0);
			for (int j = 0; j < data.numInstances(); j++) {
				Instance instance = data.instance(j);
				if (instance.stringValue(attrIndex).equals(attribute.value(i))) {
					SvInstances.add(instance);
				}
			}
			if (SvInstances.numInstances() > 0) {
				// calculate the correct impurity
				double[] Svprob = generateProb(SvInstances);
				double SSv = ((double) SvInstances.numInstances()) / ( (double) data.numInstances());
				if (giniImpurity) {
					sumValues += SSv * calcGini(Svprob);
				} else {
					sumValues += SSv * calcEntropy(Svprob);
				}
			}
		}
		return ((giniImpurity) ? calcGini(Sprob) - sumValues : calcEntropy(Sprob) - sumValues);
	}

	// return an array of probabilities from the data, split by class attribute
	public double[] generateProb(Instances data) {
		double[] prob = new double[data.classAttribute().numValues()];
		for (int i = 0; i < data.numInstances(); i++){
			Instance instance = data.instance(i);
			String value = instance.stringValue(data.classAttribute());
			for (int j = 0; j < prob.length; j++){
				// if the index of this instance's class Attr value is equal to the attributeValueIndex of this child
				if (j == data.classAttribute().indexOfValue(value)){
					prob[j] = prob[j] + 1.0;
				}
			}
		}
		for (int j = 0; j < prob.length; j++){
			prob[j] = prob[j]/ (double) data.numInstances();
		}

		return prob;
	}


	// Calculates the Entropy of a random variable.
	// Input: A set of probabilities (the fraction of each possible value in the tested set).
	// Output: The Entropy (double).
	public double calcEntropy(double[] prob) {
		double retEntropy = 0.0;
		//System.out.println();
		for (int i=0; i< prob.length; i++) {
			if (prob[i] == 0) {
				retEntropy = 0;
			} else
				retEntropy = -prob[i] * (Math.log(prob[i]) / Math.log(2));
		}
		return retEntropy;
	}


	// Calculates the return value of the node
	// Input: the instances of this node
	// Output: 1 or 0 (double).
	public double getReturnValue(Instances data) {
		double classVal1Ratio = calcClassVal1Ratio(data);
		return (( classVal1Ratio > 0.5) ? 1.0 :  0.0);
	}

	// Calculates the num of instances with return value 1 / num of instances
	// Input: the instances of this node
	// Output: 1 or 0 (double).
	public double calcClassVal1Ratio(Instances data) {
		int count1 = 0;
		// count the num of instances that have value 1
		for (int i=0; i< data.numInstances(); i++){
			if (data.instance(i).classValue() == 1.0) {
				count1++;
			}
		}
		return ((double) count1 / (double) data.numInstances());
	}

	// says if all the instances in the data have the same value for all attributes
	// Input: instances
	// Output: true or false
	public boolean dataHasSameValForAllAttr(Instances data) {
		boolean retval = true;
		InstanceComparator compare = new InstanceComparator(false);
		for (int i=0; i < data.numInstances(); i++){
			if (compare.compare(data.instance(i), data.instance(0)) != 0.0){
				return false;
			}
		}
		return retval;
	}
	// Calculates the Gini of a random variable.
	// Input: A set of probabilities (the fraction of each possible value in the tested set).
	// Output: The Gini (double).
	public double calcGini(double[] prob) {
		double retGini = 0.0;
		for (int i=0; i< prob.length; i++){
			retGini += prob[i] * prob[i];
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
		double bestGain = 0.0;
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

	void printTree() {
		recursivePreorderTree(rootNode);
	}

	// print the tree with recursive preorder algorithm
	void recursivePreorderTree(Node node)
	{
		// first print data of node
		printNode(node);

		if (node.children.length == 0) {
			return;
		}
		// then recur on left sutree
		recursivePreorderTree(node.children[0]);
		// now recur on right subtree
		recursivePreorderTree(node.children[node.children.length - 1]);
	}

	void printNode(Node node)
	{
		// create the tab indentation- 4*node.height spaces
		String spaces = ((node.height == 0)? "" : String.format("%"+(node.height*4)+"s", ""));
		//String spaces = String.format("%1$#"+(node.height*4)+"s", "");
		if (node.height == 0) {
			System.out.println("Root");
		} else {
			System.out.println(spaces + "if attribute " + node.parent.attributeIndex + " = " + node.attributeValueIndex);
		}
		if (node.children.length == 0) {
			System.out.println(spaces + "leaf. Returning value: " + node.returnValue);
		} else {
			System.out.println(spaces + "Returning value: " + node.returnValue);
		}

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
