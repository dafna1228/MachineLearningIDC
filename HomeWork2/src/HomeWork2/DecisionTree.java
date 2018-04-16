package HomeWork2;

import weka.classifiers.Classifier;
import weka.core.*;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.ListIterator;
import java.util.Queue;

class Node {
	Node[] children;
	Node parent;
	// the index if the decision attribute (like 'age', 'breast')
	int attributeIndex;
	// the index of a spesific value of the decision attribute (like 'no' or 'premeno')
	//int attributeValueIndex;
	double returnValue;
	Instances data;
	int height;
}

public class DecisionTree implements Classifier {
	private Node rootNode = new Node();
	private boolean giniImpurity = false;
	private LinkedList<Integer> classificationHeights;
	private int pValueIndex = 0;
	double[][] chiSquareTable ={
			{0, 0, 0, 0, 0, 0},
			{0, 0.102, 0.455, 1.323, 3.841, 7.879},
			{0, 0.575, 1.386, 2.773, 5.991, 10.597},
			{0, 1.213, 2.366, 4.108, 7.815, 12.838},
			{0, 1.923, 3.357, 5.385, 9.488, 14.860},
			{0, 2.675, 4.351, 6.626, 11.070, 16.750},
			{0, 3.455, 5.348, 7.841, 12.592, 18.548},
			{0, 4.255, 6.346, 9.037, 14.067, 20.278},
			{0, 5.071, 7.344, 10.219, 15.507, 21.955},
			{0, 5.899, 8.343, 11.389, 16.919, 23.589},
			{0, 6.737, 9.342, 12.549, 18.307, 25.188},
			{0, 7.584, 10.341, 13.701, 19.675, 26.757}
	};

	public void setGiniImpurity(boolean isGini) {
		giniImpurity = isGini;
	}

	// We should probably use indices instead of values (index 0 is p-value 1, index 1 is p-value 0.75 and so on).
	public void setpValueIndex(int value) {
		pValueIndex = value;
	}
	// Builds a decision tree from the training data. buildClassifier is separated from buildTree in order
	// to allow you to do extra preprocessing before calling buildTree method or post processing after.
	// Input: Instances object.
	@Override
	public void buildClassifier(Instances data) throws Exception {
		buildTree(data, rootNode);
	}

	// Builds the decision tree on given data set using either a recursive or queue algorithm.
	// Input: Instances object (probably the training data set or subset in a recursive method).
	public void buildTree(Instances data, Node currNode) throws Exception {
		currNode.returnValue = getReturnValue(data);
		if (dataHasSameValForAllAttr(data) || calcClassVal1Ratio(data) % 1.0 == 0.0 ||calcEntropy(generateProb(data)) == 0.0) {
			currNode.children = null;
			return;
		} else {
			currNode.attributeIndex = getDecisionAttr(data);
			Attribute decisionAttr = data.attribute(currNode.attributeIndex);
			Node[] children = new Node[decisionAttr.numValues()];
			currNode.children = children;
			for (int i = 0; i < children.length; i++){
				Instances childData = getData(data, currNode.attributeIndex, i);
				if (childData.numInstances() > 0){
					Node child = new Node();
					child.parent = currNode;
					currNode.children[i] = child;
					buildTree(childData, child);

				}
			}
		}
//		Queue<Node> queue = new LinkedList<Node>();
//		// initialize rootNode
//		rootNode.data = data;
//		rootNode.returnValue = getReturnValue(rootNode.data);
//		rootNode.height = 0;
//		queue.add(rootNode);
//		while ( !queue.isEmpty()){
//			//System.out.println("q size: " + queue.size());
//			Node currNode = queue.remove();
//			// If training examples in n are not perfectly classified-
//			//for (int i = 0; i < currNode.data.numInstances(); i++) {
//			//System.out.println(currNode.data.instance(i));
//			//}
//			// check stopping condition- If training examples in n are not perfectly classified,
//			// or all the instances in data have the same value fot all attributes
//			if ((calcClassVal1Ratio(currNode.data) % 1.0 != 0.0) && (!dataHasSameValForAllAttr(currNode.data)) ) {
//				// Assign the decision attribute for the node data
//				int decisionAttrIndex = getDecisionAttr(currNode.data);
//				currNode.attributeIndex = decisionAttrIndex;
//				currNode.children = new Node[0];
//				boolean prune = false;
//				// If the p_value is not 1 (stored in index 0)
//				if (pValueIndex > 0) {
//					prune = shouldPrune(currNode.data, currNode.attributeIndex, pValueIndex);
//				}
//				// if prune is true, don't create children
//				if (!prune) {
//					Attribute decisionAttr = currNode.data.attribute(decisionAttrIndex);
//					int attrNumVal = decisionAttr.numValues();
//					Node[] currChildren = new Node[attrNumVal];
//					// For each value of decisionAttr, create a new descendant of the node
//					for (int i = 0; i < attrNumVal; i++) {
//						Node childNode = new Node();
//						childNode.parent = currNode;
//						// childNode.attributeIndex = decisionAttr;
//						childNode.data = new Instances(data, 0);
//						String attributeValue = decisionAttr.value(i);
//						childNode.attributeValueIndex = decisionAttr.indexOfValue(attributeValue);
//						childNode.height = currNode.height + 1;
//						childNode.children = new Node[0];
//						currChildren[i] = childNode;
//					}
//					// Distribute training examples to descendant nodes
//					for (int i = 0; i < currNode.data.numInstances(); i++) {
//						Instance instance = currNode.data.instance(i);
//						String value = instance.stringValue(decisionAttr);
//						for (int j = 0; j < attrNumVal; j++) {
//							// if the index of this instance's decisionAttr value is equal to the attributeValueIndex of this child
//							if (currChildren[j].attributeValueIndex == decisionAttr.indexOfValue(value)) {
//								currChildren[j].data.add(instance);
//							}
//						}
//					}
//					// Insert all (non empty) descendant nodes to Q
//					for (int i = 0; i < attrNumVal; i++) {
//						if (currChildren[i].data.numInstances() > 0) {
//							currChildren[i].returnValue = getReturnValue(currChildren[i].data);
//							queue.add(currChildren[i]);
//						}
//					}
//					currNode.children = currChildren;
//				}
//			}
//		}
	}

	// Return the classification of the instance.
	// Input: Instance object.
	// Output: double number, 0 or 1, represent the classified class
	@Override
	public double classifyInstance(Instance instance) {
		Node currNode = rootNode;
		// walk down the tree
		while (currNode.children != null ){
			// set currNode as the child of currNode that has the attribute value same as the instance
			int valueOfInstance = (int) instance.value(currNode.attributeIndex);
				if(currNode.children[valueOfInstance] == null){
					return currNode.returnValue;
				}else{
					currNode = currNode.children[valueOfInstance];
				}
			classificationHeights.add(currNode.height);

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
		classificationHeights = new LinkedList<Integer>();
		for (int i=0; i < data.numInstances(); i++) {
			double prediction = classifyInstance(data.instance(i));
			// if the prediction is different from the real class value
			if (prediction != data.instance(i).classValue()){
				mistakes++;
			}
		}
		return (double) mistakes / (double) data.numInstances();
	}

	// Calculates the chi square statistic of splitting the data according to the splitting attribute as learned in class.
	// Input: Instances object (a subset of the training data), attribute index (int).
	// Output: The chi square score (double).
	protected double calcChiSquare(Instances data, int attrIndex){
		double chiSquareScore = 0;
		double[] probs = generateProb(data);
		double E0, E1 = 0;
		int Df, pf, nf = 0;
		Instances[] instancesByValuesOfAttr = getInstancesWithValue(data, attrIndex);
		for (int i = 0; i < instancesByValuesOfAttr.length; i++) {
			Df = instancesByValuesOfAttr[i].numInstances();
			pf = 0;
			nf = 0;
			for (int j = 0; j < Df; j++) {
				if (instancesByValuesOfAttr[i].instance(j).classValue() == 0){
					// Counting the number of instances that have the value i in the attribute and the class value is 0 (recurrence)
					pf++;
				}
				else if (instancesByValuesOfAttr[i].instance(j).classValue() == 1){
					// Counting the number of instances that have the value i in the attribute and the class value is 1 (no recurrence)
					nf++;
				}
			}
			E0 = Df * probs[0];
			E1 = Df * probs[1];
			// Making sure we're not dividing by zero
			if ((E0 != 0) && (E1 != 0)){
				chiSquareScore += Math.pow((pf - E0), 2) / E0 + Math.pow((nf - E1), 2) / E1;
			}
		}
		return chiSquareScore;
	}

	private Instances[] getInstancesWithValue(Instances data, int attrIndex) {
		int valueIndex;
		// Each cell of the array contains all the instances that have the attribute's value with the index of the cell
		Instances[] instancesWithValue = new Instances[data.attribute(attrIndex).numValues()];
		// Initializing the Instances array
		for (int i = 0; i < data.attribute(attrIndex).numValues(); i++) {
			instancesWithValue[i] = new Instances(data, data.numInstances());
		}
		// Adding each instance in data to instancesWithValue in the index it's value of attribute with index attrIndex
		for (int j = 0; j < data.numInstances(); j++) {
			valueIndex = (int) data.instance(j).value(attrIndex);
			instancesWithValue[valueIndex].add(data.instance(j));
		}
		return instancesWithValue;
	}

	private Instances getData(Instances data, int attrIndex, int valueIndex){
		int instanceValueIndex = 0;
		// Each cell of the array contains all the instances that have the attribute's value with the index of the cell
		Instances instancesWithValue = new Instances(data,0);
		// Initializing the Instances array

		// Adding each instance in data to instancesWithValue in the index it's value of attribute with index attrIndex
		for (int j = 0; j < data.numInstances(); j++) {

			instanceValueIndex = (int) data.instance(j).value(attrIndex);
			if (valueIndex == instanceValueIndex){
				instancesWithValue.add(data.instance(j));
			}
		}
		return instancesWithValue;
	}

	protected int getFreedomDegree(Instances data, int attrIndex){
		int[] relevantValues = new int[data.attribute(attrIndex).numValues()];
		int numValues = 0;
		// Calculating the number of relevant attribute values
		for (int i = 0; i < data.numInstances(); i++) {
			for (int j = 0; j < data.attribute(attrIndex).numValues(); j++) {
				if (data.instance(i).stringValue(attrIndex) == data.attribute(attrIndex).value(j)){
					relevantValues[j]++;
				}
			}
		}
		for (int k = 0; k < relevantValues.length; k++) {
			if (relevantValues[k] > 0){
				numValues++;
			}
		}
		// The degree of freedom is the number of relevant values minus 1
		return numValues - 1;
	}

	protected boolean shouldPrune(Instances data, int attrIndex, int pValueIndex){
		int freedomDeg = getFreedomDegree(data, attrIndex);
		double chiSquareStat = calcChiSquare(data, attrIndex);
		// If the Chi square statistic is smaller than the number from the table we should prune.
		if (chiSquareStat < chiSquareTable[freedomDeg][pValueIndex]){
			return true;
		} else {
			return false;
		}
	}

	protected int getMaxHeight(){
		ListIterator <Integer> iterator = classificationHeights.listIterator();
		int maxHeight = 0;
		int currHeight;
		while (iterator.hasNext()){
			currHeight = iterator.next();
			if (currHeight > maxHeight){
				maxHeight = currHeight;
			}
		}
		return maxHeight;
	}

	protected int getAvgHeight(){
		ListIterator <Integer> iterator = classificationHeights.listIterator();
		int sumHeights = 0;
		while (iterator.hasNext()){
			sumHeights += iterator.next();
		}
		return sumHeights / classificationHeights.size();
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
				double SvdivS = ((double) SvInstances.numInstances()) / ( (double) data.numInstances());
				if (giniImpurity) {
					sumValues += SvdivS * calcGini(Svprob);
				} else {
					sumValues += SvdivS * calcEntropy(Svprob);
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
				retEntropy -= prob[i] * (Math.log(prob[i]) / Math.log(2));
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

	// Measure the impurity before, Measure the impurity after,
	// Choose the attribute that provides the largest difference!
	public int getDecisionAttr(Instances data) {
		int retAttr = 0;
		double bestGain = 0.0;
		// itterate over the attribute
		for (int i=0; i < data.numAttributes() - 1; i++){
			double gain = calcGain(data, i);
			//System.out.println("gain: "+ gain);
			if (gain > bestGain) {
				// the gain for this attribute gives bigger change in impurity
				bestGain = gain;
				retAttr = i;
			}
		}

		return retAttr;
	}

//	void printTree() {
//		recursivePreorderTree(rootNode);
//	}

	// print the tree with recursive preorder algorithm
//	void recursivePreorderTree(Node node)
//	{
//		// first print data of node
//		printNode(node);
//
//		if (node.children.length == 0) {
//			return;
//		}
//		// then recur on left sutree
//		recursivePreorderTree(node.children[0]);
//		// now recur on right subtree
//		recursivePreorderTree(node.children[node.children.length - 1]);
//	}

//	void printNode(Node node)
//	{
//		// create the tab indentation- 4*node.height spaces
//		String spaces = ((node.height == 0)? "" : String.format("%"+(node.height*4)+"s", ""));
//		//String spaces = String.format("%1$#"+(node.height*4)+"s", "");
//		if (node.height == 0) {
//			System.out.println("Root");
//		} else {
//			System.out.println(spaces + "if attribute " + node.parent.attributeIndex + " = " + node.attributeValueIndex);
//		}
//		if (node.children.length == 0) {
//			System.out.println(spaces + "leaf. Returning value: " + node.returnValue);
//		} else {
//			System.out.println(spaces + "Returning value: " + node.returnValue);
//		}
//
//	}

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
