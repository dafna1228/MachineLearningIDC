// package HomeWork1;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegressionbad implements Classifier {
	
    private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha;

	//the method which runs to train the linear regression predictor, i.e.
	//finds its weights.
	//The learning phase start and end in this method
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		initCoefficients(trainingData.numAttributes());
		System.out.println("start buildClassifier with " + trainingData.numAttributes());
		//m_coefficients = new double[trainingData.numAttributes()];
		m_ClassIndex = trainingData.numAttributes() - 1;
		trainingData.setClassIndex(m_ClassIndex);
		//TODO: complete this method.
		boolean running = true;
		int iter = 0;
		double newErr = 0; double prevErr = 0;
		while (running){
			m_coefficients = gradientDescent(trainingData);
			iter++;
			if (iter % 100 == 0){
				// calculate difference between old and new error
				newErr = calculateMSE(trainingData);
				//System.out.println("itter: " + iter + ", error diff: " + (prevErr - newErr));
				if (Math.abs(prevErr - newErr) <= 0.003){
					running = false;
				}
				prevErr = newErr;

			}
		}
		System.out.println("end buildClassifier");
	}

	void findAlpha(Instances data) throws Exception {
		double bestErr = Double.MAX_VALUE;
		double newErr;
		double prevErr;
		double bestAlpha = 0;
		m_ClassIndex = data.numAttributes() - 1;
		for (double i = -17; i <= 0; i++){
			initCoefficients(data.numAttributes());
			m_alpha = Math.pow(3, i);
			System.out.println("start alpha " + m_alpha);
			newErr = Double.MAX_VALUE;
			prevErr = Double.MAX_VALUE;
			int iter = 0;
			while (iter < 20000 && (newErr <= prevErr)) {
				m_coefficients = gradientDescent(data);
				// Update the error every 100 iterations

				if (iter % 100 == 99){
					prevErr = newErr;
					newErr = calculateMSE(data);
					//System.out.println("New Err " + newErr + ", prev Err " + prevErr + ", itter " + iter);
				}
				iter++;
			}
			if(prevErr < newErr)
				newErr = prevErr;

			System.out.println("Current error is " + newErr);
			if(newErr < bestErr){
				System.out.println("Setting it as smallest error");
				bestErr = newErr;
				bestAlpha = m_alpha;
			}
		}
		m_alpha = bestAlpha;
		System.out.println("The best alpha is " + m_alpha);
	}

	protected void initCoefficients(int size){
		m_coefficients = new double[size];
		for (int k = 0; k < m_coefficients.length; k++) {
			m_coefficients[k] = 0;
		}
	}

	/**
	 * An implementation of the gradient descent algorithm which should
	 * return the weights of a linear regression predictor which minimizes
	 * the average squared error.
     * 
	 * @param trainingData
	 * @throws Exception
	 */
	private double[] gradientDescent(Instances trainingData) throws Exception {
		double[] retCoeff = new double[m_coefficients.length];
		//iterate and update the coefficients
		// update theta 0
		double coeff = m_coefficients[0];
		// build the sum of errors
		double sumErr = 0.0;
		for (int j = 0; j < trainingData.numInstances(); j++) {
			Instance instance = trainingData.instance(j);
			// System.out.println("classindex " + m_ClassIndex);
			sumErr += innerProduct(instance) - instance.value(m_ClassIndex);
		}
		retCoeff[0] = coeff - m_alpha*((1.0/trainingData.numInstances())*sumErr);
		// update theta 1 and more
		for (int i = 1; i < m_coefficients.length; i++){
			coeff = m_coefficients[i];
			// build the sum of errors
			sumErr = 0.0;
			for (int j = 0; j < trainingData.numInstances(); j++) {
				Instance instance = trainingData.instance(j);
				sumErr += (innerProduct(instance) - instance.value(m_ClassIndex)) * instance.value(i);
			}


			retCoeff[i] = coeff - m_alpha*((1.0/trainingData.numInstances())*sumErr);
			//System.out.println(retCoeff[i]);
		}
		//System.out.println(retCoeff[0] + " " + retCoeff[1]);
		return retCoeff;
	}

	/**
	 * Returns the prediction of a linear regression predictor with weights
	 * given by m_coefficients on a single instance.
     *
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public double regressionPrediction(Instance instance) throws Exception {
		double innerProduct = 0;
		for (int i = 1; i < instance.numAttributes() - 1; i++) {
			innerProduct += instance.value(i - 1) * m_coefficients[i];
		}
		return innerProduct + m_coefficients[0];
	}

	/**
	 * Calculates the total squared error over the data on a linear regression
	 * predictor with weights given by m_coefficients.
     *
	 * @param data
	 * @return
	 * @throws Exception
	 */
	public double calculateMSE(Instances data) throws Exception {
		double err = 0;
		for (int i = 0; i < data.numInstances(); i++){
			Instance instance = data.instance(i);
			//System.out.println(instance.classValue() + " " + instance.value(m_ClassIndex));
			err += Math.pow(regressionPrediction(instance) - instance.classValue(), 2);
		}
		return err / (2 * data.numInstances());
	}


	// Might be redundant, it's already written in regressionPrediction
	public double innerProduct(Instance instance) throws Exception {
		double innerProduct = 0;
		for (int i = 1; i < instance.numAttributes() - 1; i++) {
			innerProduct += instance.value(i - 1) * m_coefficients[i];
		}
		return innerProduct + m_coefficients[0];
	}
	
	public double getAlpha(){
		return m_alpha;
	}

    @Override
	public double classifyInstance(Instance arg0) throws Exception {
		// Don't change
		return 0;
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
