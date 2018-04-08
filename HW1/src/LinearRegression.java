package HomeWork1;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegression implements Classifier {

	private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha;

	//the method which runs to train the linear regression predictor, i.e.
	//finds its weights.
	//The learning phase start and end in this method
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		m_ClassIndex = trainingData.classIndex();
		// The last attribute is the class value
		m_truNumAttributes = trainingData.numAttributes() - 1;
		// We find alpha only once
		if (m_alpha == 0){
			findAlpha(trainingData);
		}
		initCoefficients(m_truNumAttributes + 1);
		int iter = 0;
		double newErr = Double.MAX_VALUE;
		double prevErr = Double.MAX_VALUE;
		// Calculating the error with the initialized coefficients
		newErr = calculateMSE(trainingData);
		// Performing gradient descent while the difference in errors is not smaller than 0.003
		while (Math.abs(prevErr - newErr) >= 0.003){
			m_coefficients = gradientDescent(trainingData);
			if (iter % 100 == 0){
				prevErr = newErr;
				// Calculating a new error every 100 iterations
				newErr = calculateMSE(trainingData);
			}
			iter++;
		}
	}

	void findAlpha(Instances data) throws Exception {
		m_ClassIndex = data.classIndex();
		initCoefficients(m_truNumAttributes + 1);
		double bestErr = Double.MAX_VALUE;
		double newErr;
		double prevErr;
		double bestAlpha = 0;
		for (int i = -17; i <= 0; i++){
			// Initializing the coefficients for every alpha
			initCoefficients(m_truNumAttributes + 1);
			m_alpha = Math.pow(3, i);
			prevErr = Double.MAX_VALUE;
			newErr = calculateMSE(data);
			int iter = 0;
			// Comparing errors until we've reached 20k iterations or a new error is bigger than the one 100 iterations before it.
			while (iter <= 20000 && (newErr < prevErr)) {
				m_coefficients = gradientDescent(data);
				if (iter % 100 == 0){
					prevErr = newErr;
					newErr = calculateMSE(data);
				}
				iter++;
			}
			// If the previous error is better than the new error and the best alpha's error so far, update the best error and the final alpha
			if (prevErr < newErr && prevErr < bestErr){
				bestErr = prevErr;
				bestAlpha = m_alpha;
				// If the new error is better than the new error and the best alpha's error so far, update the best error and the final alpha
			} else if (newErr < prevErr && newErr < bestErr){
				bestErr = newErr;
				bestAlpha = m_alpha;
			}
		}
		m_alpha = bestAlpha;
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
		double[] retCoeff = new double[m_truNumAttributes + 1];
		Instance instance;
		// Starting with updating theta 0
		double coeff = m_coefficients[0];
		double sumErr = 0.0;
		// Summing the errors
		for (int j = 0; j < trainingData.numInstances(); j++) {
			instance = trainingData.instance(j);
			sumErr += regressionPrediction(instance) - instance.value(m_ClassIndex);
		}
		retCoeff[0] = coeff - m_alpha*((1.0/trainingData.numInstances())*sumErr);
		// Updating all other thetas
		for (int i = 1; i <= m_truNumAttributes; i++){
			coeff = m_coefficients[i];
			// Summing the errors
			sumErr = 0.0;
			for (int j = 0; j < trainingData.numInstances(); j++) {
				instance = trainingData.instance(j);
				sumErr += (regressionPrediction(instance) - instance.value(m_ClassIndex)) * instance.value(i - 1); // changed to i - 1
			}
			retCoeff[i] = coeff - m_alpha*((1.0/trainingData.numInstances())*sumErr);
		}
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
		double res = 0;
		// Performing the inner product as learned in the recitation
		for (int i = 1; i <= m_truNumAttributes; i++) {
			res += instance.value(i - 1) * m_coefficients[i];
		}
		return res + m_coefficients[0];
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
		Instance instance;
		for (int i = 0; i < data.numInstances(); i++){
			instance = data.instance(i);
			err += Math.pow((regressionPrediction(instance) - instance.classValue()), 2); // using innerprod instead of regpred
		}
		return (err / (2 * data.numInstances()));
	}


	//public double innerProduct(Instance instance) throws Exception {

	//}

	protected void initCoefficients(int size){
		m_coefficients = new double[size];
		for (int k = 0; k < m_coefficients.length; k++) {
			m_coefficients[k] = 1.0;
		}
	}

	protected double getAlpha(){
		return m_alpha;
	}

	protected void setAlpha(double alpha){
		m_alpha = alpha;
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
