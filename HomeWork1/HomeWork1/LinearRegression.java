//package HomeWork1;

import weka.classifiers.Classifier;
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
    @Override
    public void buildClassifier(Instances trainingData) throws Exception {
        m_ClassIndex = trainingData.classIndex();
        m_truNumAttributes = trainingData.numAttributes() - 1;
        findAlpha(trainingData);
        m_coefficients = new double[m_truNumAttributes + 1];
        double current_error, previous_error = Double.MAX_VALUE;
        current_error = calculateMSE(trainingData);
        while (Math.abs(current_error - previous_error) > 0.003) {
            for (int i = 0; i < 100; i++) {
                m_coefficients = gradientDescent(trainingData);
            }
            System.out.println(current_error);
            previous_error = current_error;
            current_error = calculateMSE(trainingData);
        }
        //TODO: complete this method
        //m_coefficients = gradientDescent(trainingData);
    }

    public void findAlpha(Instances data) throws Exception {
        double lowest_error = Double.MAX_VALUE, best_alpha = 0;
        double current_error, previous_error;
        int eggs = 0;
        for (int i = -17; i <= 0; i++) {
            m_alpha = Math.pow(3, i);
            m_coefficients = new double[m_truNumAttributes + 1];
            current_error = Double.MAX_VALUE;
            for (int j = 0; j < 200; j++) {
                for (int k = 0; k < 100; k++) {
                    m_coefficients = gradientDescent(data);
                }
                previous_error = current_error;
                current_error = calculateMSE(data);
                if (previous_error < current_error) {
                    current_error = previous_error;
                    break;
                }
            }
            if (lowest_error > current_error) {
                eggs = i;
                best_alpha = m_alpha;
                lowest_error = current_error;
            }
        }
        m_alpha = best_alpha;
        System.out.println("best alpha:");
        System.out.println(m_alpha);
    }

    /**
     * An implementation of the gradient descent algorithm which should
     * return the weights of a linear regression predictor which minimizes
     * the average squared error.
     *
     * @param trainingData
     * @throws Exception
     */
    private double[] gradientDescent(Instances trainingData)
            throws Exception {
        Instance current_instance;
        double[] newCoeff = new double[m_truNumAttributes + 1];
        for (int i = 1; i <= m_truNumAttributes; i++) {
            newCoeff[i] = m_coefficients[i];
            for (int j = 0; j < trainingData.size(); j++) {
                current_instance = trainingData.get(j);
                newCoeff[i] -= m_alpha * current_instance.value(i - 1) *
                        (innerProduct(current_instance) - current_instance.classValue());
            }
        }
        newCoeff[0] = m_coefficients[0];
        for (int i = 0; i < trainingData.size(); i++) {
            current_instance = trainingData.get(i);
            newCoeff[0] -= (1.0/trainingData.size()) * m_alpha * (innerProduct(current_instance) - current_instance.classValue());
        }
        return newCoeff;

    }

    private double innerProduct(Instance instance) {
        double result = 0;
        for (int i = 0; i < m_truNumAttributes; i++) {
            result += m_coefficients[i + 1] * instance.value(i);
        }
        result += m_coefficients[0];
        return result;
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
        double result = 0;
        for (int i = 0; i < m_truNumAttributes; i++) {
            result += m_coefficients[i] * instance.value(i);
        }
        return result;
    }

    /**
     * Calculates the total squared error over the data on a linear regression
     * predictor with weights given by m_coefficients.
     *
     * @param testData
     * @return
     * @throws Exception
     */
    public double calculateMSE(Instances testData) throws Exception {
        double error = 0;
        Instance currentInstance;
        for (int i = 0; i < testData.size(); i++) {
            currentInstance = testData.instance(i);
            error += Math.pow(innerProduct(currentInstance) - currentInstance.classValue(), 2);
        }
        return (0.5/testData.size()) * error;
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