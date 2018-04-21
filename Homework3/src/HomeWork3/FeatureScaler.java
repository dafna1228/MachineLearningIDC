package HomeWork3;

import weka.core.Instance;
import weka.core.Instances;
import java.lang.Object;
import weka.filters.unsupervised.attribute.PotentialClassIgnorer;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.Filter;

public class FeatureScaler {
	/**
	 * Returns a scaled version (using standarized normalization) of the given dataset.
	 * @param instances The original dataset.
	 * @return A scaled instances object.
	 */
	public Instances scaleData(Instances instances) throws Exception {

		Standardize filter = new Standardize();
		filter.setInputFormat(instances);
		Instances standardizeData = Filter.useFilter(instances, filter);
		return standardizeData;
	}
}