package collinm.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.javatuples.Triplet;

import com.google.common.base.Joiner;

public class ConfusionMatrix {

	private int numClasses;
	private Map<String, Double> classNameMap;
	private ArrayList<ArrayList<Double>> matrix;

	public ConfusionMatrix(String[] classNames) {
		this.numClasses = classNames.length;
		this.classNameMap = new HashMap<>();
		double counter = 0;
		for (String name : classNames) {
			this.classNameMap.put(name, counter++);
		}
		this.matrix = new ArrayList<>(numClasses);
		for (int i = 0; i < numClasses; i++)
			this.matrix.add(new ArrayList<Double>(Collections.nCopies(numClasses, 0.0)));
	}
	
	public void increment(String actual, String predicted) {
		this.increment(this.classNameMap.get(actual), this.classNameMap.get(predicted));
	}

	public void increment(double actual, double predicted) {
		List<Double> x = this.matrix.get((int) actual);
		x.set((int) predicted, x.get((int) predicted) + 1);
	}

	/**
	 * 
	 * @param triplets
	 *            list of triplets where 0 = ID, 1 = actual, 2 = prediction
	 */
	public void measure(List<Triplet<String, Double, Double>> triplets) {
		for (Triplet<String, Double, Double> t : triplets)
			this.increment(t.getValue1(), t.getValue2());
	}
	
	/**
	 * @param classLabel
	 * @return precision for a single class/label
	 */
	public double precision(double classLabel) {
		int label = (int) classLabel;
		double tp = this.matrix.get(label).get(label);
		double fp = tp * -1 + IntStream.range(0, this.numClasses)
				.mapToDouble(i -> this.matrix.get(i).get(label))
				.reduce(Double::sum)
				.getAsDouble();
		return (tp + fp == 0) ? 0 : tp / (tp + fp);
	}
	
	/**
	 * @return instance-weighted precision for all classes
	 */
	public double precision() {
		return IntStream.range(0, this.numClasses)
				.mapToDouble(label -> this.precision(label) * this.classRatio(label))
				.reduce(Double::sum)
				.getAsDouble();
	}
	
	/**
	 * @param classLabel
	 * @return recall for a single class/label
	 */
	public double recall(double classLabel) {
		int label = (int) classLabel;
		double tp = this.matrix.get(label).get(label);
		double fn = tp * -1 + this.matrix.get(label).stream().reduce(Double::sum).get();
		return tp / (tp + fn);
	}

	/**
	 * @return instance-weighted recall for all classes
	 */
	public double recall() {
		return IntStream.range(0, this.numClasses)
				.mapToDouble(label -> this.recall(label) * this.classRatio(label))
				.filter(d -> !Double.isNaN(d))
				.reduce(Double::sum)
				.getAsDouble();
	}
	
	/**
	 * @param classLabel
	 * @return instance-weighted F1 for all classes
	 */
	public double f1(double classLabel) {
		double p = this.precision(classLabel);
		double r = this.recall(classLabel);
		return 2 * ((p * r) / (p + r));
	}

	/**
	 * @return instance-weighted F1 for all classes
	 */
	public double f1() {
		double p = this.precision();
		double r = this.recall();
		return 2 * ((p * r) / (p + r));
	}

	/**
	 * @param classLabel
	 * @return Proportion of the matrix that belongs to the specified class
	 */
	private double classRatio(int classLabel) {
		double classInstanceCount = this.matrix.get(classLabel).stream().reduce(Double::sum).get();
		double all = IntStream.range(0, this.numClasses)
				.mapToDouble(i -> this.matrix.get(i).stream()
						.reduce(Double::sum).get())
				.reduce(Double::sum).getAsDouble();
		return classInstanceCount / all;
	}

	/**
	 * @return <code>precision,recall,F1</code>
	 */
	public String toMetricsCSV() {
		return Joiner.on(",").join(this.precision(), this.recall(), this.f1());
	}

	public static String toMetricsCSV(List<ConfusionMatrix> cms) {
		StringBuilder sb = new StringBuilder();
		sb.append(Joiner.on(",").join("Split #", "Precision", "Recall", "F1") + "\n");
		for (int i = 0; i < cms.size(); i++)
			sb.append(Joiner.on(",").join(i, cms.get(i).toMetricsCSV()) + "\n");
		return sb.toString();
	}
	
	public String toCSV() {
		StringBuilder out = new StringBuilder();
		Joiner commaJoiner = Joiner.on(",");

		// Writer header
		out.append(",");
		out.append(commaJoiner
				.join(IntStream.range(0, this.numClasses)
						.mapToObj(Integer::toString)
						.collect(Collectors.toList())));
		out.append("\n");

		// Write each row class + data
		for (int i = 0; i < this.numClasses; i++) {
			out.append(Integer.toString(i) + ",");
			out.append(commaJoiner.join(this.matrix.get(i)));
			out.append("\n");
		}

		return out.toString();
	}
}
