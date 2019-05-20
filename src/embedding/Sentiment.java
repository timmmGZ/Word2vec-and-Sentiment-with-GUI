package embedding;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.TreeSet;
import java.util.stream.Collectors;

public class Sentiment extends Embedding {
	public double testInput[][], testOutput[][];
	int numSenmatic = 5;

	public Sentiment(String train, String test, int dim) throws IOException {
		loadWordsAndInputAndLabel(train, test);
		initializeParameters(dim, numSenmatic);
	}

	public static void main(String[] args) throws IOException {
		new Sentiment("sentimentTrain.txt", "sentimentTest.txt", 35);
	}

	public void test() {
		int count[] = new int[2];
		for (int i = 0; i < trainInput.length; i++)
			test(trainInput[i], trainOutput[i], count, 1);
		xy.trainAccuracy = ((double) count[1] / trainInput.length) * 10000 / 100.0 + "%";
		for (int i = 0; i < testInput.length; i++)
			test(testInput[i], testOutput[i], count, 0);
		xy.testAccuracy = ((double) count[0] / testInput.length) * 10000 / 100.0 + "%";
	}

	public String test(double[] input, double[] label, int[] count, int ii) {
		double output[] = calculateOutput(calculateOutput(input, w1, b1), w2, b2);
		int max = 0;
		for (int i = 0; i < output.length; i++)
			max = output[i] > output[max] ? i : max;
		count[ii] += (label[max] == 1 ? 1 : 0);
		return words.get(max);
	}

	public double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x / 10));
	}

	public void loadWordsAndInputAndLabel(String train, String test) throws IOException {
		List<String[]> sentences = new ArrayList<>();
		String stopSymbles = "¡°|¡±|:|¡®|¡¯|\\(|\\)|\\[|\\]|\\{|!|,|\\.|#|}";
		for (Scanner s = new Scanner(new FileInputStream(new File(train))); s.hasNextLine();)
			sentences.add(Arrays.stream(s.nextLine().replaceAll(stopSymbles, "").split("\\s+")).map(String::toLowerCase)
					.map(String::trim).filter(ss -> !ss.isEmpty()).collect(Collectors.toList()).toArray(new String[0]));
		for (String[] sentence : sentences)
			for (int i = 0; i < sentence.length - 1; i++)
				words.add(sentence[i]);
		words = new ArrayList<>(new TreeSet<>(words));
		trainInput = new double[sentences.size()][words.size()];
		trainOutput = new double[sentences.size()][numSenmatic];
		for (int i = 0; i < sentences.size(); i++) {
			String[] sentence = sentences.get(i);
			for (int s = 0; s < sentence.length - 1; s++)
				trainInput[i][words.indexOf(sentence[s])] += 5;
			trainOutput[i][Integer.parseInt(sentence[sentence.length - 1])] = 1;
		}
		sentences = new ArrayList<>();
		for (Scanner s = new Scanner(new FileInputStream(new File(test))); s.hasNextLine();)
			sentences.add(Arrays.stream(s.nextLine().replaceAll(stopSymbles, "").split("\\s+")).map(String::toLowerCase)
					.map(String::trim).filter(ss -> !ss.isEmpty()).collect(Collectors.toList()).toArray(new String[0]));
		testInput = new double[sentences.size()][words.size()];
		testOutput = new double[sentences.size()][numSenmatic];
		for (int i = 0; i < sentences.size(); i++) {
			String[] sentence = sentences.get(i);
			for (int s = 0; s < sentence.length - 1; s++)
				if (words.contains(sentence[s]))
					testInput[i][words.indexOf(sentence[s])] += 5;
			testOutput[i][Integer.parseInt(sentence[sentence.length - 1])] = 1;
		}
	}

}
