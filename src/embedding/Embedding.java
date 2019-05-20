package embedding;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.stream.Collectors;

import gui.CoordinateSystemUI;

public class Embedding {
	public List<String> words = new ArrayList<>(), stopWords = new ArrayList<>();
	public double trainInput[][], trainOutput[][], w1[][], w2[][], b1[], b2[], loss, learningRate = 0.1;
	int numNerons, count, inputAndLabelCount, iii = 0;
	Random r = new Random();
	Map<String, List<String>> inputAndLabel = new TreeMap<>();
	CoordinateSystemUI xy;

	public Embedding() {
	}

	public Embedding(String fileName, boolean removeStopWords, int dim) throws IOException {
		loadWordsAndInputAndLabel(fileName, removeStopWords);
		initializeParameters(dim, words.size());
	}

	public static void main(String[] args) throws IOException {
		new Embedding("musicText.txt", false, 3);
	}

	public void initializeParameters(int dim, int outputSize) {
		numNerons = dim > 2 ? dim : 3;
		w1 = new double[words.size()][numNerons];
		w2 = new double[numNerons][outputSize];
		b1 = new double[numNerons];
		b2 = new double[outputSize];
		for (int i = 0; i < w1.length; i++)
			for (int j = 0; j < w1[i].length; j++)
				w1[i][j] = r.nextDouble() - 0.5;
		for (int i = 0; i < w2.length; i++)
			for (int j = 0; j < w2[i].length; j++)
				w2[i][j] = r.nextDouble() - 0.5;
		for (int i = 0; i < b1.length; i++)
			b1[i] = r.nextDouble() - 0.5;
		for (int i = 0; i < b2.length; i++)
			b2[i] = r.nextDouble() - 0.5;
		xy = new CoordinateSystemUI(this);
	}

	public void train(int count) {
		for (int i = 0; i < count; i++) {
			int rand = r.nextInt(trainInput.length);
			firstLayer(trainInput[rand], trainOutput[rand]);
			if (i % 1000 == 999) {
				xy.loss = loss / 1000.0;
				xy.progress = i / (double) count;
				loss = 0;
				xy.repaint();
			}
		}
		test();
	}

	public void firstLayer(double[] input, double[] label) {
		SecondLayer(calculateOutput(input, w1, b1), label, input);
	}

	private void SecondLayer(double[] hiddenOutput, double[] label, double[] input) {
		double output[] = calculateOutput(hiddenOutput, w2, b2);
		double error[] = new double[output.length];
		double forOutputBP[] = new double[w2[0].length];
		for (int i = 0; i < error.length; i++) {
			error[i] = label[i] - output[i];
			forOutputBP[i] = -error[i] * output[i] * (1 - output[i]);
			loss += error[i] * error[i];
			for (int prevLayer = 0; prevLayer < w2.length; prevLayer++)
				w2[prevLayer][i] += learningRate * error[i] * hiddenOutput[prevLayer];
		}
		backPropagationForBothLayer(hiddenOutput, error, forOutputBP, input);
	}

	public void backPropagationForBothLayer(double[] hiddenOutput, double[] error, double[] forOutputBP,
			double[] input) {
		for (int prevLayer = 0; prevLayer < w2.length; prevLayer++)
			for (int currLayer = 0; currLayer < w2[prevLayer].length; currLayer++)
				w2[prevLayer][currLayer] -= learningRate * hiddenOutput[prevLayer] * forOutputBP[currLayer];
		for (int currLayer = 0; currLayer < w2[0].length; currLayer++)
			b2[currLayer] += learningRate * (error[currLayer] - forOutputBP[currLayer]);
		double forOutputBPf[] = new double[w2.length];
		for (int prevLayer = 0; prevLayer < w2.length; prevLayer++)
			for (int currLayer = 0; currLayer < w2[0].length; currLayer++)
				forOutputBPf[prevLayer] += forOutputBP[currLayer] * w2[prevLayer][currLayer];
		for (int i = 0; i < w1.length; i++)
			for (int prevLayer = 0; prevLayer < w2.length; prevLayer++)
				w1[i][prevLayer] -= learningRate * forOutputBPf[prevLayer] * hiddenOutput[prevLayer]
						* (1 - hiddenOutput[prevLayer]) * input[i];
		for (int prevLayer = 0; prevLayer < w2.length; prevLayer++)
			b1[prevLayer] -= learningRate * forOutputBPf[prevLayer] * hiddenOutput[prevLayer]
					* (1 - hiddenOutput[prevLayer]);
	}

	public void test() {
		inputAndLabel.entrySet().forEach(e -> e.getValue().forEach(label -> test(e.getKey(), label)));
		xy.trainAccuracy = ((double) count / trainInput.length) * 10000 / 100.0 + "%";
		count = 0;
	}

	public String test(String word, String label) {
		double input[] = new double[words.size()];
		input[words.indexOf(word)] = 1;
		double output[] = calculateOutput(calculateOutput(input, w1, b1), w2, b2);
		int max = 0;
		double sum = 0;
		for (int i = 0; i < output.length; i++) {
			sum += output[i];
			if (output[i] > output[max])
				max = i;
		}
		if (inputAndLabel.get(word).contains(words.get(max)))
			count++;
		System.out.println("Input: " + word + ", output: " + words.get(max) + ", probability: "
				+ (int) (output[max] / sum * 10000) / 100.0 + "%" + ", prediction: "
				+ inputAndLabel.get(word).contains(words.get(max)) + " label: " + label);
		return words.get(max);
	}

	public double[] calculateOutput(double input[], double weights[][], double bias[]) {
		double output[] = new double[weights[0].length];
		for (int prevLayer = 0; prevLayer < weights.length; prevLayer++)
			for (int currLayer = 0; currLayer < weights[prevLayer].length; currLayer++)
				output[currLayer] += weights[prevLayer][currLayer] * input[prevLayer];
		for (int currLayer = 0; currLayer < weights[0].length; currLayer++)
			output[currLayer] = sigmoid(output[currLayer] + bias[currLayer]);
		return output;
	}

	public double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}

	public List<String> loadWordsAndInputAndLabel(String filename, boolean removeStopWords) throws IOException {
		List<String> sentences = new ArrayList<>();
		String stopSymbles = "¡°|¡±|:|¡®|¡¯|\\(|\\)|\\[|\\]|\\{|\\}";
		if (removeStopWords)
			for (Scanner s = new Scanner(new FileInputStream(new File("stopWords.txt"))); s.hasNextLine();)
				stopWords.add(s.nextLine());
		stopWords = new ArrayList<>(new TreeSet<>(stopWords));
		for (Scanner s = new Scanner(new FileInputStream(new File(filename))); s.hasNextLine();)
			sentences.addAll(Arrays.asList(Arrays.stream(s.nextLine().split(",|\\.|\\?|!")).map(ss -> ss.toLowerCase())
					.map(l -> l.replaceAll(stopSymbles, " ")).map(String::trim).filter(l -> !l.isEmpty())
					.collect(Collectors.toList()).toArray(new String[0])));
		for (String sentence : sentences) {
			String sentenc[] = sentence.split("\\s+");
			for (int i = 0; i < sentenc.length - 1; i++)
				if (!stopWords.contains(sentenc[i]) && !stopWords.contains(sentenc[i + 1])) {
					String bigram[] = { sentenc[i], sentenc[i + 1] };
					if (!inputAndLabel.containsKey(bigram[0])) {
						inputAndLabelCount++;
						inputAndLabel.put(bigram[0], new ArrayList<String>(Arrays.asList(bigram[1])));
					} else if (inputAndLabel.containsKey(bigram[0])
							&& !inputAndLabel.get(bigram[0]).contains(bigram[1])) {
						inputAndLabel.get(bigram[0]).add(bigram[1]);
						inputAndLabelCount++;
					}
					words.add(sentenc[i]);
					words.add(sentenc[i + 1]);
				}
		}
		words = new ArrayList<>(new TreeSet<>(words));
		words.removeAll(stopWords);
		trainInput = new double[inputAndLabelCount][words.size()];
		trainOutput = new double[inputAndLabelCount][words.size()];
		inputAndLabel.entrySet().forEach(e -> e.getValue().forEach(label -> {
			trainInput[iii][words.indexOf(e.getKey())] = 1;
			trainOutput[iii++][words.indexOf(label)] = 1;
		}));
		return null;
	}

}
