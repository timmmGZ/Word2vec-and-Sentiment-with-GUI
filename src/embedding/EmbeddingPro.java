package embedding;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Map;
import java.util.TreeMap;

public class EmbeddingPro extends Embedding {
	double generatorRandomly = 0 / 100;

	public EmbeddingPro(String fileName, boolean removeStopWords, int dim, double rate) throws IOException {
		super(fileName, removeStopWords, dim);
		generatorRandomly = rate;
	}

	public static void main(String[] args) throws IOException {
		new EmbeddingPro("musicText.txt", true, 3, 0);
		// new EmbeddingPro("musicText.txt", false, 3, 0);
		// new EmbeddingPro("5lyrics.txt", true, 5, 0);
		// new EmbeddingPro("5lyrics.txt", false, 50, 0.05);
	}

	public void train(int count) {
		super.train(count);
		System.out.println("=========================generate some sentence=============================");
		for (int i = 0; i < 10; i++)
			generate(100);
		System.out.println("=========================closest words=============================");
		for (int i = 0; i < words.size(); i++)
			System.out.println(String.format("%-20s", words.get(i)) + ": " + cosineSimilarity(words.get(i)));// or
																												// distanceFromKNN(words.get(i))
		System.out.println("=============try again until getting 100% accuracy=================");
	}

	private void generate(int count) {
		String word = new ArrayList<>(inputAndLabel.keySet()).get(r.nextInt(inputAndLabel.size())), generate = word,
				last = word, next;
		int i;
		for (i = 0; i < count; i++) {
			do {
				next = generateNext(last, (last = word));
			} while (next.equals(word));
			if (next.equals(""))
				break;
			generate += " " + (word = next);
		}
		System.out.print((1 < i) ? generate + "\n" : "");
	}

	public String generateNext(String last, String word) {
		if (!inputAndLabel.containsKey(word))
			return "";
		double input[] = new double[words.size()];
		input[words.indexOf(word)] = 1;
		double output[] = calculateOutput(calculateOutput(input, w1, b1), w2, b2);
		int max[] = new int[2];
		if (generatorRandomly != 0) {
			if (inputAndLabel.get(word).size() > 1 && !last.equals("")) {
				input = new double[words.size()];
				input[words.indexOf(last)] = 1;
				double lastOutput[] = calculateOutput(calculateOutput(input, w1, b1), w2, b2);
				for (String w : inputAndLabel.get(word))
					if (lastOutput[words.indexOf(w)] * output[words.indexOf(w)] > lastOutput[max[0]] * output[max[0]]) {
						max[1] = max[0];
						max[0] = words.indexOf(w);
					}
				return words.get(r.nextDouble() < generatorRandomly ? max[1] : max[0]);
			}
		}
		for (int i = 0; i < output.length; i++)
			if (output[i] > output[max[0]]) {
				max[1] = max[0];
				max[0] = i;
			}
		return words.get(r.nextDouble() < generatorRandomly ? max[1] : max[0]);
	}

	public String distanceFromKNN(String word) {// Euclidean distance
		int index = words.indexOf(word);
		Map<String, Double> d = new TreeMap<>();
		double distance[] = new double[words.size()];
		for (int w = 0; w < words.size(); w++) {
			for (int i = 0; i < w1[index].length; i++)
				distance[w] += powTwo(w1[w][i] - w1[index][i]);
			d.put(words.get(w), Math.sqrt(distance[w]));
		}
		return d.entrySet().stream().sorted(Comparator.comparing(Map.Entry<String, Double>::getValue))
				.map(a -> String.format("%-20s", a.getKey())).skip(1).limit(3).reduce((sum, w) -> sum + "," + w).get();
	}

	public String cosineSimilarity(String word) {
		int index = words.indexOf(word);
		Map<String, Double> d = new TreeMap<>();
		for (int w = 0; w < words.size(); w++) {
			if (w == words.indexOf(word))
				continue;
			double wLen = 0, wordLen = 0, dot = 0;
			for (int i = 0; i < w1[index].length; i++) {
				dot += (w1[w][i] + b1[i]) * (w1[index][i] + b1[i]);
				wLen += powTwo(w1[w][i] + b1[i]);
				wordLen += powTwo(w1[index][i] + b1[i]);
			}
			wLen = Math.sqrt(wLen);
			wordLen = Math.sqrt(wordLen);
			d.put(words.get(w), dot / (wLen * wordLen));
		}
		return d.entrySet().stream().sorted(Comparator.comparing(Map.Entry<String, Double>::getValue).reversed())
				.filter(a -> !a.getValue().isNaN() && a.getValue() > 0)
				.map(a -> String.format("%-20s", a.getKey() + " " + xy.df.format(a.getValue()))).skip(0).limit(3)
				.reduce((sum, w) -> sum + " " + w).get();
	}

	public static double powTwo(double a) {
		return a * a;
	}
}
