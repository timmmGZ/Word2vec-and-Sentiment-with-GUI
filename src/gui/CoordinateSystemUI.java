package gui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.GridLayout;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JTextArea;
import javax.swing.JTextField;

import embedding.Embedding;

public class CoordinateSystemUI extends JPanel implements MouseMotionListener {
	/**
	 * timmmGz
	 */
	private static final long serialVersionUID = 1L;
	public DecimalFormat df = new DecimalFormat("0.00");
	double xy[][], b[], zoom, offsetX = 0, offsetY = 0;
	public double loss, progress;
	List<String> words;
	Set<Integer> mouse = new TreeSet<>();
	JButton learningRate = new JButton("learning rate"), train = new JButton("train");
	JTextField lr = new JTextField("0.1"), trainIteration = new JTextField("100000");
	JTextArea jta = new JTextArea();
	StringBuilder sb = new StringBuilder();
	int size = 450, dim, lastX, lastY, iii = 0;
	public String testAccuracy, trainAccuracy = "";

	public CoordinateSystemUI(Embedding nn) {
		JFrame f = new JFrame("timmmGZ¡ª¡ªword2vec and sentiment");
		jta.setPreferredSize(new Dimension(150, 0));
		jta.setEditable(false);
		xy = nn.w1;
		b = nn.b1;
		words = nn.words;
		zoom = 35 + (dim = xy[0].length);
		JPanel jp = new JPanel(new GridLayout(0, 4));
		learningRate.addActionListener(e -> nn.learningRate = Double.parseDouble(lr.getText()));
		train.addActionListener(e -> new Thread(() -> nn.train(Integer.parseInt(trainIteration.getText()))).start());
		jp.add(train);
		jp.add(trainIteration);
		jp.add(learningRate);
		jp.add(lr);
		addMouseMotionListener(this);
		addMouseWheelListener(m -> zoom -= zoom > m.getWheelRotation() ? m.getWheelRotation() : 0);
		setPreferredSize(new Dimension(900, 900));
		f.add(this, BorderLayout.CENTER);
		f.add(jp, BorderLayout.SOUTH);
		f.add(jta, BorderLayout.EAST);
		f.setResizable(false);
		f.pack();
		f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		f.setLocationRelativeTo(null);
		f.setVisible(true);
	}

	@Override
	public void paintComponent(Graphics g) {
		super.paintComponent(g);
		g.drawString("Loss: " + loss, 10, 20);
		g.drawString("Progress: " + df.format(progress * 100) + "%", 10, 35);
		g.drawString("Last accuracy on train set: " + trainAccuracy, 10, 50);
		g.drawString(testAccuracy == null ? "" : "Last accuracy on test set: " + testAccuracy, 10, 65);

		g.setFont(new Font("MS Song", Font.PLAIN, 20));
		for (int i = 0; i < xy.length; i++) {
			if (!mouse.contains(i)) {
				g.setColor(new Color((int) (255 / (1 + Math.exp(-xy[i][dim - 3] / 2))), 0, 0));
				g.drawString(words.get(i), getWidth() / 2 + (int) ((offsetX + xy[i][dim - 1] + b[dim - 1]) * zoom),
						getHeight() / 2 - (int) ((-offsetY + xy[i][dim - 2] + b[dim - 2]) * zoom));
			}
		}
		g.setFont(new Font("MS Song", Font.BOLD, 30));
		for (int i = 0, count = -mouse.size() / 2; i < xy.length; i++)
			if (mouse.contains(i)) {
				g.setColor(new Color(0, (int) (255 / (1 + Math.exp(-xy[i][dim - 3] / 2))), 0));
				g.drawString(words.get(i), getWidth() / 2 + (int) ((offsetX + xy[i][dim - 1] + b[dim - 1]) * zoom),
						getHeight() / 2 - (int) ((-offsetY + xy[i][dim - 2] + b[dim - 2]) * zoom) + count++ * 20);
			}
		jta.setText(sb.toString());
		g.setColor(Color.BLACK);
		g.drawLine(0, (int) (offsetY * zoom) + getHeight() / 2, getWidth(), (int) (offsetY * zoom) + getHeight() / 2);
		g.drawLine((int) (offsetX * zoom) + getWidth() / 2, 0, (int) (offsetX * zoom) + getWidth() / 2, getHeight());
		repaint();
	}

	@Override
	public void mouseDragged(MouseEvent m) {
		offsetX += (m.getX() - lastX) / zoom;
		lastX = m.getX();
		offsetY += (m.getY() - lastY) / zoom;
		lastY = m.getY();
	}

	@Override
	public void mouseMoved(MouseEvent m) {
		sb = new StringBuilder();
		lastX = m.getX();
		lastY = m.getY();
		sb.append("z-Axis, words\n");
		List<List<String>> l = new ArrayList<>();
		for (int i = 0; i < xy.length; i++) {
			if (isAround(m.getX() - size, size - m.getY(),
					(int) ((offsetX + xy[i][dim - 1] + b[dim - 1]) * zoom + (words.get(i).length() * 3.5)),
					(int) ((-offsetY + xy[i][dim - 2] + b[dim - 2]) * zoom) + 5)) {
				l.add(Arrays.asList(String.valueOf((255 / (1 + Math.exp(-xy[i][dim - 3] / 2)))), words.get(i) + "\n"));
				mouse.add(i);
			} else
				mouse.remove(i);
		}
		l.stream().sorted((o1, o2) -> (int) Double.parseDouble(o1.get(0)) - (int) Double.parseDouble(o2.get(0)))
				.forEach(s -> sb
						.append(String.format("%-10s", df.format(Double.parseDouble(s.get(0)) - 127.5)) + s.get(1)));
	}

	private boolean isAround(int x, int y, int X, int Y) {
		return X - 25 < x && x < X + 25 && Y - 10 < y && y < Y + 10;
	}

}