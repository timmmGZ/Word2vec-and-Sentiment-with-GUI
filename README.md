# Word2vec-and-Sentiment-with-GUI
The program is using a 2 layer neural network, because single network can't do word embedding.
# Neural network structure:  
Input: One-Hot encoding vector  
Hidden layer:[3,infinite) neurons  
Output: vector with same size of input  
# How to embed?
The vector of a word is determined by the weights between input and hidden layer.  
e.g. for a 3 neurons hidden layer:  
vector("Guangzhou") = (x,y,z) = (w1["guangzhou"][0]+b1[0],w1["guangzhou"][1]+b1[1],w1["guangzhou"][2]+b1[2])=(5.20,-1.67,5.17)  
There has 3 *.java, each has main() function to run, the program is using basic algorithms for making it easy to understand the idea of embedding.  
# Embedding.java
This is the normal one only to show GUI, for convinence I am not implementing PCA for GUI here, so the panel just show a 3 axes CoordinateSystem even if your embedding words have a higher dimension(dimension = number of neurons in first hidden layer), the color of words means: the darker, the smaller value on z-axis, you could change the field of view by dragging mouse or rolling wheel.
# EmbeddingPro.java  
Produce more information such as generating sentence, similarity of words(calculate by using Euclidean distance or Cosine similarity chosen by you)
![image](https://github.com/timmmGZ/Word2vec-and-Sentiment-with-GUI/blob/master/images/embedding.png?raw=true)
# Sentiment.java  
For a small data set, Navie-Bayes-Classification will give a good accuracy, but I prefer to use neural nertork, here the structure of neural network is a bit different, because sentiment analysis is for other purpose though it is base on word embedding, so here the output size is 5, which means:  
0 - negative  
1 - somewhat negative  
2 - neutral  
3 - somewhat positive  
4 - positive  
and I am using the data set from a Kaggle competition https://www.kaggle.com/c/movie-review-sentiment-analysis-kernels-only/data  
The train set has 156k sentences and 15.5K unique words, the test set has 66K sentences, it is not a good idea to use One-Hot encoding on training, just think about large amount of word vectors with 15.5K dimension in memory.... so I decrease the size of train set to be 15k sentences, and test set 6k.  Also you may think we could decrease the input size by using CNN, but this actually decrease the accuracy, because CNN extracts features from photo, but such word vector doesn't have particular features, accuracy may decrease everytime you do downsampling (the more pooling, the lower accuracy), this is my guess by logical thinking, you could just try to check it in Python
![image](https://github.com/timmmGZ/Word2vec-and-Sentiment-with-GUI/blob/master/images/Sentiment.png?raw=true)
The test set accuracy is around 66%, maximal I get is 68%+, train set accuracy is not what sentiment focus on, anyway you could train it longer to get 99%+, but the test accuracy will decrease in the meanwhile. Here is the best accuracy on kaggle
https://www.kaggle.com/c/movie-review-sentiment-analysis-kernels-only/leaderboard
![image](https://github.com/timmmGZ/Word2vec-and-Sentiment-with-GUI/blob/master/images/scores.png?raw=true)  
Do not try to fight for a high accuracy, because sentiment is subjective like emotion, you are even hard to guess a stranger's feeling when it suddenly talks to you on a street, and we are not having only 0 happy 1 sad, but 5 emotions as output, and you may have different opinion of senmention than the one who subjectively labelled the data set.
