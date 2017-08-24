word2vec
========================================================
author: 
date: 
autosize: true
transition: none

Outline
========================================================

1.  Preliminaries (why?): differing representations
    *  No Free Lunch Theorem
2.  Understanding Where to use Word2Vec 
    *  Term frequency
    *  Topic models
    *  Vectors?
3.  Building Word2Vec from scratch...
4.  Toolkits!

1. Motivation
========================================================

To understand the different parameters in word2vec models:

*  What is skip-gram or CBOW?
*  What is negative sampling?

This session is not...

*  Detailed introduction to Neural Networks
*  About deep learning

... though hopefully you will learn a bit about these things

(let me know if you want more theoretical sessions)

1. No Free Lunch
========================================================

(AI theory)

>  There is no representation/algorithm/model that will outperform all other algorithms for any problem (paraphrased)

Sometimes term frequency/topic models/word2vec models are better, other times they are not. 


1. Consequences of No Free Lunch (NFL)
========================================================

Several Scenarios I recently encounted:

*  I have running in production a model generating scores off 50 features
*  I have new data coming in (50 features) which are shown to be equally predictive
 
What do I do?

1.  Combine all features (100 features) and rebuild the model
2.  Build a model using only the new 50 features and do an ensemble
3.  ???


<!-- mention you can have 2, 3, 4, maybe 5 models in prod, which one is the best? how do you decide? 
that is a discussion for another day! --> 

<!-- and by extension there is no text representation, tfidf, lda, stemming lemmentization wordnet or otherwise which is strictly better than any other approach --> 

2. Understanding Word2Vec
========================================================

Bag of words/Topic models/word2vec all aim to convert: word(s) to numbers `==>` usefulness to a machine. 

*  Term frequency: word counts, can be normalised (TFIDF)
*  Topic model: vector represents distribution of words, i.e. association to a particular topic (supervised or unsupervised)
*  Word2Vec: some arbitary vector in some vector space???

2. Understanding Word2Vec
========================================================

Vectors allow you to measure things:

*  How close vectors are (how similar)
*  Are vectors orthogonal to each other (dissimilar)
*  Can do "arithmetic"!

Examples from the original paper:

```
vec(King) - vec(man) + vec(woman) = vec(Queen)
```

...and many other examples. 

2. Understanding Word2Vec
========================================================

Differences with other approaches:

*  Context!

Word2Vec considers context of a word in its construction. The 2 approaches in "converting" the unsupervised problem to a supervised one:

*  skip-gram: ` Pr(context|target word)`
*  continuous bag of words (CBOW): ` Pr(target word|context)`


3. Building Word2Vec from Scratch (Building Training Set)
========================================================

**Skip-gram/CBOW**

Training set construction:

```
1.  Pick window size (odd number)
2.  Extract all tokens based on this chosen window size
3.  Remove the middle word in each window; this becomes your target word, other words are your context
```

3. Building Word2Vec from Scratch (Building Training Set)
========================================================

**Skip-gram** (window size 3)

>  The cat sat on the mat

window size of 3:

*  the cat sat
*  cat sat on
*  sat on the
*  on the mat

3. Building Word2Vec from Scratch (Building Training Set)
========================================================

**Skip-gram** (window size 3)

>  The cat sat on the mat

window size of 3:
```
context: the sat,   target: cat
context: cat on,    target: sat
context: sat them,  target: on
context: on mat,    target: the
```
Now we can perform some supervised learning!

3. Building Word2Vec from Scratch (Attempt 1)
========================================================

Ignoring everything I said previous about word2vec, we can do...a multinomial regression!

Attempt 1 pseudo code
```
Model: Pr(context | target word), using multinomial regression
```

3. Building Word2Vec from Scratch (Keras)
========================================================

**Multinomial regression in Keras**

Relationship between multinomial regression and neural networks: 

Two parts for both models:

1.  linear part
2.  non-linear part

3. Building Word2Vec from Scratch (Keras)
========================================================

If $x$ is your input...

**Linear Part**

$w$ is the linear component. In (OLS) linear regression that is it:

$$
\hat{y} = wx
$$

**Non-Linear Part**

For logistic regression (and simple Neural Net) it is the logistic function

$$
\hat{y} = \frac{1}{1+e^{wx}}
$$


3. Building Word2Vec from Scratch (Keras)
========================================================

We have just shown a neural network! It has the properties:

*  No hidden layer
*  Input layer same size as number of features
*  Output layer size 2

![logistic](logistic-nn.png)

3. Building Word2Vec from Scratch (Keras)
========================================================

![logistic](logistic-nn.png)

If we have dataset with 4 features, then if $w \in \mathbb{R}^{4\times 2}$, then $\hat{y} = wx$ is of dimension $1 \times 2$; which when fed into the logistic function, would represent probability of being in class 1 (element $1, 1$) or class 2 (element $1, 2$).


3. Building Word2Vec from Scratch (Keras)
========================================================

![logistic](logistic-nn.png)

(not a good example, but you get the idea with dimensions)

```
incoming data    matrix w     =   yhat
[1, 2, 3, 4]  *  [[1,  2         [50  60]
                 3,  4
                 5,  6
                 7,  8]]

```


3. Building Word2Vec from Scratch (Keras)
========================================================

For 2 classes:

$$
\sigma(wx)_j = \frac{1}{1+e^{wx}}
$$

For more than 2 classes:

$$
\sigma(wx)_j = \frac{e^{wx}}{\sum_{\forall j} e^{wx}}
$$








