# N-Grams

## N-gram models

N-gram models assume each word (event) `depends only on the previous n−1` words (events):

$$\text{Unigram model: } P(w^{(1)} \ldots w^{(i)} ) = \prod_{i=1}^{N} P(w^{(i)})$$

$$\text{Bigram model: } P(w^{(1)} \ldots w^{(i)} ) = \prod_{i=1}^{N} P(w^{(i)}|w^{(i-1)})$$

$$\text{Trigram model: } P(w^{(1)} \ldots w^{(i)} ) = \prod_{i=1}^{N} P(w^{(i)}|w^{(i-1)},w^{(i-2)})$$

- Independence assumptions where the n-th event in a sequence depends only on the last n-1 events are called Markov assumptions (of order n−1).

## Section table of contents

```{tableofcontents}

```
