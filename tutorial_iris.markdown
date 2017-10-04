Most of the following floats are limited to a precision of 2 decimals

# DATA DESCRIPTION ###

- name = iris
- Number of features = 4
- Number of classes = 3
- Classes = [0, 1, 2]
- Samples with only weak = 135
- Samples with true labels = 15

## HOW THE WEAK LABELS ARE GENERATED ###

This dataset only has true labels. In order to create the weak
            labels we designed a mixing matrix with the following steps:

Given the decimal true labels y

- y = [1 0 1 0 2 0]

We create artificially the mixing matrix M

- method = quasi_IPL
- alpha = 0.5
- beta = 0.3

Resulting mixing matrix M

| | **0** | **1** | **2** |
| :-: | :-: | :-: | :-: |
| **0** | 0.00 | 0.00 | 0.00 |
| **1** | 0.00 | 0.00 | 0.54 |
| **2** | 0.00 | 0.54 | 0.00 |
| **3** | 0.00 | 0.23 | 0.23 |
| **4** | 0.54 | 0.00 | 0.00 |
| **5** | 0.23 | 0.00 | 0.23 |
| **6** | 0.23 | 0.23 | 0.00 |
| **7** | 0.00 | 0.00 | 0.00 |



Binarize the true labels and store in Y

<pre>
[[0 1 0]
 [1 0 0]
 [0 1 0]
 [1 0 0]
 [0 0 1]
 [1 0 0]]
</pre>


Generate the weak labels with the mixing matrix M

<pre>
[3 5 3 5 1 5]
</pre>


Binarize the weak labels and store in Z

<pre>
[[0 1 1]
 [1 0 1]
 [0 1 1]
 [1 0 1]
 [0 0 1]
 [1 0 1]]
</pre>


## DATA SAMPLE ###

Example of true labels Y

- In decimal y =

<pre>
[1 0 1 0 2 0]
</pre>


- In binary Y =

<pre>
[[0 1 0]
 [1 0 0]
 [0 1 0]
 [1 0 0]
 [0 0 1]
 [1 0 0]]
</pre>


Example of corresponding weak labels Z

- In decimal z =

<pre>
[2 5 3 4 5 5]
</pre>


- In binary Z =

<pre>
[[0 1 0]
 [1 0 1]
 [0 1 1]
 [1 0 0]
 [1 0 1]
 [1 0 1]]
</pre>


Prior distribution for **all** true labels. $P(Y)$

<pre>
[ 0.33  0.33  0.33]
</pre>


# EXPECTED ERROR FOR A BASELINE ###

We show the performance of a simple model that always predicts the prior

## BRIER SCORE #
Error matrix with Brier score: $\Psi_{BS} = BS(P(Y), I)$

| | **100** | **010** | **001** |
| :-: | :-: | :-: | :-: |
| **0.33** | 0.44 | 0.11 | 0.11 |
| **0.33** | 0.11 | 0.44 | 0.11 |
| **0.33** | 0.11 | 0.11 | 0.44 |



Expected Brier score: $\mathbb{E}_{y\sim P(y)} [\Psi_{BS}(S, y)] = \sum_{j=1}^K P(y=j) \Psi_{BS}(S, y_j)$

<pre>
0.666666666667
</pre>


## LOG-LOSS #
Error matrix with Log-loss: $\Psi_{LL} = LL(P(Y), I)$

| | **100** | **010** | **001** |
| :-: | :-: | :-: | :-: |
| **0.33** | 1.10 | -0.00 | -0.00 |
| **0.33** | -0.00 | 1.10 | -0.00 |
| **0.33** | -0.00 | -0.00 | 1.10 |



Expected Log-loss: $\mathbb{E}_{y\sim P(y)} [\Psi_{LL}(S, y)] = \sum_{j=1}^K P(y=j) \Psi_{LL}(S, y_j)$

<pre>
1.09861228867
</pre>


# EXAMPLE OF ESTIMATION OF THE MIXING MATRIX M ###

From now, $M_0$ is for the weak set and $M_1$ for the **full set** that contains the true labels

Lets imagine our full set of weak and true labels is this small sample

- z = [2 5 3 4 5 5]

- y = [1 0 1 0 2 0]

We can estimate the probability of each weak label given the true
         label by counting first the number of occurrences of both happening at
         the same time

| | **0** | **1** | **2** |
| :-: | :-: | :-: | :-: |
| **0** | 0 | 0 | 0 |
| **1** | 0 | 0 | 0 |
| **2** | 0 | 1 | 0 |
| **3** | 0 | 1 | 0 |
| **4** | 1 | 0 | 0 |
| **5** | 2 | 0 | 1 |
| **6** | 0 | 0 | 0 |
| **7** | 0 | 0 | 0 |



Where there is one column per true label and one row per each
         possible weak label

Then we can compute the probability of each weak label given the true
         label by dividing every column by its sum. If we do that, we will get
         a possible estimation of $M_0$

Estimated $M_0$

| | **0** | **1** | **2** |
| :-: | :-: | :-: | :-: |
| **0** | 0.00 | 0.00 | 0.00 |
| **1** | 0.00 | 0.00 | 0.00 |
| **2** | 0.00 | 0.50 | 0.00 |
| **3** | 0.00 | 0.50 | 0.00 |
| **4** | 0.33 | 0.00 | 0.00 |
| **5** | 0.67 | 0.00 | 1.00 |
| **6** | 0.00 | 0.00 | 0.00 |
| **7** | 0.00 | 0.00 | 0.00 |



However, because given a small data size it is possible that some of
         the weak labels does not occur. We can apply a Laplace correction by
         adding one count to each possible weak label given the true label

| | **0** | **1** | **2** |
| :-: | :-: | :-: | :-: |
| **0** | 1 | 1 | 1 |
| **1** | 1 | 1 | 1 |
| **2** | 1 | 2 | 1 |
| **3** | 1 | 2 | 1 |
| **4** | 2 | 1 | 1 |
| **5** | 3 | 1 | 2 |
| **6** | 1 | 1 | 1 |
| **7** | 1 | 1 | 1 |



Estimated $M_0$ with Laplace correction

| | **0** | **1** | **2** |
| :-: | :-: | :-: | :-: |
| **0** | 0.09 | 0.10 | 0.11 |
| **1** | 0.09 | 0.10 | 0.11 |
| **2** | 0.09 | 0.20 | 0.11 |
| **3** | 0.09 | 0.20 | 0.11 |
| **4** | 0.18 | 0.10 | 0.11 |
| **5** | 0.27 | 0.10 | 0.22 |
| **6** | 0.09 | 0.10 | 0.11 |
| **7** | 0.09 | 0.10 | 0.11 |



The mixing matrix for the clean data $M_1$

| | **0** | **1** | **2** |
| :-: | :-: | :-: | :-: |
| **0** | 1.00 | 0.00 | 0.00 |
| **1** | 0.00 | 1.00 | 0.00 |
| **2** | 0.00 | 0.00 | 1.00 |



# ESTIMATION OF THE MIXING MATRIX M FOR ALL DATA ###

Now lets do the same but with the full set of weak and true labels

This is the count

| | **0** | **1** | **2** |
| :-: | :-: | :-: | :-: |
| **0** | 0 | 0 | 0 |
| **1** | 0 | 0 | 1 |
| **2** | 0 | 3 | 0 |
| **3** | 0 | 2 | 3 |
| **4** | 2 | 0 | 0 |
| **5** | 3 | 0 | 1 |
| **6** | 0 | 0 | 0 |
| **7** | 0 | 0 | 0 |



Estimated $M_0$ without Laplace correction

| | **0** | **1** | **2** |
| :-: | :-: | :-: | :-: |
| **0** | 0.00 | 0.00 | 0.00 |
| **1** | 0.00 | 0.00 | 0.20 |
| **2** | 0.00 | 0.60 | 0.00 |
| **3** | 0.00 | 0.40 | 0.60 |
| **4** | 0.40 | 0.00 | 0.00 |
| **5** | 0.60 | 0.00 | 0.20 |
| **6** | 0.00 | 0.00 | 0.00 |
| **7** | 0.00 | 0.00 | 0.00 |



Estimated $M_0$ with Laplace correction

| | **0** | **1** | **2** |
| :-: | :-: | :-: | :-: |
| **0** | 0.08 | 0.08 | 0.08 |
| **1** | 0.08 | 0.08 | 0.15 |
| **2** | 0.08 | 0.31 | 0.08 |
| **3** | 0.08 | 0.23 | 0.31 |
| **4** | 0.23 | 0.08 | 0.08 |
| **5** | 0.31 | 0.08 | 0.15 |
| **6** | 0.08 | 0.08 | 0.08 |
| **7** | 0.08 | 0.08 | 0.08 |



The mixing matrix for the clean data $M_1$

| | **0** | **1** | **2** |
| :-: | :-: | :-: | :-: |
| **0** | 1.00 | 0.00 | 0.00 |
| **1** | 0.00 | 1.00 | 0.00 |
| **2** | 0.00 | 0.00 | 1.00 |



## COMBINATION OF BOTH MATRICES FOR THE FULL DATASET ###

Proportion of samples with weak ($q_0$) and with true labels ($q_1$)

- $q_0$ = 0.9
- $q_1$ = 0.1

Composition of mixing matrices $M = [q_0*M_0 \mathtt{ , } q_1*M_1]^T$

| | **0** | **1** | **2** |
| :-: | :-: | :-: | :-: |
| **0** | 0.07 | 0.07 | 0.07 |
| **1** | 0.07 | 0.07 | 0.14 |
| **2** | 0.07 | 0.28 | 0.07 |
| **3** | 0.07 | 0.21 | 0.28 |
| **4** | 0.21 | 0.07 | 0.07 |
| **5** | 0.28 | 0.07 | 0.14 |
| **6** | 0.07 | 0.07 | 0.07 |
| **7** | 0.07 | 0.07 | 0.07 |
| **8** | 0.10 | 0.00 | 0.00 |
| **9** | 0.00 | 0.10 | 0.00 |
| **10** | 0.00 | 0.00 | 0.10 |



The corresponding indices of the weak labels to the rows of the matrix M

<pre>
[2 5 3 4 5 5]
</pre>


The corresponding indices of the true labels to the rows of the matrix M

<pre>
[ 9  8  9  8 10  8]
</pre>


# TRAINING SOME MODELS ###

From here I will add an extra feature fixed to 1 for the bias

<pre>
[[ 5.7  2.8  4.5  1.3  1. ]
 [ 5.1  3.8  1.9  0.4  1. ]
 [ 7.   3.2  4.7  1.4  1. ]
 [ 5.1  3.8  1.6  0.2  1. ]
 [ 6.3  2.9  5.6  1.8  1. ]
 [ 4.6  3.1  1.5  0.2  1. ]]
</pre>


## SIMULATION OF GRADIENT DESCENT FOR A LR ###

Initial weights w

<pre>
[[ 1.76  0.4   0.98]
 [ 2.24  1.87 -0.98]
 [ 0.95 -0.15 -0.1 ]
 [ 0.41  0.14  1.45]
 [ 0.76  0.12  0.44]]
</pre>


The small sample of X

<pre>
[[ 5.7  2.8  4.5  1.3  1. ]
 [ 5.1  3.8  1.9  0.4  1. ]
 [ 7.   3.2  4.7  1.4  1. ]
 [ 5.1  3.8  1.6  0.2  1. ]
 [ 6.3  2.9  5.6  1.8  1. ]
 [ 4.6  3.1  1.5  0.2  1. ]]
</pre>


Initial model activations m_a = X m_W

<pre>
[[ 21.9    7.14   4.71]
 [ 20.24   9.03   2.11]
 [ 25.32   8.39   5.72]
 [ 19.88   9.05   1.85]
 [ 24.43   7.47   5.82]
 [ 17.33   7.55   2.05]]
</pre>


Initial model outputs m_q = softmax(m_a)

<pre>
[[  1.00e+00   3.88e-07   3.43e-08]
 [  1.00e+00   1.35e-05   1.33e-08]
 [  1.00e+00   4.43e-08   3.07e-09]
 [  1.00e+00   1.98e-05   1.48e-08]
 [  1.00e+00   4.30e-08   8.22e-09]
 [  1.00e+00   5.68e-05   2.32e-07]]
</pre>


Initial model predictions m_pred = argmax(m_q)

<pre>
[0 0 0 0 0 0]
</pre>


Compute the error of the predicted probabilities q true labels Y

<pre>
[  2.00e+00   3.65e-10   2.00e+00   7.85e-10   2.00e+00   6.48e-09]
</pre>


Compute the gradient of the Brier score

<pre>
[[ 19.  -12.7  -6.3]
 [  8.9  -6.   -2.9]
 [ 14.8  -9.2  -5.6]
 [  4.5  -2.7  -1.8]
 [  3.   -2.   -1. ]]
</pre>


Update the weights $W_{t+1} = W_t - G$

<pre>
[[-17.24  13.1    7.28]
 [ -6.66   7.87   1.92]
 [-13.85   9.05   5.5 ]
 [ -4.09   2.84   3.25]
 [ -2.24   2.12   1.44]]
</pre>


### WE CAN PERFORM GD FOR SOME ITERATIONS FULL TRUE SET ###

Iterations :

| it | mean bs | acc |
| :--- | :-----------: | :-----: |
| 0 | 1.33E+00 | 0.33 |
| 1 | 1.33E+00 | 0.33 |
| 2 | 1.33E+00 | 0.33 |
| 3 | 1.25E+00 | 0.33 |
| 4 | 1.33E+00 | 0.33 |
| 5 | 6.67E-01 | 0.67 |
| 6 | 6.67E-01 | 0.67 |
| 7 | 6.67E-01 | 0.67 |
| 8 | 6.67E-01 | 0.67 |
| 9 | 6.67E-01 | 0.67 |
| 10 | 6.67E-01 | 0.67 |
| 11 | 6.67E-01 | 0.67 |
| 12 | 6.67E-01 | 0.67 |
| 13 | 6.67E-01 | 0.67 |
| 14 | 1.17E+00 | 0.40 |
| 15 | 1.31E+00 | 0.33 |
| 16 | 1.33E+00 | 0.33 |
| 17 | 6.63E-01 | 0.67 |
| 18 | 1.33E+00 | 0.33 |
| 19 | 7.35E-01 | 0.60 |

## SIMULATION OF THE EXPECTATION MAXIMIZATION STEPS FOR A LR ###

Initial weights w

<pre>
[[ 1.76  0.4   0.98]
 [ 2.24  1.87 -0.98]
 [ 0.95 -0.15 -0.1 ]
 [ 0.41  0.14  1.45]
 [ 0.76  0.12  0.44]]
</pre>


The small sample of X

<pre>
[[ 5.7  2.8  4.5  1.3  1. ]
 [ 5.1  3.8  1.9  0.4  1. ]
 [ 7.   3.2  4.7  1.4  1. ]
 [ 5.1  3.8  1.6  0.2  1. ]
 [ 6.3  2.9  5.6  1.8  1. ]
 [ 4.6  3.1  1.5  0.2  1. ]]
</pre>


#### EXPECTATION STEP: Assume the weights of the model are right ###

Initial prediction Z = X w

<pre>
[[ 21.9    7.14   4.71]
 [ 20.24   9.03   2.11]
 [ 25.32   8.39   5.72]
 [ 19.88   9.05   1.85]
 [ 24.43   7.47   5.82]
 [ 17.33   7.55   2.05]]
</pre>


Initial q = softmax(Z)

<pre>
[[  1.00e+00   3.88e-07   3.43e-08]
 [  1.00e+00   1.35e-05   1.33e-08]
 [  1.00e+00   4.43e-08   3.07e-09]
 [  1.00e+00   1.98e-05   1.48e-08]
 [  1.00e+00   4.30e-08   8.22e-09]
 [  1.00e+00   5.68e-05   2.32e-07]]
</pre>


Initial predictions

<pre>
[0 0 0 0 0 0]
</pre>


Compute the virtual labels Q = q * M[Z_index]

<pre>
[[  6.92e-02   1.07e-07   2.38e-09]
 [  2.77e-01   9.34e-07   1.84e-09]
 [  6.92e-02   9.21e-09   8.50e-10]
 [  2.08e-01   1.37e-06   1.03e-09]
 [  2.77e-01   2.98e-09   1.14e-09]
 [  2.77e-01   3.93e-06   3.21e-08]]
</pre>


Normalize the virtual labels to sum to one

<pre>
[[  1.00e+00   1.55e-06   3.43e-08]
 [  1.00e+00   3.37e-06   6.65e-09]
 [  1.00e+00   1.33e-07   1.23e-08]
 [  1.00e+00   6.60e-06   4.94e-09]
 [  1.00e+00   1.07e-08   4.11e-09]
 [  1.00e+00   1.42e-05   1.16e-07]]
</pre>


Search for infinite or NaN values and remove them for this training step
The following samples are NOT removed from the training

<pre>
[0 1 2 3 4 5]
</pre>


#### MAXIMIZATION STEP: In our case update the weights to MINIMIZE the error###

Compute the error of the predicted probabilities q against the new Virtual labels Q

<pre>
[  2.71e-12   2.05e-10   1.75e-14   3.49e-10   2.38e-15   3.64e-09]
</pre>


Compute the gradient of the Brier score

<pre>
[[ -2.64e-08   2.63e-08   4.71e-11]
 [ -1.81e-08   1.80e-08   3.19e-11]
 [ -8.63e-09   8.62e-09   1.53e-11]
 [ -1.18e-09   1.18e-09   2.03e-12]
 [ -5.65e-09   5.64e-09   1.02e-11]]
</pre>


Update the weights $W_{t+1} = W_t - G$

<pre>
[[ 1.76  0.4   0.98]
 [ 2.24  1.87 -0.98]
 [ 0.95 -0.15 -0.1 ]
 [ 0.41  0.14  1.45]
 [ 0.76  0.12  0.44]]
</pre>


### WE CAN PERFORM EM FOR SOME ITERATIONS ###

Iterations :

| it | mean bs | acc |
| :-- | :-------: | :---: |
| 0 | 5.59E-10 | 0.33
| 1 | 5.59E-10 | 0.33
| 2 | 5.59E-10 | 0.33
| 3 | 5.59E-10 | 0.33
| 4 | 5.59E-10 | 0.33
| 5 | 5.59E-10 | 0.33
| 6 | 5.59E-10 | 0.33
| 7 | 5.59E-10 | 0.33
| 8 | 5.59E-10 | 0.33
| 9 | 5.59E-10 | 0.33
| 10 | 5.59E-10 | 0.33
| 11 | 5.59E-10 | 0.33
| 12 | 5.59E-10 | 0.33
| 13 | 5.59E-10 | 0.33
| 14 | 5.59E-10 | 0.33
| 15 | 5.59E-10 | 0.33
| 16 | 5.59E-10 | 0.33
| 17 | 5.59E-10 | 0.33
| 18 | 5.59E-10 | 0.33
| 19 | 5.59E-10 | 0.33
