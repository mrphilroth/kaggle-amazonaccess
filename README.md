kaggle-amazonaccess
===================

Code I used in the Amazon Employee Access Challenge competition on
Kaggle. Find my profile
[here](http://www.kaggle.com/users/25160/proth).

My other compilations of code for kaggle competitions have been much
more organized. This is a bit of dumping ground.

Amazon was interested in automating the process of granting access to
various computer resources to its employees. They released a bunch of
data about an employee's role at the company along with what resources
they had been granted access to and those they'd been denied
access. The goal was to build a model that accurately predicted that
final permission.

The employee's roles, managers, divisions, and other information was
already vectorized into unique integers. So data processing and
cleaning was not a big part of this competition. I initially dumped
this data into the gradient boosted decision trees and stochastic
gradient descent models that had performed well in the last
competition. Their performance was OK.

I then found that competitors were sharing a lot of information in the
forums. The most successful strategies involved using simple logistic
regression on a highly engineered sample of features. Specifically,
each unique integer is converted into its own column where the value
is one where it appears in the original data and zero otherwise in a
process called one-hot encoding. Extra columns can then be generated
from groups of the original features and binary columns constructed
from unique combinations of integers. This generates a very large set
of binary features. Greedy forward feature selection can then be used
to optimize the output of the final logistic regression model.

I successfully replicated those all those feature engineering
techniques to achieve a pretty good score. I then extended the
dimensionality of the feature grouping and added greedy backward
feature selection. Averaging all of these models together produced my
best score.

I also tried out some techniques that did not work. I tried to
optimize the stacking of the many models that I came up with. These
results always produced a better CV score but a lower leaderboard
score. I also tried to use the selected features in different models
like naïve bayesian. This didn't really affect the score very much.

I probably learned the most from this competition than any of the
others. I have to thank all the people on the forums that really made
that possible. I think the importance of feature engineering has
finally sunk in.

