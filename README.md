adaboost
========

AdaBoost

This is based on a linear regression trainer and feature selection class initially developed to help
analyze and make predictions for the MIT Big Data Challenge. The trainer can use any provided solver to
perform a linear regression (by default, it uses the numpy provided linear least squares regression).
The training class provides a simple way to do feature selection over a large feature space.
The trainer does k-fold cross validation to find features that improve validation scores. When complete,
the class has the model coefficients as well as a score.

Dependencies: Python 2.7, numpy

Usage:

    import adaboost
    
    t = adaboost.AdaBoost()
    
    # print detailed debugging information regarding the classifier selection
    t.debug = 2
    
    # train classifier
    t.train(x, y) # x is a matrix, y is a actual classifications (-1 or 1)
    
	# classify novel set of values, the sign of the return value is predicted binary class
    novel_y_prime = t.apply_to_matrix(novel_x)

Methods
-------

The following attributes are available for instances of the Trainer class.

* `train(x, y)` Will begin training on matrix x and classification set y, where y contains
  binary classification data (either 0/1 or -1/1). The system will evaluate potential weak
  classifiers and iteratively add the weak classifier that best minimizes the weighted 
  error.

* `apply_to_matrix(p_x)` Applies the feature selected classifiers to novel values and
  returning a vector with the classification predicted. Each returned classification
  ranges from -1 to 1. The sign is the predicted class, and the absolute value is the 
  confidence.


Attributes
----------

The following attributes are available for instances of the Trainer class.

* `debug` Allows printing of information about the training process. Can be 0 (no 
   debugging), 1 (minimal debugging) or 2 (detailed debugging). Minimal debugging prints 
   final scores and such data, while detailed debugging prints individual classifier 
   additions.

* `max_iterations` The maximum number of weak classifiers to use.

* `target_error` The target error for the training data set. Once the training error is
   less than this value, the algorithm will stop.


