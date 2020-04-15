# Investigation of Best Algorithm to Use

## Methods
We used the training dataset for investigating the best algorithm to use for our
final results. The training dataset was split by 10 fold K-fold validation by
patients, and there was no overlap of patient data between training and testing sets
in a single fold. We performed this process first with a grid search to find optimal
parameterization of our models. The best parameterization that we found is marked and
then results for the algorithm by running it on the train set in KFold is reported.

### Neural Net
Best params {'activation': 'tanh', 'learning_rate_init': 0.01, 'algorithm': 'adam', 'hidden_layer_sizes': [64]}

```
+-------+----------+-------------+-------------+-----------+-----------+----------+-------------+------------+
| label | f1-score | sensitivity | specificity | precision | train_len | test_len | n_train_pts | n_test_pts |
+-------+----------+-------------+-------------+-----------+-----------+----------+-------------+------------+
|  VC   |   0.98   |    0.996    |    0.981    |   0.964   |    nan    |   nan    |     nan     |    nan     |
|  PC   |  0.906   |     0.9     |    0.972    |   0.913   |    nan    |   nan    |     nan     |    nan     |
|  PS   |  0.948   |    0.927    |    0.982    |   0.969   |    nan    |   nan    |     nan     |    nan     |
|  CPAP |  0.984   |    0.977    |    0.999    |   0.992   |    nan    |   nan    |     nan     |    nan     |
|  PAV  |  0.983   |    0.975    |    0.998    |   0.992   |    nan    |   nan    |     nan     |    nan     |
+-------+----------+-------------+-------------+-----------+-----------+----------+-------------+------------+
```

### SVM
Best params {'kernel': 'rbf', 'C': 10}

```
+-------+----------+-------------+-------------+-----------+-----------+----------+-------------+------------+
| label | f1-score | sensitivity | specificity | precision | train_len | test_len | n_train_pts | n_test_pts |
+-------+----------+-------------+-------------+-----------+-----------+----------+-------------+------------+
|  VC   |  0.984   |    0.994    |    0.984    |   0.974   |    nan    |   nan    |     nan     |    nan     |
|  PC   |  0.927   |    0.891    |    0.989    |   0.967   |    nan    |   nan    |     nan     |    nan     |
|  PS   |  0.947   |    0.958    |    0.976    |   0.937   |    nan    |   nan    |     nan     |    nan     |
|  CPAP |  0.971   |    0.954    |    0.999    |   0.988   |    nan    |   nan    |     nan     |    nan     |
|  PAV  |  0.967   |    0.993    |    0.984    |   0.943   |    nan    |   nan    |     nan     |    nan     |
+-------+----------+-------------+-------------+-----------+-----------+----------+-------------+------------+
```

### Logistic Regression
Best params: {'penalty': 'l2', 'C': 4, 'max_iter': 100, 'tol': 0.0001, 'solver': 'lbfgs’}

```
+-------+----------+-------------+-------------+-----------+-----------+----------+-------------+------------+
| label | f1-score | sensitivity | specificity | precision | train_len | test_len | n_train_pts | n_test_pts |
+-------+----------+-------------+-------------+-----------+-----------+----------+-------------+------------+
|  VC   |  0.977   |    0.998    |    0.958    |   0.956   |    nan    |   nan    |      nan     |     nan      |
|  PC   |  0.603   |    0.536    |    0.908    |    0.69   |    nan    |   nan    |      nan     |     nan      |
|  PS   |  0.734   |    0.785    |    0.869    |    0.69   |    nan    |   nan    |      nan     |     nan      |
|  CPAP |   0.99   |    0.985    |    0.999    |   0.995   |    nan    |   nan    |      nan     |     nan      |
|  PAV  |  0.991   |    0.995    |    0.996    |   0.988   |    nan    |   nan    |      nan     |     nan      |
+-------+----------+-------------+-------------+-----------+-----------+----------+-------------+------------+
```

### Random Forest
best params: {'max_features': 'auto', 'n_estimators': 30, 'max_depth': 15, 'criterion': 'entropy', 'random_state': 1}

```
+-------+----------+-------------+-------------+-----------+-----------+----------+-------------+------------+
| label | f1-score | sensitivity | specificity | precision | train_len | test_len | n_train_pts | n_test_pts |
+-------+----------+-------------+-------------+-----------+-----------+----------+-------------+------------+
|  VC   |  0.998   |     1.0     |    0.998    |   0.997   |    nan    |   nan    |     nan     |    nan     |
|  PC   |  0.987   |    0.976    |    0.999    |   0.998   |    nan    |   nan    |     nan     |    nan     |
|  PS   |  0.985   |    0.993    |    0.992    |   0.978   |    nan    |   nan    |     nan     |    nan     |
|  CPAP |  0.996   |    0.994    |     1.0     |   0.998   |    nan    |   nan    |     nan     |    nan     |
|  PAV  |  0.994   |    0.995    |    0.998    |   0.993   |    nan    |   nan    |     nan     |    nan     |
+-------+----------+-------------+-------------+-----------+-----------+----------+-------------+------------+
```

### LSTM
Architecture: 1 LSTM layer with 32 hidden units, followed by a fully connected layer and softmax

Hyperparams: Epochs - 10, Learning Rate - .001, Batch Size - 32.

```
              precision    recall  f1-score   support

          VC       0.99      0.99      0.99     24030
          PC       0.94      0.96      0.95     48507
          PS       0.96      0.94      0.95     56794
          CPAP     0.91      0.90      0.90      4738
          PAV      0.97      0.99      0.98     17483

avg / total        0.96      0.96      0.96    151552
```
