# Neural Networks

Experimentation with neural networks is broadly conducted through the following approaches:

1) Architecture: Varied configurations involving different numbers of layers and neurons in each layer.
2) Activation function: Exploration of functions such as ReLU, Sigmoid, TanH, etc.
3) Random initialization of weights.
4) Optimization algorithms: Testing with RMSProp, SGD, Adam, among others.
5) Regularization: Comparison of models with and without dropout, and with and without batch normalization.
6) Data experimentation: Exploring oversampling and undersampling techniques.
7) Transfer learning: Initially training on data with only two classes and subsequently training the same model on the original dataset.
8) Hyperparameter tuning.
9) Neural Network Ensemble: Combining predictions from multiple neural network models.


## Feature Engineering
The feature engineering for neural networks is similar to that of SVM as done previously. <br>
In case of SVMs, we had utilized top 15 features for training and evaluation, but in the case of neural networks, maximum accuracy was achieved while considering top 24 features.

```bash
Before dropping less important columns: 
Training set: (56988, 72) 
Validation set: (14248, 72) 

After dropping less important columns: 
Training set: (56988, 24) 
Validation set: (14248, 24)
```

## Hyper-parameter tuning:
```bash
{'num_layers': 2,
 'layer_units_0': 32,
 'layer_units_1': 10,
 'dropout': 0.23657712326928104,
 'lr': 0.0018333742114024943,
 'weight_decay': 9.976104578576312e-06,
 'batch_size': 19,
 'activation_function': 'ReLU',
 'n_epochs': 17
 }
 ```
### Number of layers 
The number of hidden layers in the network was suggested as an integer between 1 and 3. This is because we wanted to prevent overfitting of data that might occur if we use more number of hidden layers in the neural network. Otherwise, we would have high training accuracy but low test accuracy, which is undesirable. <br>
On hyper-parameter tuning, the number of layers comes out to be 2.

### Number of nodes in each layer
It can be defined as the number of neurons in each hidden layer. More neurons allow each layer to process more information and potentially learn more complex features, but since our data has rather simpler features, more number of neurons are not suggested as it might lead to overfitting or slower training. <br>
On hyper-parameter tuning, the number of layer units for the 2 hidden layers come out to be 32 and 10 respectively.

### Dropout
Dropout randomly sets some network neurons to 0 during training, forcing other neurons to compensate and learn more robust features. This reduces overfitting by preventing co-dependency between neurons. <br>
On hyper-parameter tuning, the dropout is roughly 0.236 which signifies that 23.6% of the neurons in the layer will be randomly deactivated and set to 0 for each training iteration.

### Learning Rate
Learning rate controls the step size for updating the network's weights during training. A higher learning rate takes larger steps towards minima, potentially reaching them faster but risking overshooting and missing the optimal solution. <br>
On hyper-parameter tuning, the learning rate comes out to be 0.00183 which is roughly 10e-3.

### Weight decay
Weight decay penalizes the L2 norm of the weights, encouraging sparsity and preventing overfitting by discouraging large weight values. <br>
On hyper-parameter tuning, the weight decay roughly comes out to be 9.97e-06.

### Batch size
Batch data defines the number of data samples processed together during training. Larger batches utilize hardware more efficiently but can lead to slower updates and unstable gradients. <br>
On hyper-parameter tuning, the weight decay roughly comes out to be 19.

### Activation function
Activation function introduces non-linearity into the network, allowing it to learn complex relationships between features. Different activation functions have varying properties and impact gradients. <br> 
We had experimented with different activation functions like ReLU, LeakyReLU, and Sigmoid, but ReLu is the most optimal activation function.

### Epochs

On trial for optimal number of epochs between 10 and 50, it was found out that 17 is the optimal number.

## Results:

During model training and evaluation without any hyper-parameter tuning, we had hard-coded the hyper-parameters based on repeated experimentation. We had used 2 hidden layers in which the layers consist of 100 and 20 neurons respectively. <br>
The number of epochs were 25. <br>
Activation function utilized was ReLU. <br>
Dropout=0.2 <br>

For hyper-parameter-tuning, the optimal parameters have already been shown previously.
1. Without Hyper-parameter tuning: <br>
a) Time taken for training the model: 136 seconds <br>
b) Best test accuracy and f1 score:Test acc:  0.72284  and 0.50761 <br>
c) Best f1 score on Kaggle: 0.722 <br>

2. With Hyper-parameter tuning: <br>
a) Time taken for training the model: approx. 15 minutes <br>
b) Best test accuracy and f1 score: <br>
c) Best f1 score on Kaggle: <br>


## Experiments with initialization of weights:

All the conducted experiments utilized default weight initialization. Consequently, we decided to explore various neural network initializations, each of which is briefly explained below. Despite multiple attempts due to a significant accuracy drop from 72.2%, we couldn't understand the cause.

1) **Xavier Uniform Initialization**: Xavier (Glorot) Uniform Initialization initializes weights from a uniform distribution based on the layer's input and output units. This method aims to prevent issues like vanishing or exploding gradients, contributing to stable training.

   `Results`: Minimal improvement; in fact, the accuracy score dropped. Notably, it exhibited an overemphasis on predicting 0's, leading to the decline in accuracy.

2) **Xavier Normal Initialization**: Xavier (Glorot) Normal Initialization draws weights from a normal distribution with a mean of 0 and a variance determined by input and output units. It addresses challenges like vanishing or exploding gradients.

   `Results`: Limited improvement; accuracy score decreased. Similar to Xavier Uniform, it exhibited an aggressive prediction of 0's and performed poorly with 2's.

3) **He Uniform Initialization**: He Normal Initialization draws initial weights from a normal distribution with a variance determined by input units. It is well-suited for deep networks, addressing issues like vanishing or exploding gradients.

   `Results`: Limited improvement; accuracy score dropped. Notably, it exhibited an aggressive prediction of 1's.

4) **He Normal Initialization**: He Uniform Initialization initializes weights from a uniform distribution based on the layer's input units. It is designed for ReLU activation functions, addressing issues like vanishing or exploding gradients.

   `Results`: Minimal improvement; accuracy score dropped. It predicted 0's less frequently than others.

Despite these efforts, achieving a significant improvement in accuracy remained challenging, and certain patterns, such as the aggressive prediction of specific classes, persisted across different initialization methods.

## Experimentation with data (oversampling and undersampling)

Similar to the midsem, we conducted experiments with the SMOTE model, involving oversampling and undersampling techniques to address the lower number of data samples for class 0 in the training data. Two specific experiments were conducted using this strategy:

1) Undersampling of class 2 to 30,000 (from approximately 34,000 datapoints) and oversampling of class 0 to 15,000 (from around 8,000 samples). Unfortunately, this led to a deterioration in accuracy to 33% on the validation data and performed similarly on the Kaggle dataset.

2) Oversampling of class 0 to 25,000 (from around 8,000 datapoints). Although it exhibited poor performance on the training data, it outperformed Experiment 1 with a 44% accuracy on the Kaggle dataset.

Due to the significant decline in performance and considering that predicting all instances as class 2 already yields a score of 54%, we decided to discontinue this experiment for evident reasons.

## Experimentation with Neural Ensemble methods

We implemented an ensemble of 10 neural networks, each trained on a randomly sampled fraction (50%) of the original data. The predictions of each neural network were then multiplied by weights, and similar predictions were aggregated. We assigned weights proportionally based on the performance of each model on its training data. The prediction with the highest weightage was utilized. Remarkably, this ensemble model exhibited strong performance compared to other experiments, achieving a Kaggle dataset accuracy of 68.4%. Given time constraints and the resource-intensive nature of this task, we conducted only two experiments by adjusting the weights. We believe that further testing and hyperparameter tuning for this ensemble could potentially yield results exceeding 70%.

## Experimentation with Transfer Learning



# Non Neural Ensemble methods

Given that the midsem data was identical to the current dataset, we conducted preliminary experiments with ensemble methods, including bagging and boosting, during the first phase(pre-midterm) of the project. The details of these experiments are outlined below. In light of the introduction of SVM in the second part of the project, our focus shifted to combining SVM with RFC and Logistic Regression, both of which demonstrated reliability in the first phase.

#### Experiments conducted before midsem:

1) **XGBBoostClassifier()**
2) **CatBoostClassifier()**
3) **GradientBoostingClassifier()**
4) **RandomForestClassifier()**
4) Converted the problem into three separate classification tasks. Utilized Logistic Regression and Bayesian models, predicting the final class by selecting the one with the maximum probability among the individual models.
5) Employed the same architecture as above, but instead of using the maximum probability, applied a (RandomForest/Multi-Class Logistic Regression) on the probabilities of each of the three classifiers. However, these two methods struggled to effectively capture class 0 or performed well with class 0 but faced challenges with class 1.
6) Recognizing the inefficiency in capturing class 0, a logistic regression was employed to predict for class 0 (Yes/No). Subsequently, a separate RandomForestClassifier was trained to handle class (1 and 2), predicting 0 if the probability of class 0 exceeded a certain threshold; otherwise, the output from RandomForestClassifier was considered. This approach performed nearly on par with the model, although it exhibited challenges in predicting class 1.

#### Experiment with Ensemble of SVM, RFC and Logistic Regression

Initially, our model comprised four SVMs, two RFCs, and one LR. However, LR demonstrated greater reliability, particularly in aggressively predicting 0's. Consequently, we adjusted our ensemble to include three SVMs, two RFCs, and two LR, leveraging LR's enhanced predictive capabilities. Similar to the Neural Ensemble method, we assigned weights based on the models' reliability in predicting the dataset. All models were trained on an 80% random sample of the original dataset. Once again, the ensemble exhibited comparable performance to the neural network ensemble, achieving an accuracy of 66.8% on the Kaggle dataset.

