# Motivation

Cardiovascular diseases (CVDs) stand as the foremost cause of mortality worldwide, claiming approximately 17.9 million lives annually. This category encompasses a spectrum of disorders affecting the heart and blood vessels, such as coronary heart disease, cerebrovascular disease, rheumatic heart disease, among others. Alarmingly, more than four out of five CVD deaths are due to heart attacks and strokes, and one third of these deaths occur prematurely in people under 70 years of age \[1].

The use of machine learning methods can help to identify possible incidences of the disease in advance, greatly aiding rapid and better medical intervention. This can be clearly seen in the articles \[2-7].

In this work we explore not only the application of different machine learning methods to a dataset of cardiovascular patients, but also the interpretability of the methods, which is essential to increase the credibility of the model and its potential practical application. A good article on Explainable Artificial Intelligence (XAI) can be seen here \[8] 

---
# Dataset

The dataset was obtained from the [IEEEDataPort database](https://ieee-dataport.org/open-access/heart-disease-dataset-comprehensive). It has 11 features and 1190 instances. It is almost balanced with 47.1% pations without heart disease and 52.9% patients with disease.

**Data statistical analysis**

| **Variable**                   | **With disease (n=507)** | **Without disease (n=410)** | **P-value** |
| ------------------------------ | ------------------------ | --------------------------- | ----------- |
| Age                            | 55.876 ± 8.7             | 50.551 ± 9.4                | <0.001      |
| Sex (male)                     | 457 (90%)                | 267 (65%)                   | <0.001      |
| chest pain type (asymptomatic) | 392 (77%)                | 149 (25%)                   | <0.001      |
| resting bp s                   | 134.154 ± 19.8           | 130.180 ± 16.5              | 0.001       |
| Cholesterol                    | 250.091 ± 52.9           | 238.425 ± 54.1              | 0.001       |
| fasting blood sugar (false)    | 338 (68%)                | 366 (89%)                   | <0.001      |
| resting ecg (normal)           | 284 (56%)                | 267 (65%)                   | 0.004       |
| max heart rate                 | 127.647 ± 23.4           | 148.151 ± 23.3              | <0.001      |
| exercise angina (yes)          | 316 (62%)                | 55 (13%)                    | <0.001      |
| oldpeak                        | 1.277 ± 1.2              | 0.408 ± 0.7                 | <0.001      |
| ST slope (flat)                | 380 (75%)                | 79 (19%)                    | <0.001      |

note: T-test in numeric variables and chi square for nominal and binary features.

---
# Pre-processing

1. imputation of zero cholesterol values

```python
# Regression model to fill invalid cholesterol data

from sklearn.ensemble import RandomForestRegressor

cholesterol_model = RandomForestRegressor(n_estimators=40, min_samples_leaf=3,
                                                   max_features=0.5,
                                                   n_jobs=-1,
                                                   oob_score=True,max_depth=12)

cholesterol_x= cholesterol_train_frame.drop('cholesterol',axis=1)
cholesterol_y= cholesterol_train_frame.cholesterol.values
cholesterol_model.fit(cholesterol_x, cholesterol_y)
cholesterol_prediction['cholesterol']=cholesterol_model.predict(cholesterol_prediction.drop('cholesterol',axis=1))

dataset=cholesterol_train_frame.append(cholesterol_prediction)
sns.scatterplot(x = 'cholesterol', y = 'age', hue = 'target', data = dataset)
```

![imput cholesterol](https://i.imgur.com/TWGXRE4.png)


2. Removing duplicated data

```python
#Verification of duplicated data
print('Duplicated dataset: ',dataset.shape)
dataset.drop_duplicates(inplace=True)
print('Dataset after drop duplicates: ',dataset.shape)
```

Duplicated dataset:  (1189, 12)
Dataset after drop duplicates:  (917, 12)


3. Creating a unseen data set (or test set)

Usually, a part of the data is separated in order to carry out the final validation and assess the generalisation capacity of our algorithm. This was done in this step.

![train test cut](https://i.imgur.com/PupS6ME.png)

```python
data = dataset.sample(frac=0.95, random_state=786)
data_unseen = dataset.drop(data.index)

data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions ' + str(data_unseen.shape))
```

Data for Modeling: (871, 12)
Unseen Data For Predictions (46, 12)

---
# Training and evaluation

PyCaret was used to train the different machine learning algorithms. PyCaret is a comprehensive, open-source, low-code machine learning library in Python designed to simplify the end-to-end process of building, training, evaluating, and deploying machine learning models. It offers a wide range of functionalities including automated data preprocessing, feature selection, model comparison, hyperparameter tuning, ensemble methods, and model interpretation, making it an ideal choice for both beginners and experienced data scientists looking to streamline their workflow and accelerate model development \[9].

```python
exp_clf102 = setup(data = data, target = 'target',
                  normalize = True, 
                  transformation = True, 
                  ignore_low_variance = True,
                  remove_outliers = True,
                  fix_imbalance = True,
                  remove_multicollinearity = True,
                  log_experiment = True, 
                  experiment_name = 'heartdisease')
```

**Some explanation:**
- **ignore_low_variance:** When set to True, all categorical features with statistically insignificant variances are removed from the dataset.
- **remove_multicolinearity:** Variables that have a high dependence between each other will be removed.
- **transformation:** Will transform data to have a more linear distribution.

Our approach was a three-way holdout method, explained in detail on Raschka et al. work \[10]. Similar in some ways to the picture below:

![holdout method](https://i.imgur.com/TyWHBzu.png)

We selected the three most effective models based on their performance with the training data. These were fine-tuned by experimenting with various settings using different tuning libraries and algorithms. Specifically, the Scikit-Learn library with the RandomizedSearchCV algorithm, the Optuna library with OptunaSearchCV, and the Scikit-optimize library with the BayesSearchCV algorithm were used in order to optimize each classifier's performance.

- **RandomizedSearchCV:** it randomly selects combinations of hyperparameters from a predefined range to efficiently explore the hyperparameter space and find the optimal configuration for a given model.
- **OptunaSearchCV:** employs the Optuna framework, utilizing an efficient algorithm to iteratively explore the hyperparameter space and find the optimal configuration for a given model by maximizing a chosen objective function.
- **BayesSearchCV:** uses Bayesian optimization principles to efficiently search through the hyperparameter space, gradually learning which configurations are more likely to yield better results while balancing exploration and exploitation.

---
# Results

All the following models were trained and evaluated:
- Catboost
- Light gradient boosting
- Extreme gradient boosting
- Gradient boosting
- Extra Trees
- Random Forest
- Ada Boost
- Decision Tree
- Linear Discriminant analysis
- Logistic Regression
- Ridge
- SVM – Linear kernel
- K nearest neighbors
- Quadratic discriminant analysis
- Naive Bayes

<ins>The best model was RF, that achieved the best results using a Bayesian optimization technique - BayesSearchCV.</ins>

**Sensitivity on train and test:**

|               | Train | Test  |
| ------------- | ----- | ----- |
| Random Forest | 0.911 | 0.884 |

**Final result on unseen data: (after retraining on both train and test data)**

|                     | **Accuracy** | **AUC** | **Precision** | **Sensitivity** | **F1 Score** |
| ------------------- | ------------ | ------- | ------------- | --------------- | ------------ |
| Tuned Random Forest | 0.9565       | 0.9754  | 0.9231        | 1               | 0.9600       |

More details on code: [https://github.com/franciscomesquitaAI/Hearth-Disease-Interpretability-Research/](https://github.com/franciscomesquitaAI/Hearth-Disease-Interpretability-Research/)

---
# Interpretability

SHAP (SHapley Additive exPlanations) is a powerful technique for explaining the output of machine learning models. It provides a unified framework for understanding the contribution of each feature to the prediction made by the model. SHAP values are based on cooperative game theory, specifically the Shapley value, which is a concept used to fairly distribute the payout among players in a cooperative game .

<ins>Here's a breakdown of how SHAP works and how it can be used to interpret ML models:</ins>

- **Individual/Local Interpretability:** SHAP values offer local interpretability by explaining why a particular prediction was made for a specific instance. This can be particularly useful for debugging and understanding individual model predictions.

- **Global Explanations:** SHAP can provide insights into the overall behavior of the model by analyzing the average impact of each feature across all predictions.

- **Model-Agnostic:** One of the key advantages of SHAP is its model-agnostic nature. It can be applied to any machine learning model, whether it's a decision tree, random forest, gradient boosting machine, neural network, or any other model.

- **Visualizations:** SHAP values can be visualized in various ways to aid interpretation, such as summary plots, dependence plots, and force plots. These visualizations help users understand how individual features affect predictions and how feature values interact with each other.

![Shap Method](https://i.imgur.com/juqZKAj.png)

**To see the results in detail and interpretability graphs look at the jupyter notebook file at:** [https://github.com/franciscomesquitaAI/Hearth-Disease-Interpretability-Research/blob/main/%5BMost_Recent%5D_Reseach_Hearth_Disease_study_ipynb_txt.ipynb](https://github.com/franciscomesquitaAI/Hearth-Disease-Interpretability-Research/blob/main/%5BMost_Recent%5D_Reseach_Hearth_Disease_study_ipynb_txt.ipynb)

---
# Conclusion

With this work, we found out that we can build a model that works very well and also understand why it makes certain predictions using the SHAP method. When we were making the model, we tried out different ways to fine-tune it for better performance. We found that the algorithm called OptunaSearchCV worked really well for Catboost and ET models, but for our main model, RF, a method called BayesSearchCV gave us the best results. We developed a model that consistently performed well and we were also able to understand the reasons behind its predictions.

## Strengths
- Creation of a ML model capable of predicting heart disease with a very good performance
- Interpretable model, providing all the information needed to understand how the model arrived at a given prediction. This makes the process much more transparent, paving the way for its potential practical application.
- Model validated and evaluated in three different phases (training, validation, unseen data). In addition, different models and hyperparameter tuning algorithms were tested and compared.

## Limitations
- In the preprocessing phase, other procedures and methods could be explored and tested. 
- The selection of the ML models was not made by any specific criteria such as the size of the data or the representativeness of the sample itself.
- It is not possible to guarantee that the performance will be maintained when applied in a real context because only one dataset was used here.

## Future work
- Adding more data from different sources, in order to create more diversity and thus guarantee greater confidence in the results and the classifier's ability to generalize.
- Make the model available online using MLaas, using solutions like MLCapsule \[11].

---
# References
\[1] : [https://www.who.int/health-topics/cardiovascular-diseases](https://www.who.int/health-topics/cardiovascular-diseases)

\[2]: [https://www.sciencedirect.com/science/article/abs/pii/S0933365722000549](https://www.sciencedirect.com/science/article/abs/pii/S0933365722000549)

\[3]: [https://ieeexplore.ieee.org/abstract/document/10306469](https://ieeexplore.ieee.org/abstract/document/10306469)

\[4]: [https://jpmm.um.edu.my/index.php/MJCS/article/view/35980](https://jpmm.um.edu.my/index.php/MJCS/article/view/35980)

\[5]: [https://iopscience.iop.org/article/10.1088/1757-899X/1022/1/012072/meta](https://iopscience.iop.org/article/10.1088/1757-899X/1022/1/012072/meta)

\[6]: [https://journals.sagepub.com/doi/full/10.1177/1179546820927404](https://journals.sagepub.com/doi/full/10.1177/1179546820927404)

\[7]: [https://ieeexplore.ieee.org/abstract/document/10087837](https://ieeexplore.ieee.org/abstract/document/10087837)

\[8]: [https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.1424](https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.1424)

\[9]: [https://pycaret.gitbook.io/docs](https://pycaret.gitbook.io/docs)

\[10]: [https://arxiv.org/abs/1811.12808](https://arxiv.org/abs/1811.12808)

\[11]: [https://arxiv.org/abs/1808.00590](https://arxiv.org/abs/1808.00590)
