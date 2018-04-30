import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from collections import defaultdict
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split#,GridSearchCV
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC  
from sklearn import cross_validation,metrics
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, classification_report, roc_curve, auc, accuracy_score  
n_groups = 4
auc_score = (0.532, 0.469, 0.746, 0.549)
ex_auc_score = (0.803, 0.807, 0.651, 0.778)
    # create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.25
opacity = 0.8
rects1 = plt.bar(index, auc_score, bar_width,alpha=opacity,color='b',label='ACC')
rects2 = plt.bar(index + bar_width, ex_auc_score, bar_width,alpha=opacity,color='g',label='Exclude ACC')
plt.xlabel('Feature groups')
plt.ylabel('Accuracy scores')
plt.title('Feature Performance')
plt.xticks(index + bar_width, ('user-feature', 'business-feature', 'user-category', 'review-feature'))
plt.legend()
plt.tight_layout()
plt.show()