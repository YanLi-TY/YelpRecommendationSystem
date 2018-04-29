# item-based collaborative filtering system

import pandas as pd
import graphlab
import csv
import macros as m


# data preparing
df = pd.read_csv("./input/review_category_user.csv")
df_copy = df
data = df_copy.drop(["Unnamed: 0", "category", "review_id",
                    "stars", "review_useful", "review_funny",
                     "review_cool", "review_count", "user_useful",
                     "user_funny", "user_cool", "user_fans",
                     "user_sum_compliment"
                    ], axis=1)
data.to_csv('data_cfs.csv')

# split training and test dataset
with open("./input/data_cfs.csv", "rb") as f:
    data = f.read().split("\t")
np.random.seed(0)
train_data = data[:int(0.8*(data.shape[0]))]
test_data = data[int(0.8*(data.shape[0])):]

# build item based model
item_based_model = graphlab.item_similarity_recommender.create(train_data, user_id='user_id',
                                                                   item_id='business_id', 
                                                                   target='user_average_stars',
                                                                   similarity_type='pearson')

# predict on test data
pred = item_based_model.predict(test_data)
test_data = test_data.add_column(pred, name='predicted_ave_stars')

# set up threshold = 4
binary_rating = test_data.select_column('user_average_stars').apply(
	                                   lambda x: 0 if (round(x * 2) / 2.0) < 4.0 else 1, dtype=int)
binary_pred = test_data.select_column('predicted_ave_stars').apply(
	                                   lambda x: 0 if (round(x * 2) / 2.0) < 4.0 else 1, dtype=int)

# accuracy, precision, recall
correct_count = 0
true_positive = 0
positive_pred = 0
positive_rating = 0
for i in range(len(binary_rating)):
    if binary_pred[i] == binary_rating[i]:
        correct_count += 1.0
    if binary_rating[i] == 1 and binary_pred[i] == 1:
        true_positive +=  1.0
    if binary_pred[i] == 1:
        positive_pred += 1.0
    if binary_rating[i] == 1:
        positive_rating += 1.0
accuracy = correct_count / len(binary_rating) * 100
precision = true_positive / positive_pred * 100
recall = true_positive / positive_rating * 100

