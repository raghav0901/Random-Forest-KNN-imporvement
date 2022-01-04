 
# Random Forest Algorithm 
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import math
from sklearn.feature_selection import chi2
import numpy as np
import pandas as pd
featureImp=None
 
# Load a CSV file into a List(dataset).
def load_csv(filename):
    dataset = list()	
    with open(filename, 'r') as file:
        file_reader = reader(file)
        for row in file_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Split a dataset into k number of folds
def cross_validation_split(dataset, num_of_folds):
    dataset_split = list()
    dataset_temp_copy = list(dataset)
    max_fold_size = int(len(dataset) / num_of_folds)

    for x in range(num_of_folds):

        new_fold = list()
        while len(new_fold) < max_fold_size:
            random_index = randrange(len(dataset_temp_copy))
            new_fold.append(dataset_temp_copy.pop(random_index))
            
        dataset_split.append(new_fold)
    return dataset_split
 
# Calculate accuracy of algorithm comparing actual and predicted values.
def accuracy_metric(actual, predicted):
    num_of_correct_values = 0
    total_num_of_values = len(actual)

    for i in range(total_num_of_values):
        if actual[i] == predicted[i]:
            num_of_correct_values += 1

    return num_of_correct_values / float(total_num_of_values) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, num_of_folds, *args):
    folds = cross_validation_split(dataset, num_of_folds)
    scores = list()

    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()

        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None

        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]

        accuracy = accuracy_metric(actual, predicted)

        scores.append(accuracy)

    return scores
 
# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()

    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)

    return left, right
 
# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        group_size = float(len(group))
        # avoid divide by zero
        if group_size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / group_size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (group_size / n_instances)
    return gini
 
# Select the best split point for a dataset
def get_split(dataset, n_features):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    len_of_first_element = len(dataset[0])

    while len(features) < n_features:
        index = randrange(len_of_first_element - 1)
        if index not in features:
            features.append(index)

    for index in features:
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)

            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups

    return {'index':b_index, 'value':b_value, 'groups':b_groups}
 
# Create a terminal node value
def to_terminal(group):
    return group
    #outcomes = [row[-1] for row in group]
    #return max(set(outcomes), key=outcomes.count)
 
# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return

    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth+1)

    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth+1)
 
# Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root


## Using a CHI - SQUARE test to calculate feautre importance:
def featureImportance(some):
    global featureImp
    new_some=some.copy()
    X = new_some.drop(new_some.columns[len(some.columns) - 1],axis=1)
    y = new_some.iloc[:,len(some.columns) - 1]
    chi_scores = chi2(X,y)
    featureImp=chi_scores[0]

    print("Feature Ratio:")
    print(featureImp)

## KNN insertion at leaf nodes:
def doKnn(neighbors,row):
    ## Calculating the feature ranges:
    feature_ranges = list()
    num_of_features = len(row) - 1
    
    for each_feature in range(num_of_features):
        list_vals = list()

        for column in dataset:
            list_vals.append(int(float(column[each_feature])))

        feature_ranges.append( max(list_vals) - min(list_vals) )

    ## Calcualting the distance to the neighbors in leaf node
    score = {}
    i=0

    for x in neighbors:
        neighbor_dist=0
        for y in range(num_of_features):          
            neighbor_dist+=(abs(int(float(x[y]))-int(float(row[y])))/feature_ranges[y])*featureImp[y]/(featureImp.sum())
        score[i]=neighbor_dist
        i=i+1
    score_list=sorted(score.items(), key=lambda x:x[1])
    score=dict(score_list)
    ## performing KNN with 50% cutoff threshold:
    threshold=math.floor(len(score.keys())/2)
    list_eligible_neighbors=[]
    if threshold==0:
        threshold=1
    for n in range(threshold):
         list_eligible_neighbors.append(neighbors[list(score.keys())[n]])
    outcomes = [row[-1] for row in list_eligible_neighbors]
    return max(set(outcomes), key=outcomes.count)

# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return doKnn(node['left'],row)
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return doKnn(node['right'],row)
 
# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)
 
# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample

# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return(predictions)
 
# Test the random forest algorithm
seed(2)
# load and prepare data
dataset1 = 'new_hybrid.csv'
dataset2 = "new_processed.cleveland.data"

#change fine name here
currentFile = dataset1

dataset = load_csv(currentFile)
# removing header
dataset.remove(dataset[0])
# convert string attributes to integers 
print("Making the Dataframe.")
df = pd.read_csv(currentFile)
print("Calcualting feature importances")
featureImportance(df)

# evaluate algorithm
num_of_folds = 5
max_depth = 10
min_size = 1
sample_size = 1.0
n_features = int(sqrt(len(dataset[0])-1))
for n_trees in [1, 5, 10]:
    scores = evaluate_algorithm(dataset, random_forest, num_of_folds, max_depth, min_size, sample_size, n_trees, n_features)
    print('Trees: %d' % n_trees)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

