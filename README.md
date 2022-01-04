# Random-Forest-KNN-imporvement

This paper describes the working of the combination of two existing Machine learning algorithms to produce a better algorithm to provide better efficiency on the test dataset.


# Motivation
Our basic motivation was to combine Random Forests and concepts of KNN to produce a hybrid algorithm that acts as an improvement to Random Forests. One of the very fundamental algorithms in Machine learning is Decision trees which work well for almost all types of datasets. Random forest is an improvement of decision trees that boosts the accuracy of decision trees by using the concepts of bootstrapping, random subspace, feature selection and bagging. KNN is another algorithm used in machine learning which makes classifications and solves regression problems by finding the closest neighbors to the given test subject based on the distance calculated using a sort of naive mathematical formula. Random forest almost always outperforms KNN due to more complexity in its work.

# Idea
Our idea is to insert a KNN like algorithm at the leaf nodes of Random Forest to improve the accuracy rather than relying totally on pure majority voting. 
The improved algorithm has almost the usability as the standard random forests and could be used in almost all cases where random forests could be used.
The final algorithm was tested on our chosen dataset and the accuracy was also compared to random forests by plotting a graph.

## Link to the official document:
https://docs.google.com/document/d/1bK7dL24KY7DTxjbEuf9mzkuLQnak5L58MbTxFhqLSFs/edit?usp=sharing
