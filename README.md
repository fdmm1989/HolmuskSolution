# HolmuskSolution


The solution for SKD prediction
prepareX.py read the data from scv files and convert it to a numpy file. X refers to the feature version mentioned in the solution.pdf. It also prepare the label numpy file.

gbt_classifier.py performs gbt/rf classification for the input feature. 5-fold cross validation is performed and evaluated by classification_report/AUC and the running time for each fitting process is also recorded. 
svm_classifier.py performs svm classification for the input feature.
NB_classifier.py performs svm classification for the input feature.

