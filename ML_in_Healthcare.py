# !pip install mahotas

# !pip install seaborn

# !pip install seaborn

from matplotlib import pyplot as plt

import mahotas as mh
#import seaborn as sns
#from matplotlib import pyplot as plt 
from glob import glob
import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 

#Classifiers
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay



# !pip install skillsnetwork

# import skillsnetwork

# await skillsnetwork.prepare(
#     "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/data-science-in-health-care-advanced-machine-learning-classification/LargeData/Covid19-dataset.zip",
#     path="Covid19-dataset",
#     overwrite=True
# )

# await skillsnetwork.prepare(
#     "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/data-science-in-health-care-advanced-machine-learning-classification/NN.zip",
#     path="NN",
#     overwrite=True
# )


IMM_SIZE = 224

def get_data(folder):
    
    class_names = [f for f in os.listdir(folder) if not f.startswith('.')] # ctreate a list of SubFolders
    data = []
    # class_names = ['Covid', 'Normal', 'Viral Pneumonia']
    print(class_names)
    for t, f in enumerate(class_names):
        images = glob(folder + "/" + f + "/*") # create a list of files
        print("Downloading: ", f)
        fig = plt.figure(figsize = (50,50)) 
        for im_n, im in enumerate(images):
            plt.gray() # set grey colormap of images
            image = mh.imread(im)
            if len(image.shape) > 2:
                image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE, image.shape[2]]) # resize of RGB and png images
            else:
                image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE]) # resize of grey images    
            if len(image.shape) > 2:
                image = mh.colors.rgb2grey(image[:,:,:3], dtype = np.uint8)  # change of colormap of images alpha chanel delete
            plt.subplot(int(len(images)/5)+1,5,im_n+1) # create a table of images
            plt.imshow(image)
            data.append([image, f])
        plt.show()

    return np.array(data, dtype=object)   

d = "Covid19-dataset/Covid19-dataset/train"
train = get_data(d)

d = "Covid19-dataset/Covid19-dataset/test"
val = get_data(d)

print("Train shape", train.shape) # Size of the training DataSet
print("Test shape", val.shape) # Size of the test DataSet
print("Image size", train[0][0].shape) # Size of image



# %config InlineBackend.figure_formats = ['svg']
import matplotlib.pyplot as plt
import matplotlib
matplotlib.pyplot.rc('text', usetex=True)

import matplotlib.pyplot as plt
from collections import Counter

# Get label counts
label_counts = Counter([i[1] for i in train])

# Extract labels and counts
labels = list(label_counts.keys())
counts = list(label_counts.values())

# Plot using matplotlib
plt.figure(figsize=(8, 6))
plt.bar(labels, counts)
plt.xlabel('Class Labels')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.grid(True, axis='y')
plt.show()


# plt.figure(figsize = (5,5))
# plt.imshow(train[np.where(train[:,1] == 'Viral Pneumonia')[0][0]][0])
# plt.title('Viral Pneumonia')

plt.figure(figsize=(5, 5))

# Find the first image with label 'Viral Pneumonia'
for img, label in train:
    if label == 'Viral Pneumonia':
        plt.imshow(img, cmap='gray')  # Add cmap if image is grayscale
        plt.title('Viral Pneumonia')
        plt.axis('off')
        break

plt.show()


# plt.figure(figsize = (5,5))
# plt.imshow(train[np.where(train[:,1] == 'Covid')[0][0]][0])
# plt.title('Covid')

plt.figure(figsize=(5, 5))

# Loop through to find the first 'Covid' image
for img, label in train:
    if label == 'Covid':
        plt.imshow(img, cmap='gray')  # Remove cmap='gray' if image is in color
        plt.title('Covid')
        plt.axis('off')
        break

plt.show()



def create_features(data):
    features = []
    labels = []
    for image, label in data:
        features.append(mh.features.haralick(image).ravel())
        labels.append(label)
    features = np.array(features)
    labels = np.array(labels)
    return (features, labels) 

features_train, labels_train = create_features(train)
features_test, labels_test = create_features(val)

clf = Pipeline([('preproc', StandardScaler()), ('classifier', LogisticRegression())])
clf.fit(features_train, labels_train)
scores_train = clf.score(features_train, labels_train)
scores_test = clf.score(features_test, labels_test)
print('Training DataSet accuracy: {: .1%}'.format(scores_train), 'Test DataSet accuracy: {: .1%}'.format(scores_test))
ConfusionMatrixDisplay.from_estimator(clf, features_test, labels_test)
plt.show() 

names = ["Logistic Regression", "Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(reg_param=0.1)]
scores_train = []
scores_test = []
for name, clf in zip(names, classifiers):
    print("Fitting:", name)
    clf = Pipeline([('preproc', StandardScaler()), ('classifier', clf)])
    clf.fit(features_train, labels_train)
    score_train = clf.score(features_train, labels_train)
    score_test = clf.score(features_test, labels_test)
    scores_train.append(score_train)
    scores_test.append(score_test)

res = pd.DataFrame(index = names)
res['scores_train'] = scores_train
res['scores_test'] = scores_test
res.columns = ['Test','Train']
res.index.name = "Classifier accuracy"
pd.options.display.float_format = '{:,.2f}'.format
print(res)

x = np.arange(len(names))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, scores_train, width, label='Train')
rects2 = ax.bar(x + width/2, scores_test, width, label='Test')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy of classifiers')
ax.set_xticks(x)
plt.xticks(rotation = 90)
ax.set_xticklabels(names)
ax.legend()

fig.tight_layout()

plt.show()