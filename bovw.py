import cv2
import os
import numpy as np
from scipy.cluster.vq import kmeans, vq

train_path = 'dataset_car_logo/Train'

train_names = os.listdir(train_path)

# Get path to all the images and save them.


img_paths = []
img_classes = []

class_id = 0

# Utility function to get all the filenames.
def imglist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]



# Add all the paths into one list along with their ids
for train_name in train_names:
    direc = os.path.join(train_path, train_name)
    class_path = imglist(direc)
    img_paths += class_path
    img_classes += [class_id] * len(class_path)
    class_id += 1


# Initialize sift object
sift = cv2.SIFT_create()
des_list = []

for img_path in img_paths:
    im = cv2.imread(img_path)
    kpts, des = sift.detectAndCompute(im, None)
    des_list.append((img_path, des))


descriptors = des_list[0][1]

for img_path, descriptor in des_list:
    descriptors = np.vstack((descriptors, descriptor))


descriptors_float = descriptors.astype(float)

k = 200
voc, variance = kmeans(descriptors_float, k, 1)
im_features = np.zeros((len(img_paths), k), "float32")
for i in range(len(img_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(img_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
#Standardize features by removing the mean and scaling to unit variance
#In a way normalization
from sklearn.preprocessing import StandardScaler
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

#Train an algorithm to discriminate vectors corresponding to positive and negative training images
# Train the Linear SVM
from sklearn.svm import LinearSVC
clf = LinearSVC(max_iter=10000)  #Default of 100 is not converging
clf.fit(im_features, np.array(img_classes))

#Train Random forest to compare how it does against SVM
#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators = 100, random_state=30)
#clf.fit(im_features, np.array(img_classes))


# Save the SVM
#Joblib dumps Python object into one file
from sklearn.externals import joblib
joblib.dump((clf, training_names, stdSlr, k, voc), "bovw.pkl", compress=3)
