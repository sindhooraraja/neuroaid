import os
import cv2
import numpy as np
from skimage import feature
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import pickle
import matplotlib.pyplot as plt

def quantify_image(image):
    features = feature.hog(image, orientations=9,
                           pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                           transform_sqrt=True, block_norm="L1")
    return features

def load_test_data(path):
    imagePaths = list(paths.list_images(path))
    data = []
    labels = []
    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))
        image = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        features = quantify_image(image)
        data.append(features)
        labels.append(label)
    return np.array(data), np.array(labels)

def test_model(model, testX, testY):
    predictions = model["classifier"].predict(testX)
    cm = confusion_matrix(testY, predictions).ravel()
    tn, fp, fn, tp = cm
    accuracy = (tp + tn) / float(cm.sum())
    sensitivity = tp / float(tp + fn)
    specificity = tn / float(tn + fp)
    return accuracy, sensitivity, specificity, cm

# Load the trained model
spiralModel = pickle.load(open("spiralModel.pkl", "rb"))

# Define the path to the testing directory
testPath = r"D:\\GyanMela\\PD\\spiraltest\\spiral\\test"

# Load test data
(testX, testY) = load_test_data(testPath)

# Encode labels
le = LabelEncoder()
testY = le.fit_transform(testY)

# Test the model on the new data
accuracy, sensitivity, specificity, cm = test_model(spiralModel, testX, testY)

# Print the results
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Sensitivity: {:.2f}%".format(sensitivity * 100))
print("Specificity: {:.2f}%".format(specificity * 100))

# Plot Confusion Matrix
class_names = le.classes_
disp = plot_confusion_matrix(spiralModel["classifier"], testX, testY,
                             display_labels=class_names,
                             cmap=plt.cm.Blues,
                             normalize=None)
disp.ax_.set_title('Confusion Matrix')
plt.show()
