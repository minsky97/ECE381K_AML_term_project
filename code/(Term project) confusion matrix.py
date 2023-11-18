import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(conf_matrix, classes):
    plt.figure(figsize=(6, 6))
    sns.set(font_scale=2.2)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# Values for the best performance cases
tp = [58,49,204,176]
fp = [2,0,0,0]
fn = [2,1,0,2]
tn = [58,50,204,178]

# Create confusion matrix for the best cases
for i in range(0,4):
    conf_matrix = np.array([[tp[i], fp[i]], [fn[i], tn[i]]])
    class_labels = ["Positive", "Negative"]
    plot_confusion_matrix(conf_matrix, class_labels)
    
# Values for the worst performance cases
tp = [35,49,139,40]
fp = [16,8,65,31]
fn = [25,23,65,5]
tn = [44,64,139,14]

# Create confusion matrix for the worst cases
for i in range(0,4):
    conf_matrix = np.array([[tp[i], fp[i]], [fn[i], tn[i]]])
    class_labels = ["Positive", "Negative"]
    plot_confusion_matrix(conf_matrix, class_labels)