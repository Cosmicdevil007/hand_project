import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from model import KeyPointClassifier
import csv

def load_dataset():
    """Load the keypoint dataset"""
    dataset = 'model/keypoint_classifier/keypoint.csv'
    X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
    y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))
    return X_dataset, y_dataset

def load_labels():
    """Load gesture labels"""
    labels = []
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            labels.append(row[0])
    return labels

def create_confusion_matrix():
    # Load the model
    model = tf.keras.models.load_model('model/keypoint_classifier/keypoint_classifier.keras')
    
    # Load dataset and labels
    X_dataset, y_dataset = load_dataset()
    labels = load_labels()
    
    # Make predictions
    y_pred = model.predict(X_dataset)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Create confusion matrix
    cm = confusion_matrix(y_dataset, y_pred_classes)
    
    # Create a prettier visualization
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels)
    
    plt.title('Hand Gesture Recognition - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Save the plot
    plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_dataset, y_pred_classes, target_names=labels))
    
    return cm

def analyze_model_performance():
    cm = create_confusion_matrix()
    
    # Calculate accuracy per class
    accuracy_per_class = cm.diagonal() / cm.sum(axis=1)
    
    # Create a bar plot of accuracy per class
    plt.figure(figsize=(12, 6))
    labels = load_labels()
    plt.bar(labels, accuracy_per_class * 100)
    plt.title('Accuracy per Gesture Class')
    plt.xlabel('Gesture')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('accuracy_per_class.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == '__main__':
    analyze_model_performance()