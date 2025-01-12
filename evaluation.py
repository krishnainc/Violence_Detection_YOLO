# Import required libraries
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Load the trained model
model_path = "violence_detection_final_model.keras"  # Update with your .h5 file path
model = tf.keras.models.load_model(model_path)

# Load the test dataset
test_dir = "Dataset/preprocessed_dataset/test"  # Update this path as needed
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
test_data = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False
)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_data, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Generate predictions
predictions = (model.predict(test_data) > 0.5).astype("int32").flatten()

# True labels
true_labels = test_data.classes

# Confusion Matrix
cm = confusion_matrix(true_labels, predictions)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
class_labels = list(test_data.class_indices.keys())
print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=class_labels))

# Plot Confusion Matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

plot_confusion_matrix(cm, class_labels)
plt.show()

# Optional: Save the confusion matrix plot
plt.savefig("confusion_matrix.png")
