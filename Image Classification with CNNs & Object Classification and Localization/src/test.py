import tensorflow as tf
from .dataset import create_dataset
from .models import create_custom_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def test(
        model,
        dataset,
        model_name,
        p,
        lr,
        batch_size,
        test_file):
    y_true_list = []
    y_pred_list = []

    for inputs, labels in dataset:
        predictions = model.predict(inputs)
        y_true_list.extend(np.argmax(labels.numpy(), axis=1))
        
        # Check if predictions is a TensorFlow tensor or a NumPy array
        y_pred_list.extend(np.argmax(predictions, axis=1))

    # Convert lists to numpy arrays
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    # Compute ROC-AUC
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f'src/cms/{model_name}_{p}_{lr}_{batch_size}.png')

    report = classification_report(y_true, y_pred)

    loss, categorical_accuracy, precision, recall = model.evaluate(dataset, verbose=1)

    with open(test_file, 'w') as file:
        file.write(f'Test Loss: {loss}\n')
        file.write(f'Test Categorical Accuracy: {categorical_accuracy}\n')
        file.write(f'Test Precision: {precision}\n')
        file.write(f'Test Recall: {recall}\n')
        file.write(f'Test Confusion Matrix:\n {str(cm)}' + '\n')
        file.write(f'Test Classification Report:\n {str(report)}' + '\n')

