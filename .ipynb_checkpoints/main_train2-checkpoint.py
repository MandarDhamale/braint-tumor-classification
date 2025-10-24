# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from itertools import cycle

# --- 1. SETUP DATA PATHS AND PARAMETERS ---
train_dir = 'data/train'
validation_dir = 'data/validation'
test_dir = 'data/test'

# Model parameters from your notebook
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
N_CLASSES = 4 # Number of classes

# --- 2. PREPROCESSING AND DATA AUGMENTATION ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    color_mode='rgb'
)

# --- 3. BUILD THE MODEL ARCHITECTURE ---
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(N_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# --- Print Model Summary for Report ---
print("\n--- Model Architecture Summary ---")
model.summary()
print("\n")


# --- 4. TWO-STAGE TRAINING AND FINE-TUNING ---
print("--- Starting Stage 1: Training the top layers ---")
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history_stage1 = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

print("\n--- Starting Stage 2: Fine-tuning the model ---")
for layer in base_model.layers[10:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history_stage2 = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# --- 5. EVALUATE THE FINAL MODEL ---
print("\n--- Evaluating the final model on the test set ---")
loss, accuracy = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)
print(f"Final Test Accuracy: {accuracy*100:.2f}%")
print(f"Final Test Loss: {loss:.4f}")

# Generate predictions for the test set
test_steps = int(np.ceil(test_generator.samples / BATCH_SIZE))
Y_pred_probs = model.predict(test_generator, steps=test_steps)
y_pred_labels = np.argmax(Y_pred_probs, axis=1)
y_true_labels = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# One-hot encode true labels for ROC/PR curve functions
y_true_one_hot = tf.keras.utils.to_categorical(y_true_labels, num_classes=N_CLASSES)


# --- 6. DISPLAY PERFORMANCE METRICS (as required by rubric) ---

# Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_true_labels, y_pred_labels, target_names=class_labels))

# Confusion Matrix
print("\n--- Generating Confusion Matrix ---")
cm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.show()

# AUC-ROC Curve
print("\n--- Generating AUC-ROC Curve ---")
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(N_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], Y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
plt.figure(figsize=(8, 6))
for i, color in zip(range(N_CLASSES), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(class_labels[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('auc_roc_curve.png')
plt.show()


# Precision-Recall Curve
print("\n--- Generating Precision-Recall Curve ---")
precision = dict()
recall = dict()
pr_auc = dict()
for i in range(N_CLASSES):
    precision[i], recall[i], _ = precision_recall_curve(y_true_one_hot[:, i], Y_pred_probs[:, i])
    pr_auc[i] = auc(recall[i], precision[i])

plt.figure(figsize=(8, 6))
for i, color in zip(range(N_CLASSES), colors):
    plt.plot(recall[i], precision[i], color=color, lw=2,
             label='PR curve of class {0} (area = {1:0.2f})'
             ''.format(class_labels[i], pr_auc[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Multi-class Precision-Recall Curve')
plt.legend(loc="lower left")
plt.savefig('precision_recall_curve.png')
plt.show()