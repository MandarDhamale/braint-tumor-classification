import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
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

# Model parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16 # Keeping batch size 16 as in your successful M2 run
N_CLASSES = 4
EPOCHS_STAGE_1 = 50 # Train the new head
EPOCHS_STAGE_2 = 50 # Fine-tune the base model (Total 100 epochs)

# --- 2. PREPROCESSING AND DATA AUGMENTATION ---
# (Same as Milestone 2)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20, # Added rotation for more robustness
    width_shift_range=0.2,
    height_shift_range=0.2
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

# --- 3. BUILD THE NOVEL MODEL (EfficientNetB0) ---
# Load the pretrained EfficientNetB0 model as the base
# We include_top=False to remove the original classifier
base_model = EfficientNetB0(
    weights='imagenet', 
    include_top=False, 
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

# --- This is our NOVEL CONTRIBUTION ---
# 1. We use GlobalAveragePooling2D instead of Flatten to reduce parameters
# 2. We add a Dropout layer to prevent overfitting
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x) # A dense layer to learn complex features
x = Dropout(0.5)(x) # Dropout for regularization
predictions = Dense(N_CLASSES, activation='softmax')(x) # Output layer for 4 classes
# ----------------------------------------

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# --- 4. TWO-STAGE TRAINING AND FINE-TUNING ---

# === STAGE 1: Train Only the Top Layers ===
print("--- Starting Stage 1: Training the new top layers ---")
# Freeze all layers of the base EfficientNet model
base_model.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary for Stage 1 (Head Training)
print("--- Model Summary (Stage 1: Head Frozen) ---")
model.summary()

# Train the model
history_stage1 = model.fit(
    train_generator,
    epochs=EPOCHS_STAGE_1,
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# === STAGE 2: Fine-Tuning ===
print("\n--- Starting Stage 2: Fine-tuning the base model ---")
# Unfreeze the base model to allow fine-tuning
base_model.trainable = True

# We can choose to unfreeze only the top blocks
# For EfficientNetB0, unfreezing from block 6 onwards is a good start
# Let's unfreeze the last 20 layers for this experiment
fine_tune_at_layer = len(base_model.layers) - 20 

for layer in base_model.layers[:fine_tune_at_layer]:
   layer.trainable = False

# Re-compile the model with a much lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary for Stage 2 (Fine-Tuning)
print("--- Model Summary (Stage 2: Fine-Tuning) ---")
model.summary()

# Continue training the model (fine-tuning)
history_stage2 = model.fit(
    train_generator,
    epochs=EPOCHS_STAGE_2,
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    initial_epoch=history_stage1.epoch[-1] # Continue epoch count
)

# --- 5. EVALUATE THE FINAL MODEL ---
print("\n--- Evaluating the final model on the test set ---")
loss, accuracy = model.evaluate(test_generator, steps=int(np.ceil(test_generator.samples / BATCH_SIZE)))
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
print("\n--- Classification Report (Novel Model) ---")
print(classification_report(y_true_labels, y_pred_labels, target_names=class_labels))

# Confusion Matrix
print("\n--- Generating Confusion Matrix (Novel Model) ---")
cm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix - EfficientNetB0 Model (Milestone 3)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('m3_confusion_matrix.png')
plt.show()

# AUC-ROC Curve
print("\n--- Generating AUC-ROC Curve (Novel Model) ---")
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
plt.title('Multi-class ROC Curve - EfficientNetB0 Model (Milestone 3)')
plt.legend(loc="lower right")
plt.savefig('m3_auc_roc_curve.png')
plt.show()

# Precision-Recall Curve
print("\n--- Generating Precision-Recall Curve (Novel Model) ---")
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
plt.title('Multi-class Precision-Recall Curve - EfficientNetB0 Model (Milestone 3)')
plt.legend(loc="lower left")
plt.savefig('m3_precision_recall_curve.png')
plt.show()

print("Milestone 3 script complete. Plots saved as m3_*.png")