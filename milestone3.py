import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, BatchNormalization, Activation, Add, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.utils import class_weight
from itertools import cycle
import os

# --- CONFIGURATION ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
N_CLASSES = 4

# Setup Paths
train_dir = 'data/train'
validation_dir = 'data/validation'
test_dir = 'data/test'

# --- DATA GENERATORS WITH VGG PREPROCESSING ---
from tensorflow.keras.applications.vgg16 import preprocess_input

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, 
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

#Re-load the iterators
print("Reloading Data with Correct Preprocessing...")
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

# --- CUSTOM RESIDUAL BLOCK (Algorithmic Change) ---
def residual_block(input_tensor, filters):
    """
    Creates a residual block with a skip connection.
    Input -> [Conv -> BN -> ReLU -> Conv -> BN] + Input -> ReLU
    """
    x = input_tensor
    
    # 1. Convolutional Path (F(x))
    fx = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    fx = BatchNormalization()(fx)
    
    fx = Conv2D(filters, (3, 3), padding='same')(fx)
    fx = BatchNormalization()(fx)
    
    # 2. Skip Connection (x)
    # Adding the input directly to the output allows gradients to flow 
    # more easily, helping the model learn complex features for 'glioma'.
    x = Add()([x, fx])
    
    # 3. Final Activation
    x = Activation('relu')(x)
    return x

# --- CALCULATE CLASS WEIGHTS ---
# Get all labels from the training generator
y_train_labels = train_generator.classes

# Compute weights: higher weight for under-represented classes
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
# Convert to dictionary format required by Keras
class_weights_dict = dict(enumerate(class_weights))

print(f"\nComputed Class Weights: {class_weights_dict}")

# --- BUILD NOVEL HYBRID ARCHITECTURE ---

# 1. Load VGG16 Base
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# 2. Truncate VGG16
vgg_output = base_model.get_layer('block4_pool').output

# 3. Add Custom Layers
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(vgg_output)

# --- Inject Residual Blocks ---
x = residual_block(x, filters=512)
x = residual_block(x, filters=512)

# --- NORMALIZE BEFORE FLATTENING ---
# This scales the high-value VGG features down so the Dense layers can learn
x = BatchNormalization(name='adapter_bn')(x) 

# 4. Add Custom Classification Head
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)

predictions = Dense(N_CLASSES, activation='softmax')(x)

# 5. Final Model Assembly
model = Model(inputs=base_model.input, outputs=predictions)

print("\n--- Novel VGG16-ResNet Hybrid Model Summary (Fixed) ---")
# model.summary()

# --- STAGE 1: TRAIN TOP LAYERS ---
print("\n--- Starting Stage 1: Training Custom Layers ---")
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history_stage1 = model.fit(
    train_generator,
    epochs=50, 
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    class_weight=class_weights_dict
)

# --- STAGE 2: FINE-TUNING ---
print("\n--- Starting Stage 2: Fine-Tuning Hybrid Architecture ---")

# Unfreeze block4 (Index 11)
for layer in base_model.layers[11:]:
    layer.trainable = True

# Very low learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history_stage2 = model.fit(
    train_generator,
    epochs=50, 
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    class_weight=class_weights_dict
)

# --- 6. EVALUATE THE NOVEL MODEL ---

print("\n--- Evaluating the Novel Hybrid Model on the Test Set ---")
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)
print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Final Test Loss: {test_loss:.4f}")

# --- GENERATE PREDICTIONS ---
# Reset generator to ensure order matches
test_generator.reset() 
test_steps = int(np.ceil(test_generator.samples / BATCH_SIZE))

# Get probabilities for all classes
Y_pred_probs = model.predict(test_generator, steps=test_steps)

# Convert probabilities to class labels (0, 1, 2, 3)
y_pred_labels = np.argmax(Y_pred_probs, axis=1)

# Get true labels
y_true_labels = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# --- CLASSIFICATION REPORT ---
print("\n--- Novel Model Classification Report ---")
print(classification_report(y_true_labels, y_pred_labels, target_names=class_labels))

print("Generating predictions...")
# Reset the test generator to ensure the order of images matches the order of predictions
test_generator.reset()

y_true_labels = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

steps = int(np.ceil(test_generator.samples / test_generator.batch_size))
Y_pred_probs = model.predict(test_generator, steps=steps)

# Convert probabilities to class indices (0, 1, 2, 3)
y_pred_labels = np.argmax(Y_pred_probs, axis=1)

# --- PRINT PERFORMANCE METRICS ---
print("\n--- Final Test Performance Metrics ---")
# Calculate Loss and Accuracy
loss, accuracy = model.evaluate(test_generator, steps=steps, verbose=0)
print(f"Final Test Accuracy: {accuracy*100:.2f}%")
print(f"Final Test Loss: {loss:.4f}")

print("\n--- Detailed Classification Report ---")
print(classification_report(y_true_labels, y_pred_labels, target_names=class_labels))

# --- PLOT CONFUSION MATRIX ---
print("\n--- Generating Confusion Matrix ---")
cm = confusion_matrix(y_true_labels, y_pred_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Novel Hybrid Model Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# --- PLOT ROC CURVES ---
print("\n--- Generating AUC-ROC Curves ---")
# One-hot encode true labels for ROC calculation
y_true_one_hot = tf.keras.utils.to_categorical(y_true_labels, num_classes=N_CLASSES)

fpr = dict()
tpr = dict()
roc_auc = dict()

plt.figure(figsize=(10, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])

for i, color in zip(range(N_CLASSES), colors):
    # Calculate ROC for each class
    fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], Y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of {class_labels[i]} (area = {roc_auc[i]:0.2f})')

# Plot the "Random Guess" line
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class ROC Curve (Novel Model)')
plt.legend(loc="lower right")
plt.show()

# --- PLOT PRECISION-RECALL CURVES ---
print("\n--- Generating Precision-Recall Curves ---")
precision = dict()
recall = dict()
pr_auc = dict()

plt.figure(figsize=(10, 8))

for i, color in zip(range(N_CLASSES), colors):
    # Calculate Precision-Recall for each class
    precision[i], recall[i], _ = precision_recall_curve(y_true_one_hot[:, i], Y_pred_probs[:, i])
    pr_auc[i] = auc(recall[i], precision[i])
    
    # Plot
    plt.plot(recall[i], precision[i], color=color, lw=2,
             label=f'PR curve of {class_labels[i]} (area = {pr_auc[i]:0.2f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Multi-Class Precision-Recall Curve (Novel Model)')
plt.legend(loc="lower left")
plt.show()
# --- END OF FILE ---
