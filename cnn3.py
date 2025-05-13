import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tf.keras import layers, models, optimizers, callbacks, applications, utils


# Configuration
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
NUM_CLASSES = 3
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
OUTPUT_DIR = 'trained_model'

# Prepare output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) Load datasets with image_dataset_from_directory
train_ds = tf.keras.utils.image_dataset_from_directory(
    'generated_images/train',
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='training'
)  # :contentReference[oaicite:4]{index=4}

val_ds = tf.keras.utils.image_dataset_from_directory(
    'generated_images/train',
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='validation'
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    'generated_images/test',
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    shuffle=False
)

# 2) Build a data‐augmentation pipeline using Keras preprocessing layers
data_augmentation = tf.keras.Sequential([
    layers.Rescaling(1./255),                 # scale pixels to [0,1]
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])  # :contentReference[oaicite:5]{index=5}

# 3) Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y)).prefetch(AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (layers.Rescaling(1./255)(x), y)).prefetch(AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (layers.Rescaling(1./255)(x), y)).prefetch(AUTOTUNE)

# 4) Build the model with transfer learning (VGG16)
base_model = applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMAGE_SIZE, 3)
)

# Freeze base
base_model.trainable = False

# Add classification head
inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
x = data_augmentation(inputs)         # include augmentation in model
x = base_model(x, training=False)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = models.Model(inputs, outputs)

# 5) Compile with proper optimizer argument
model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),  # use learning_rate :contentReference[oaicite:6]{index=6}
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 6) Callbacks
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-4
)
checkpoint = callbacks.ModelCheckpoint(
    os.path.join(OUTPUT_DIR, 'best_model.keras'),
    monitor='val_loss',
    save_best_only=True
)

# 7) Initial training
history = model.fit(
    train_ds,
    epochs=NUM_EPOCHS,
    validation_data=val_ds,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# 8) Evaluate on test set
loss, accuracy = model.evaluate(test_ds, verbose=1)
print(f'Test accuracy: {accuracy:.2f}')

# 9) Metrics with scikit-learn
y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
y_pred = np.argmax(model.predict(test_ds), axis=1)
y_true_labels = np.argmax(y_true, axis=1)

print(classification_report(y_true_labels, y_pred))
print(confusion_matrix(y_true_labels, y_pred))

# 10) Fine-tuning: unfreeze last layers of the base model
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_ds,
    epochs=5,
    validation_data=val_ds,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# 11) Save final model
model.save(os.path.join(OUTPUT_DIR, 'final_model.keras'))
