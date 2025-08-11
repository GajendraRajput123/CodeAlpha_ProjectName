# Step 1: Install dependencies if needed
# pip install tensorflow tensorflow-datasets matplotlib

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Step 2: Load EMNIST dataset (ByClass contains digits + uppercase + lowercase)
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/byclass',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# Step 3: Preprocess images
def normalize_img(image, label):
    # Convert from uint8 to float32 and normalize to [0,1]
    return tf.cast(image, tf.float32) / 255.0, label

ds_train = ds_train.map(normalize_img).cache().shuffle(1000).batch(128).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(normalize_img).batch(128).cache().prefetch(tf.data.AUTOTUNE)

# Step 4: Build CNN model
num_classes = ds_info.features['label'].num_classes

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Step 5: Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train model
model.fit(ds_train, epochs=5, validation_data=ds_test)

# Step 7: Evaluate
test_loss, test_acc = model.evaluate(ds_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Step 8: Visualize predictions
for image, label in ds_test.take(1):
    predictions = model.predict(image)
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(image[i].numpy().squeeze(), cmap='gray')
        pred_label = tf.argmax(predictions[i]).numpy()
        plt.title(f"Pred: {pred_label}")
        plt.axis('off')
    plt.show()
