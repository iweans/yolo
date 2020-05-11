import pathlib
# ----------------------------------------
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
# ----------------------------------------
from network import yolo_backbone
# --------------------------------------------------


DATASET_DIR = '/home/iweans/Datasets/cats_and_dogs_filtered'
BATCH_SIZE = 32
EPOCHS = 100
IMG_HEIGHT = 150
IMG_WIDTH = 150

dataset_dir = pathlib.Path(DATASET_DIR)
train_dir = dataset_dir / 'train'
validation_dir = dataset_dir / 'validation'

train_cats_dir = train_dir / 'cats'
train_dogs_dir = train_dir / 'dogs'
validation_cats_dir = validation_dir / 'cats'
validation_dogs_dir = validation_dir / 'dogs'

num_cats_train = len(list(train_cats_dir.glob('*')))
num_dogs_train = len(list(train_dogs_dir.glob('*')))
num_cats_val = len(list(validation_cats_dir.glob('*')))
num_dogs_val = len(list(validation_dogs_dir.glob('*')))

num_total_train = num_cats_train + num_dogs_train
num_total_val = num_cats_val + num_dogs_val

print('total training cat images:', num_cats_train)
print('total training dog images:', num_dogs_train)
print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("Total training images:", num_total_train)
print("Total validation images:", num_total_val)


print('-'*50)
# --------------------------------------------------
train_image_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                     horizontal_flip=True,
                                                                     rotation_range=45,
                                                                     zoom_range=0.5,
                                                                     width_shift_range=.15,
                                                                     height_shift_range=.15)
validation_image_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_data_gen = train_image_generator.flow_from_directory(str(train_dir),
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary',
                                                           batch_size=BATCH_SIZE,
                                                           shuffle=True)
val_data_gen = validation_image_generator.flow_from_directory(validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary',
                                                              batch_size=BATCH_SIZE)

# --------------------------------------------------
# sample_training_images, sample_training_label = next(train_data_gen)
# print(sample_training_images.shape)
# print(sample_training_label.shape)
import matplotlib.pyplot as plt
plt.close()
# def plotImages(images_arr):
#     fig, axes = plt.subplots(1, 5, figsize=(20,20))
#     axes = axes.flatten()
#     for img, ax in zip( images_arr, axes):
#         ax.imshow(img)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()
# plotImages(sample_training_images[:5])


# --------------------------------------------------
# model = yolo_backbone
# model.add(keras.layers.AveragePooling2D())
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(1024, activation='relu'))
# model.add(keras.layers.Dense(1))
# model.compile(optimizer='adam',
#               loss=keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# model.summary()
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam',
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=num_total_train // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=num_total_val // BATCH_SIZE
)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
