import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

n = os.getcwd()
category_dir = os.path.join(n, 'covid19_dataset')
train = os.path.join(category_dir, 'train')
test = os.path.join(category_dir, 'test')

train_batch_size = 20
epochs = 10
IMG_HEIGHT = 150
IMG_WIDTH = 150

image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our training data

train_data_gen = image_generator.flow_from_directory(batch_size=train_batch_size,
                                                     directory=train,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='categorical')

model = Sequential([
    Conv2D(64,7, strides=(2, 2)),

    MaxPooling2D(strides=(2, 2)),
#Conv1
    Conv2D(64, 1, activation='relu'),
    Conv2D(64, 3, activation='relu'),
    Conv2D(256, 1, activation='relu'),

    Conv2D(64, 1, activation='relu'),
    Conv2D(64, 3, activation='relu'),
    Conv2D(256, 1, activation='relu'),

    Conv2D(64, 1, activation='relu'),
    Conv2D(64, 3, activation='relu'),
    Conv2D(256, 1, activation='relu'),
#Conv2
    Conv2D(128, 1, activation='relu'),
    Conv2D(128, 3, activation='relu'),
    Conv2D(512, 1, activation='relu'),

    Conv2D(128, 1, activation='relu'),
    Conv2D(128, 3, activation='relu'),
    Conv2D(512, 1, activation='relu'),

    Conv2D(128, 1, activation='relu'),
    Conv2D(128, 3, activation='relu'),
    Conv2D(512, 1, activation='relu'),

    Conv2D(128, 1, activation='relu'),
    Conv2D(128, 3, activation='relu'),
    Conv2D(512, 1, activation='relu'),
#Conv3
    Conv2D(256, 1, activation='relu'),
    Conv2D(256, 3, activation='relu'),
    Conv2D(1024, 1, activation='relu'),

    Conv2D(256, 1, activation='relu'),
    Conv2D(256, 3, activation='relu'),
    Conv2D(1024, 1, activation='relu'),

    Conv2D(256, 1, activation='relu'),
    Conv2D(256, 3, activation='relu'),
    Conv2D(1024, 1, activation='relu'),

    Conv2D(256, 1, activation='relu'),
    Conv2D(256, 3, activation='relu'),
    Conv2D(1024, 1, activation='relu'),

    Conv2D(256, 1, activation='relu'),
    Conv2D(256, 3, activation='relu'),
    Conv2D(1024, 1, activation='relu'),

    Conv2D(256, 1, activation='relu'),
    Conv2D(256, 3, activation='relu'),
    Conv2D(1024, 1, activation='relu'),
#Conv4
    Conv2D(512, 1, activation='relu'),
    Conv2D(512, 3, activation='relu'),
    Conv2D(2048, 1, activation='relu'),

    Conv2D(512, 1, activation='relu'),
    Conv2D(512, 3, activation='relu'),
    Conv2D(2048, 1, activation='relu'),

    Conv2D(512, 1, activation='relu'),
    Conv2D(512, 3, activation='relu'),
    Conv2D(2048, 1, activation='relu'),

    tf.keras.layers.AveragePooling2D(pool_size=(1, 1), strides=None, padding="valid", data_format=None),
    Flatten(),
    Dense(3, activation='softmax')
])

# model = Sequential([
#     Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),    #3차원 layer
#     MaxPooling2D(),
#     Conv2D(32, 3, padding='same', activation='relu'),
#     MaxPooling2D(),
#     Conv2D(64, 3, padding='same', activation='relu'),
#     MaxPooling2D(),
#     Flatten(),
#     Dense(512, activation='relu'),      # 1차원 layer
#     Dense(3, activation='softmax')
# ])

op = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD")

model.compile(optimizer=op,
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

result = model.fit(train_data_gen,
                   epochs=epochs,    # batch_size
                   steps_per_epoch=train_batch_size)
