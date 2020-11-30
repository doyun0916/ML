from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
from matplotlib import pyplot as plt

n = os.getcwd()
category_dir = os.path.join(n, 'covid19_dataset')
train = os.path.join(category_dir, 'train')

train_batch_size = 20
epochs = 10
IMG_HEIGHT = 150
IMG_WIDTH = 150

image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our training data

train_data_gen = image_generator.flow_from_directory(batch_size=train_batch_size,
                                                     directory=train,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='categorical'
                                                    )

model = tf.keras.applications.ResNet50(
    include_top=True,          # 마지막 단계에서 FC의 유무
    weights=None,
    pooling='avg',
    classes=3,
    classifier_activation="softmax",
)

model.compile(optimizer='SGD',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

result = model.fit(
    train_data_gen,
    epochs=epochs,    # batch_size
    steps_per_epoch=train_batch_size
)