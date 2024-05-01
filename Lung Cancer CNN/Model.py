import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Data Processing

Training_path = "C:\\Users\\Ahmed\\Desktop\\Dataset2\\LungCancerDetectionUsingCNN-main\\dataset\\training_set"
Testing_Path = "C:\\Users\\Ahmed\\Desktop\\Dataset2\\LungCancerDetectionUsingCNN-main\\dataset\\test_set"

#Train Data
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory(Training_path,
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


#test Data
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(Testing_Path,
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


#Building The Model
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64,3]))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(x = training_set, validation_data = test_set, epochs = 30)

model.save("LCDCNN.h5")