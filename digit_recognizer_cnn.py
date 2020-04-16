import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
np.random.seed(2)

sns.set(style='white', context='notebook', palette='deep')

# Load the data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Drop 'label' column for train data
train_x = train_data.drop("label", axis=1)
train_y = train_data["label"]


digit_counts = sns.countplot(train_y)

# Normalize the data
train_x = train_x / 255.0
test = test_data / 255.0

# Reshape data
train_x = train_x.values.reshape(-1, 28, 28, 1)
test = test_data.values.reshape(-1, 28, 28, 1)

train_y = to_categorical(train_y, num_classes=10)
# Set the random seed
random_seed = 2

# Split the train and the validation set for the fitting
train_x, x_valid, train_y, y_valid = train_test_split(train_x, train_y, test_size=0.1, random_state=random_seed)
# example digit
examples = plt.imshow(train_x[2][:, :, 0])

# Set the simple CNN model

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
epochs = 20
batch_size = 40


datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)
datagen.fit(train_x)

history = model.fit_generator(datagen.flow(train_x, train_y, batch_size=batch_size),
          epochs=epochs,
          verbose=2,
          validation_data=(x_valid, y_valid),
          steps_per_epoch=train_x.shape[0] // batch_size)
score = model.evaluate(x_valid, y_valid, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Plot the loss and accuracy curves for training and validation
fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results, axis=1)

results = pd.Series(results, name="Label")

submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)

submission.to_csv("data/submission.csv", index=False)
