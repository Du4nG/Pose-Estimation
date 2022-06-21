from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, BatchNormalization, Flatten, LeakyReLU, MaxPool2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import pandas as pd

X_DATA_PATH = 'data.pickle'
Y_DATA_PATH = 'Pose Estimation/point.csv'

y_data = pd.read_csv(Y_DATA_PATH)
y_data.head(None)

x_data = pickle.load(open(X_DATA_PATH, 'rb'))
x_data = np.array(x_data, dtype='float32')
x_data /= 255
print('Shape of x data: ', x_data.shape)


input_shape = x_data.shape[1:4]
y_data = np.array(y_data, dtype='float')
num_class = y_data.shape[1]
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.15)
print('Input shape: ', input_shape)
print('Number of output: ', num_class)
print('x train shape: ', x_train.shape)
print('y train shape: ', y_train.shape)
print('x test shape: ', x_test.shape)
print('y test shape: ', y_test.shape)


n = x_data.shape[0]
plt.figure(figsize=(15, 15))
plt.subplots_adjust(hspace=.2)
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(cv2.cvtColor(x_data[i*50+1], cv2.COLOR_BGR2RGB))
plt.suptitle('Một vài tư thế trong tập dữ liệu', size=20)
plt.show()


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
          use_bias=True, input_shape=input_shape))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(num_class))
model.summary()


model.compile(
    optimizer='Adam',
    loss="mean_squared_error",
    metrics=['mae']
)

callback = EarlyStopping(monitor='val_mae', patience=25)
hist = model.fit(x_train, y_train, epochs=400,
                 batch_size=16, validation_split=0.15).history

final_loss, final_accuracy = model.evaluate(x_test, y_test)
print('Final loss: {:.2f}'.format(final_loss))
print('Final mae: {:.2f}'.format(final_accuracy))

model_history = pd.DataFrame(hist)
# add epoch column
model_history['epoch'] = np.arange(1, len(model_history) + 1)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
epochs = model_history.shape[0]
ax1.plot(np.arange(0, epochs), model_history['mae'], label='Training accuracy')
ax1.plot(np.arange(0, epochs),
         model_history['val_mae'], label='Validation accuracy')
ax1.legend(loc='lower right')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax2.plot(np.arange(0, epochs), model_history['loss'], label='Training loss')
ax2.plot(np.arange(0, epochs),
         model_history['val_loss'], label='Validation loss')
ax2.legend(loc='upper right')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
plt.tight_layout()
plt.show()


PATH = 'Pose Estimation'
MODEL_NAME = "Model_MAE" + str(round(final_accuracy)) + ".h5"
if round(final_loss) < 4:
    model.save(PATH + MODEL_NAME)
