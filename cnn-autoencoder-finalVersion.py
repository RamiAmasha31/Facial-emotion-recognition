
"""


@author: RAMI AMASHA
"""


import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Flatten
from keras.models import Model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Load the FER2013 dataset
import pandas as pd
import os
from PIL import Image
filesTrain = os.listdir('aug_data_64_by_48')
filesTest = os.listdir('aug_test_data_64_by_48')
n_iter=400
#n_iter = int(sys.argv[1])
tag_list = ['AN', 'SA', 'SU', 'HA', 'DI', 'FE', 'NE']

def GetTestTargets(filename):
    targets = []
    with open("example.txt", "r") as file:
    # Read all the lines from the file and store them in a list
        lines = file.readlines()
    for f in lines:
        if tag_list[0] in f:
            targets.append(0)
        if tag_list[1] in f:
            targets.append(1)
        if tag_list[2] in f:
            targets.append(2)
        if tag_list[3] in f:
            targets.append(3)
        if tag_list[4] in f:
            targets.append(4)
        if tag_list[5] in f:
            targets.append(5)
        if tag_list[6] in f:
            targets.append(6)
    return np.array(targets)

def testOnFER2013(): 
    """
    this function runs the trained model on fer2013 dataset 
    prints the test loss and the accuracy

    Returns
    -------
    None.

    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv('fer2013.csv')
    # Open the file in write mode
    with open("example.txt", "w") as file:
        # Split the data into training and test sets
        x_train, y_train, x_test, y_test = [], [], [], []
        for index, row in df.iterrows():
            emotion, pixels, usage = row['emotion'], row['pixels'], row['Usage']
            pixels = np.array(pixels.split(), dtype=np.int)
            pixels = pixels.reshape(48, 48, 1)
            if usage == 'Training':
                x_train.append(pixels)
                y_train.append(emotion)
            elif usage == 'PrivateTest':
                x_test.append(pixels)
                y_test.append(emotion)
                # Write the contents to the file
                file.write(emotion+'\n')
    y_test=GetTestTargets('rr')
    # Convert the lists to numpy arrays
   
    x_test = np.array(x_test)
    
    # Normalize the pixel values
 
    x_test = x_test.astype('float32') / 255.
    
    # Reshape the data to (num_samples, 48, 48, 1)
   
    x_test = np.reshape(x_test, (x_test.shape[0], 48, 48, 1))
    
    # One-hot encode the labels

    y_test = to_categorical(y_test, num_classes=7)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    print("Test Loss on fer2013 dataset:", test_loss)
    print("test accuracy on fer2013 dataset:",test_acc)

    
    
def targets(filename):
    targets = []
    for f in filename:
        if tag_list[0] in f:
            targets.append(0)
        if tag_list[1] in f:
            targets.append(1)
        if tag_list[2] in f:
            targets.append(2)
        if tag_list[3] in f:
            targets.append(3)
        if tag_list[4] in f:
            targets.append(4)
        if tag_list[5] in f:
            targets.append(5)
        if tag_list[6] in f:
            targets.append(6)
    return np.array(targets)


def dataTrain(filename):
    train_images = []
    for f in filename:
        current = f
        train_images.append(np.array(Image.open('aug_data_64_by_48/'+current).getdata()))    
    return np.array(train_images)

def dataTest(filename):
    train_images = []
    for f in filename:
        current = f
        train_images.append(np.array(Image.open('aug_test_data_64_by_48/'+current).getdata()))    
    return np.array(train_images)

y_train = targets(filesTrain)
print ("Fetching Data. Please wait......")
x_train = dataTrain(filesTrain)
print ("Fetching Complete.")
x_test=dataTest(filesTest)
y_test=targets(filesTest)

# Normalize the pixel values
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Reshape the data to (num_samples, 48, 48, 1)
x_train = np.reshape(x_train, (x_train.shape[0], 48, 48, 1))
x_test = np.reshape(x_test, (x_test.shape[0], 48, 48, 1))

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

# Create the input layer for the autoencoder
input_img = Input(shape=(48, 48, 1))

# Create the encoder layers
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

# Create the bottleneck layer
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Create the decoder layers
x = Conv2D(10, (5, 5), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(10, (5, 5), activation='relu', padding='same')(x)
x = UpSampling2D((5, 5))(x)
x = Conv2D(10, (5, 5), activation='relu')(x)
x = UpSampling2D((2, 2))(x)

# Create the output layer
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Create the autoencoder model
autoencoder = Model(input_img, decoded)

# Compile the autoencoder model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


# Create the encoder model for later use
encoder = Model(input_img, encoded)

# Add classification layers on top of the encoder
x = Flatten()(encoded)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(265, activation='relu')(x)
x = BatchNormalization()(x)
output = Dense(7, activation='softmax')(x)
weights=autoencoder.get_weights()
np.savetxt("autoencoderWeights.txt", [str(i) for i in weights],fmt='%s')

# Create the full model
model = Model(input_img, output)
plot_model(autoencoder, to_file='autoencoder_plot.png', show_shapes=True, show_layer_names=True)

# Train only the top layers (which were randomly initialized)
for layer in model.layers[:-8]:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='MAE', metrics=['accuracy'])

# # Augmenting Data
# datagen = ImageDataGenerator(
#     rotation_range=30,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest')

# training the model 
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
history=model.fit(x_train, y_train, batch_size=32,
                    epochs=50,
                    validation_data=(x_test, y_test),
                    steps_per_epoch=len(x_train) / 32)
					# Plot the training and validation loss
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['accuracy'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation','acc'], loc='upper right')
plt.show()
test_loss, test_acc = model.evaluate(x_test, y_test,batch_size=1990656, verbose=0)

print("Test Loss on jaffe dataset:", test_loss)

print("Test Accuracy jaffe dataset:", test_acc*110)
testOnFER2013()

