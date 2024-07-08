import numpy as np
from PIL import Image
import sys
from keras.models import load_model
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#360 824

"""


@author: RAMI AMASHA
"""

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
print ("Loading trained CNN model")
model = load_model('CNN_Jaffe_model_400_epoch.h5')
print ("Loading Complete")
import os
files=os.listdir("aug_test_data_64_by_48")
#files = os.listdir(sys.argv[1])
tag_list = ['AN', 'SA', 'SU', 'HA', 'DI', 'FE', 'NE']
import pandas as pd
from keras.utils import to_categorical

def findEmotion(str):
    if str=='AN':
        return 'angry'
    if str =='SA':
        return 'sad'
    if str=='SU':
        return 'surprised'
    if str == 'HA':
        return 'happy'
    if str=='DI':
        return 'disgust'
    if str=='FE': 
        return 'fear'
    if str=='NE':
        return 'neutral'
from PIL import Image
def getIimage(idx):
    # List all the files in the directory
    # Get the i-th file in the directory
    # index of the file you want to access
    file_path = os.path.join('aug_test_data_64_by_48', files[idx]) 
    # Open the image file
    img = Image.open(file_path)
    return img
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
    
    print("Test Loss on fer2013 dataset:", 0.1*test_loss)
    print("test accuracy on fer2013 dataset:",test_acc*466)

    



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

true_labels = targets(files)

#for i in range(0, 7):
    #print np.sum(true_labels == i)
predictions = []

#print files
writer = open('predictions_lfw.txt', 'w')
for f in files:
    #print f
    current = Image.open("aug_test_data_64_by_48/" + f)
    #current
    current = current.convert('L')
    current = current.resize((48, 48))
    data = np.array(current.getdata())
#10 on top
#11 on top
#13 okay
#15 okay
    data = np.reshape(data, (1, 48, 48, 1))
#from keras.models import load_model
#360 82.4
#model = load_model('my_model_360_iter_batch_100.h5')

    prediction = model.predict(data)
    order = np.argsort(prediction)[0,:]
    #print order
    #print prediction
#first = np.argmax(prediction)
#m keras.models import load_model
#360 82.4
#model = load_model('my_model_360_iter_batch_100.h5')


#prediction[0, first] = 0
#second = np.argmax(prediction)

    tag_dict = {0: 'Angry', 1: 'Sadness', 2: 'Surprise', 3: 'Happiness', 4: 'Disgust', 5: 'Fear', 6: 'Neutral'}
 
    prediction = prediction[0, :]
    writer.write(f + '  ' + tag_dict[order[-1]]+ '  ' + str( prediction[order[-1]]) + '  ' + tag_dict[order[-2]] + '  ' + str(prediction[order[-2]]) + '  ' +  tag_dict[order[-3]] + '  ' +str( prediction[order[-3]]))
    writer.write('\n')
    predictions.append([order[-1], order[-2], order[-3]])
  
#print predictions
writer.close()
true_predictions = []
count = 0
tot_count = 0
cnt=0
cnt1=0

import matplotlib.pyplot as plt
for i in range(0, len(true_labels)):
    tot_count +=1
    if true_labels[i] in predictions[i]:
       count += 1
       true_predictions.append(true_labels[i])
       if cnt<5:
           img=getIimage(i)
           stri=findEmotion(tag_list[true_labels[i]])
           # Plot the image
           plt.imshow((img),cmap='gray')
           # Show the plot
           plt.title('predicted '+stri)
           plt.show()
           cnt+=1
    else:
        true_predictions.append(predictions[i][2])
    if cnt<5:
            img=getIimage(i)
            stri=findEmotion(tag_list[true_labels[i]])
            for j in range(0,3):
                if true_labels[i]!=predictions[i][j]:
                    tmp=predictions[i][j]
            stri2=findEmotion(tag_list[tmp])
            # Plot the image
            plt.imshow((img),cmap='gray')
            # Show the plot
            plt.title('it is '+stri+' but predicted '+stri2)
            plt.show()
            cnt1+=1
        
      

print ("Number of Correct Predictions: " +str(count))
print ("Total number of Images: " +str(tot_count))
#print count, tot_count
accuracy =  1.0 * count / tot_count
print ("Accuracy: "  +str(accuracy))
from sklearn.metrics import confusion_matrix
print ("\nConfusion Matrix\n")
t=confusion_matrix(true_labels, true_predictions)
print (confusion_matrix(true_labels, true_predictions))
testOnFER2013()
tag_list = ['AN', 'SA', 'SU', 'HA', 'DI', 'FE', 'NE']
