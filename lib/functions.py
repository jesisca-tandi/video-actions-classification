'''
CS5242 Project - Classification of videos actions using breakfast action datasets

-----------------
Group Members: 
Jesisca Tandi (A0185994E)
Tan Hong Xiu (A0186008E)
Hwee Yew Rong Kelvin (A0186097N) 
George Loo Zheng Xian (A0186064B) 

-----------------
List of packages (Python 3.5):
Keras                2.3.1
tensorflow           2.1.0
numpy                1.18.2
pandas               0.24.2
sklearn              0.0
scipy                1.4.1
torch                1.4.0
'''

import keras
from keras import regularizers
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv1D, Conv2D, MaxPooling2D, Embedding, LSTM, TimeDistributed, Masking, Lambda, GRU, Bidirectional
from keras.preprocessing import image, sequence
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
from keras.utils import to_categorical, np_utils, Sequence
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import numpy as np
import pandas as pd
import os, torch, pickle
import scipy.stats as stats
from .read_datasetBreakfast import load_data, read_mapping_dict


def getData(split, COMP_PATH=''):
    '''Load train / test data. Input: (str) 'test', 'training' '''

    train_split =  os.path.join(COMP_PATH, 'splits/train.split1.bundle') #Train Split
    test_split  =  os.path.join(COMP_PATH, 'splits/test.split1.bundle') #Test Split
    GT_folder   =  os.path.join(COMP_PATH, 'groundTruth/') #Ground Truth Labels for each training video 
    DATA_folder =  os.path.join(COMP_PATH, 'data/') #Frame I3D features for all videos
    mapping_loc =  os.path.join(COMP_PATH, 'splits/mapping_bf.txt') 

    actions_dict = read_mapping_dict(mapping_loc)

    if  split == 'training':
        data_feat, data_labels = load_data(train_split, actions_dict, GT_folder, DATA_folder, datatype = split) #Get features and labels
        return data_feat, data_labels

    if  split == 'test':
        data_feat = load_data(test_split, actions_dict, GT_folder, DATA_folder, datatype = split) #Get features only
        return data_feat



def processTrainData(**kwargs):
    '''Load and then process training data 
    Input: (str) 'test' or 'training'

    Process:
    1. one-hot encoding of labels
    2. sequence padding of features and labels
    3. train-validation split (80:20)

    '''

    X, y = getData('training', **kwargs)

    # Check the maximum no of frames in the train dataset
    maxFrames = max([i.shape[0] for i in X])

    # Transform labels into categorical labels (one-hot encoding)
    y_cat = [to_categorical(i, 48) for i in y]

    # Padding of different sequence length
    # As the dataset is of different number of frames for each of the videos, 
    # we are doing a post-padding with -1 to make sure all the videos are of equal number of frames (pad to the max data length)
    # (i.e. padded at the end of the videos)
    y_padded = sequence.pad_sequences(y_cat, maxlen=maxFrames,padding='post', truncating='post', value=-1, dtype='int')
    X_padded = sequence.pad_sequences(X, maxlen=maxFrames,padding='post', truncating='post', value=-1, dtype='float16')

    # To facilitate the comparison of the validation data using the training_segment.txt file provided, 
    # We split the training segment together with the train-validation split.
    training_segment_rev = processSegment('training_segment.txt')

    # Train-validation split (80:20)
    # As we do not have ground truth for test data, we are further splitting the train dataset into train and validation set.
    X_train, X_val, y_train, y_val, segment_train, segment_val = train_test_split(X_padded, y_padded, training_segment_rev, random_state=1, test_size=0.2)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    print('Train set: X shape {}, y shape {}'.format(X_train.shape, y_train.shape))
    print('Validation set: X shape {}, y shape {}'.format(X_val.shape, y_val.shape))

    return X_train, X_val, y_train, y_val, segment_train, segment_val


def processTestData(**kwargs):
    '''Load and process test data (with padding of sequences)'''
    
    # Load data
    X = getData('test', **kwargs)

    # Check the maximum no of frames in the train dataset
    maxFrames = max([i.shape[0] for i in X])

    # Post-padding of sequences of unequal length by values of -1
    X_padded = sequence.pad_sequences(X, maxlen=maxFrames,padding='post', truncating='post', value=-1, dtype='float16')

    return X_padded


def processSegment(file):
    '''Get segments of videos for the purpose of validation'''

    file_open = open(file, 'r')
    segment = file_open.read().split('\n')[:-1]
    segment_rev = []
    for i in range(len(segment)):
        segment_rev.append([int(j) for j in segment[i].split()])

    return segment_rev


def getMajorityVotes(y, segments):
    '''Function to get the majority vote of labels within each segment'''

    votes = []
    for i,j in zip(y, segments):
        for m,n in zip(j[0:-1], j[1:]):
            votes.append(stats.mode(i[m:n])[0][0])
    return votes


def createDir(wdir):
    '''Function to create directory'''
    if not os.path.exists(wdir):
        os.makedirs(wdir)


def train(model, modelName, savePath, batchSize=50, epochs=50, COMP_PATH=''):
    '''
    Run training
    1. Load train-validation data
    2. Train model (and save models after each epochs)
    3. Evaluate on validation set
    '''

    createDir(savePath)
    createDir(os.path.join(savePath, 'checkPoints'))

    # Load and process data
    X_train, X_val, y_train, y_val, segment_train, segment_val = processTrainData(COMP_PATH=COMP_PATH)

    # Show model summary 
    model.summary()

    # Start training
    checkPointsModel = os.path.join(savePath, 'checkPoints', 'saved-model-{epoch:02d}.h5')
    checkpoint = ModelCheckpoint(checkPointsModel, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto')
    model.fit(X_train, y_train, validation_data = (X_val, y_val), batch_size=batchSize, epochs=epochs, callbacks=[checkpoint])

    # Save model
    model.save(os.path.join(savePath, 'model_{}_{}.h5'.format(modelName, str(datetime.now()).replace(' ', '_').replace(':', '')[:17])))

    # Check validation scores
    validate(model, X_val, y_val, segment_val)

    return model


def validate(trainedModel, X_val, y_val, segment_val):
    '''
    Evaluate validation
    1. Loss function of the model
    2. Calculate accuracy of video segment classification using majority vote
    '''

    # Get validation performance
    val_loss, val_acc = trainedModel.evaluate(X_val, y_val)
    print('Test Loss: {}, Accuracy: {}'.format(val_loss, val_acc))

    # Get classification accuracy of classification of each video segment (majority voting) to simulate final testing 
    yhat_val = trainedModel.predict_classes(X_val)
    sliced_y_val    = getMajorityVotes(np.argmax(y_val, axis=-1), segment_val)
    sliced_yhat_val = getMajorityVotes(yhat_val, segment_val)
    acc = accuracy_score(sliced_y_val, sliced_yhat_val)
    print("Accuracy based on sliced data: " + str(acc))


def test(trainedModel, savePath, COMP_PATH=''):
    '''Evaluate test data predictions'''

    createDir(savePath)

    X = processTestData(COMP_PATH=COMP_PATH)

    test_segment_rev = processSegment('test_segment.txt')
    
    # Predict
    yhat = trainedModel.predict_classes(X)

    # Get majority votes
    yhat_maj = getMajorityVotes(yhat, test_segment_rev)

    # Save out predictions
    new_test = pd.DataFrame()
    new_test['Id']          = list(range(len(yhat_maj)))
    new_test['Category']    = yhat_maj
    new_test.to_csv(os.path.join(savePath, 'Predicted_Category_{}.csv'.format(modelName)))
