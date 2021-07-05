from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv1D, Conv2D, MaxPooling2D, Embedding, LSTM, TimeDistributed, Masking, Lambda, GRU, Bidirectional, LeakyReLU
from keras.optimizers import Adam


def createModel(modelName='ModelL'):

    opt = Adam(lr=0.005)
    
    model = Sequential()
    model.add(Masking(mask_value=-1, input_shape=(None, 400)))
    model.add(Bidirectional(LSTM(128, input_shape= (None, 400), return_sequences = True)))
    model.add(Bidirectional(LSTM(128, return_sequences = True)))
    model.add(TimeDistributed(Dropout(0.05)))
    model.add(Bidirectional(LSTM(128, return_sequences = True)))
    model.add(TimeDistributed(Dense(48, activation='softmax')))

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model, modelName