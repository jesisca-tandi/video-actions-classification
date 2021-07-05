from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv1D, Conv2D, MaxPooling2D, Embedding, LSTM, TimeDistributed, Masking, Lambda, GRU, Bidirectional, LeakyReLU
from keras.optimizers import Adam


def createModel(modelName='ModelC'):

    opt = Adam(lr=0.01)

    model = Sequential()
    model.add(Masking(mask_value=-1, input_shape=(None, 400)))
    model.add(LSTM(128, return_sequences = True))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Dense(48, activation='softmax')))

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model, modelName