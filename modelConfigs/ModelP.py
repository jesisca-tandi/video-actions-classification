from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv1D, Conv2D, MaxPooling2D, Embedding, LSTM, TimeDistributed, Masking, Lambda, GRU, Bidirectional, LeakyReLU
from keras.optimizers import Adam


def createModel(modelName='ModelP'):

    opt = Adam(lr=0.005)

    model = Sequential()
    model.add(Masking(mask_value=-1, input_shape=(None, 400)))
    model.add(Bidirectional(GRU(196, return_sequences = True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(GRU(128, return_sequences = True)))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(48, activation='softmax')))

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model, modelName