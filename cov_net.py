from sleep_helper_functions import normalize_across_rows_in_place
from sleep_helper_functions import k_fold_patient_partition
from sleep_helper_functions import down_sample
from sleep_helper_functions import visualize_data
from sleep_helper_functions import serialize_data_to_file
from sleep_helper_functions import getValidationData

import pandas as pd
import numpy as np
import os
import time

from sklearn.metrics import confusion_matrix
from sklearn import datasets

from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, Dropout, Flatten, Concatenate, Maximum, merge, Input
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras import metrics 
from keras import initializers
K.set_image_dim_ordering('th')

# settings
down_sample_training_data = True
load_weights_from_file = True
save_weights_to_file = True
save_results_to_file = True

# load pickle into memory
un_pickle = pd.read_pickle('/lustre/medusa/dsawant/sleep-raw-fpz-cz-plus.pickle')
data_set = un_pickle
#['dataset'], un_pickle['summary']
#del un_pickle  # remove a gigabyte from ram

# columns that we are looking at plus constants from the model
#cols = ['t' + str(i) for i in range(1, 6001)]
cols = data_set.columns.values[5:]
number_of_sleep_stages = len(set(data_set.stage))
patient_nums = set(data_set.patID)


def data_partition(data_set, patient_num):
    train_data_frame, test_data_frame = k_fold_patient_partition(data_set, patient_num=patient_num)

    if down_sample_training_data:
        train_data_frame = down_sample(train_data_frame)

    train_data_frame, valid_data_frame = getValidationData(train_data_frame)

    train_data, valid_data, test_data = train_data_frame[cols].as_matrix(), valid_data_frame[cols].as_matrix(), test_data_frame[cols].as_matrix()
    train_data_ints, valid_data_ints, test_data_ints = train_data_frame.stage.as_matrix(), valid_data_frame.stage.as_matrix(), test_data_frame.stage.as_matrix()
    train_labels, valid_labels, test_labels = to_categorical(train_data_ints, number_of_sleep_stages), to_categorical(valid_data_ints, number_of_sleep_stages), to_categorical(test_data_ints, number_of_sleep_stages)

    # covnet specific reshaping <- this is because images works in channels,
    # because images are typically done in 3 channels
    # even though our data is one dimensional, it doesn't assume one channel, it wants us to specifically
    # put that it has one channel. This took me forever to figure out.
    train_data = train_data.reshape(train_data.shape[0], len(train_data[0]), 1)
    valid_data = valid_data.reshape(valid_data.shape[0], len(valid_data[0]), 1)
    test_data = test_data.reshape(test_data.shape[0], len(test_data[0]), 1)

    return train_data, train_labels, test_data, test_labels, valid_data, valid_labels


# build the model

def make_model(input_shape):
    input = Input(shape = (train_data.shape[0], len(train_data[0]), 1))
    model1 = Sequential()
    model1.add(Conv1D(128, kernel_size=12, strides=4, padding='valid', activation='relu', input_shape=input_shape))
    model1.add(MaxPooling1D(pool_size=6))	
    model1.add(Conv1D(256, kernel_size=12, strides=4, padding='valid', activation='relu'))
    model1.add(MaxPooling1D(pool_size=6))
    model1.add(Dropout(.5))
    model1.add(Flatten())
    model1.add(Dense(256, activation='relu'))
    model1.add(Dropout(.5))
    model1.add(Dense(128, activation='relu'))
    
    model2 = Sequential()
    model2.add(Conv1D(128, kernel_size=20, strides=5, padding='valid', activation='relu', input_shape=input_shape))
    model2.add(MaxPooling1D(pool_size=5))
    model2.add(Conv1D(256, kernel_size=20, strides=5, padding='valid', activation='relu'))
    model2.add(MaxPooling1D(pool_size=5))
    model2.add(Dropout(.5))
    model2.add(Flatten())
    model2.add(Dense(256, activation='relu'))
    model2.add(Dropout(.5))
    model2.add(Dense(128, activation='relu'))
    
    model_concat =  Concatenate(axis=-1)([model1.output, model2.output])
    model_concat = Dropout(0.35)(model_concat)
    # output layer
    model_final = Dense(number_of_sleep_stages, activation='softmax')(model_concat)
    
    model = Model(inputs=[model1.input, model2.input], outputs=model_final)
    return model


# iterate through the patients

for patient_num in sorted(patient_nums):
    start_time = time.time()
    check_point_file_path = os.path.realpath("./patient{}.hdf5".format(int(patient_num)))

    # get partitioned data
    print('Partitioning Data for Patient {}'.format(patient_num))
    train_data, train_labels, test_data, test_labels, valid_data, valid_labels = data_partition(data_set, patient_num)

    # normalize the data
    print("Normalizing Data")
    normalize_across_rows_in_place(train_data)
    normalize_across_rows_in_place(test_data)

    # new model
    print("input shapes:")
    print(train_data[0].shape)
    print(train_labels[0].shape)
    print(valid_data[0].shape)
    print(valid_labels[0].shape)
    print(test_data[0].shape)
    print(test_labels[0].shape)

    model = make_model(input_shape=train_data[0].shape)

    # compile
    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[metrics.categorical_accuracy])

    # callbacks
    model_checkpoint = ModelCheckpoint(check_point_file_path, monitor='categorical_accuracy',
                                       verbose=1, save_best_only=True, mode='max')
    early_termination = EarlyStopping(monitor='val_loss', min_delta=.005,
                                      patience=10, verbose=1, mode='auto')

    callbacks = [model_checkpoint, early_termination] if save_weights_to_file else [early_termination]

    # load weights if flag is set
    if load_weights_from_file:
        if os.path.exists(check_point_file_path):
            try:
                model.load_weights(check_point_file_path)
                print("Weights for patient {} loaded successfully".format(patient_num))
            except Exception:
                print("Error Loading Weights for patient {}".format(patient_num))


    # fit/train the model
    print("Fitting Patient {}".format(patient_num))
   # model.fit(train_data, train_labels,
    #          batch_size=256, epochs=5, validation_data=(test_data, test_labels),
     #         verbose=1, callbacks=callbacks)  
    ## Few other things that changed
	model.fit([train_data, train_data], train_labels,
              batch_size=256, epochs=30, validation_data=([test_data, test_data], test_labels),
             verbose=1, callbacks=callbacks)
    # get results
    print("1st CM")
    predictions = np.argmax(model.predict([test_data, test_data], 1))
    # get results
    print("1st CM")
    predictions = np.argmax(model.predict(test_data, 1))
    real = np.argmax(test_labels, 1)
    confusion_data = confusion_matrix(predictions, real)

    # visualize data
    visualize_data(confusion_data)
    print("2nd CM")
    confusion_data = confusion_data.astype('float') / confusion_data.sum(axis=1)[:, np.newaxis]
    print(np.round(confusion_data,2))
    print("done showing results")
    # save the results to file
    if save_results_to_file:
        results_file_path = os.path.realpath('./patient{}.json'.format(int(patient_num)))
        results = {patient_num: confusion_data.tolist()}
        serialize_data_to_file(results, results_file_path)
        print("Writing Patient {} results to file".format(patient_num))
    print("Finished Processing Patient {} took {} seconds\n\n".format(patient_num, start_time - time.time()))