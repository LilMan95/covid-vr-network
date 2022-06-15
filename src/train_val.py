import glob
import os
import random
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

from datetime import datetime

import tensorflow as tf
from tensorflow.keras.metrics import binary_accuracy


# ---------------------------------
# READING VARIABLES
# ---------------------------------

CFG_FILE="../config/train_val.yaml" 

config = {}
with open(CFG_FILE, "r") as f: 
    config = yaml.safe_load(f)

# Reading variables from config files
AXIS_LIST = ["axis{}".format(i) for i in config["DATASET"]["AXIS"]]
DATASET_NAME = config["DATASET"]["DATASET_NAME"]
DATA_ROOT_INPUT = config["DATASET"]["DATA_ROOT_INPUT"]
DATA_ROOT_OUTPUT = config["DATASET"]["DATA_ROOT_OUTPUT"]
IMAGES_PATH = config["DATASET"]["IMAGES_PATH"]

EPOCHS = config["MODEL"]["TRAIN_EPOCHS"]
IMG_WIDTH, IMG_HEIGHT = config["MODEL"]["INPUT_SIZE"] 
NUM_CLASSES = config["MODEL"]["NUM_CLASSES"]
LEARNING_RATE = config["MODEL"]["LEARNING_RATE"]
NUM_FOLDS = config["MODEL"]["NUM_FOLDS"]

# default variables
IMG_CHANNELS = 3
ACCURACY = 'accuracy' if NUM_CLASSES > 2 else 'binary_accuracy'

# paths
SRC_PATH = os.path.abspath(".")
LOG_PATH = os.path.join(SRC_PATH, "..", "logs", DATASET_NAME)
EXPERIMENTS_PATH = os.path.join(SRC_PATH, "..", "experiments", DATASET_NAME)

INPUT_PATH = os.path.join(SRC_PATH, "..", DATA_ROOT_INPUT, DATASET_NAME)
OUTPUT_PATH = os.path.join(SRC_PATH, "..", DATA_ROOT_OUTPUT, DATASET_NAME)



def get_files(path):

    if os.path.exists(path):
        return [os.path.join(path, f) for f in os.listdir(path)]
    return []


def fill_image_dict(patient_label, axis):
    
    dataset_dict = dict(id=[], label=[])
    for patient, label in patient_label.items():
        path = os.path.join(IMAGES_PATH, patient, axis)        
        files = get_files(path)
        dataset_dict["id"].extend(files)
        dataset_dict["label"].extend([label] * len(files))
    return pd.DataFrame(data=dataset_dict)


def get_data_set(axis, train_set, validation_set):
    
    patient_train_label = dict([ (r["patient"], r["label"]) for _, r in train_set.iterrows() ])
    patient_validation_label = dict([ (r["patient"], r["label"]) for _, r in validation_set.iterrows() ])

    df_train = fill_image_dict(patient_train_label, axis)
    df_val = fill_image_dict(patient_validation_label, axis)
    
    print(df_train.head())
    
    print("Train fold with {} images".format(len(df_train)))
    print(df_train.groupby("label").label.count())
    print()
    print("Validation fold with {} images".format(len(df_val)))
    print(df_val.groupby("label").label.count())
    print("-" * 30)

    return df_train, df_val


    
def get_data_generator(dataframe, x_col, y_col, subset=None, shuffle=True, batch_size=32, class_mode="binary"):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.05,
        horizontal_flip=False,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )
    
    data_generator = datagen.flow_from_dataframe(
        dataframe=dataframe,
        x_col=x_col,
        y_col=y_col,
        subset=subset,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return data_generator



def get_base_model():
    base_model = tf.keras.applications.ResNet101(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    return base_model


def get_model():
    with tf.device('/GPU:0'):
        conv_base = get_base_model()
        conv_base.trainable = True

        x = conv_base.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Flatten(name='flatten')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1024, activation = 'relu')(x)
        x = tf.keras.layers.Dense(1024, activation = 'relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        preds = None
        if NUM_CLASSES > 2:
            preds = tf.keras.layers.Dense(units=NUM_CLASSES, activation = 'softmax')(x) # Ternary
        else:
            preds = tf.keras.layers.Dense(units=1, activation = 'sigmoid')(x) 
        
        model = tf.keras.Model(inputs=conv_base.input, outputs=preds)

        if NUM_CLASSES > 2:
            model.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy']) # Ternary
        else:
            model.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-5), loss='binary_crossentropy', metrics=[binary_accuracy]) 
            
        model.summary()
        return model
    

def train_model(model, df_train, df_val, epochs, fold='', axis=''):
    batch_size = 8
    mode = "categorical" if NUM_CLASSES > 2 else "binary"
    
    train_generator = get_data_generator(df_train, "id", "label", batch_size=batch_size, class_mode=mode)
    validation_generator = get_data_generator(df_val, "id", "label", batch_size=batch_size, class_mode=mode)

    step_size_train = train_generator.n // train_generator.batch_size
    step_size_validation = validation_generator.n // validation_generator.batch_size

    if step_size_train == 0:
        step_size_train = train_generator.n // 2
        step_size_validation = validation_generator.n // 2
        
    checkpoint_path = os.path.join(EXPERIMENTS_PATH, "weights", axis, "fold{}".format(fold), "my_checkpoint")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)

    history_path = os.path.join(EXPERIMENTS_PATH, "history", axis)
    if not os.path.exists(history_path):
        os.makedirs(history_path, exist_ok=True)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1,
        monitor='val_' + ACCURACY,
        mode='max',
        save_best_only=True
    )
        
    history = model.fit(train_generator, 
        steps_per_epoch=step_size_train,
        epochs=epochs, 
        validation_data=validation_generator,
        validation_steps=step_size_validation,
        callbacks=cp_callback
    )
    
    # Save history
    df_history = pd.DataFrame(history.history) 
    hist_csv_file = os.path.join(history_path, "history{}.csv".format(fold))
    with open(hist_csv_file, mode='w') as f:
        df_history.to_csv(f)
    
    # Save classes
    np.save(os.path.join(EXPERIMENTS_PATH, 'legend'), train_generator.class_indices)
    return history.history


def plot_results(history, axis, fold):
    acc = history[ACCURACY]
    val_acc = history['val_' + ACCURACY]
    loss = history['loss']
    val_loss = history['val_loss']

    image_path = os.path.join(LOG_PATH, axis)
    if not os.path.exists(image_path): 
        os.makedirs(image_path, exist_ok=True)
    
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and Validation accuracy') 
    plt.legend()
    
    plt.savefig(os.path.join(image_path, "{}_fold{}".format(ACCURACY, fold)), pad_inches=0.1)
    
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    
    plt.savefig(os.path.join(image_path, "loss_fold{}".format(fold)), orientation='portrait', pad_inches=0.1)

    plt.show()


# -------------------
# Main
# -------------------
for axis in AXIS_LIST:
    folds = [i+1 for i in range(max(1, NUM_FOLDS))]
    for num_fold in folds:
        print("{} - Fold: {}".format(axis, num_fold))
        
        train_path = os.path.join(INPUT_PATH, "train", "train{}.csv".format(num_fold))
        train_set = pd.read_csv(train_path)
        validation_path = os.path.join(INPUT_PATH, "val", "val{}.csv".format(num_fold))
        validation_set = pd.read_csv(validation_path)

        df_train, df_val = get_data_set(axis, train_set, validation_set)
        validation_set_dict = dict(zip(validation_set["patient"], validation_set["label"]))
        
        model= get_model()
        history = train_model(model, df_train, df_val, EPOCHS, num_fold, axis)
        
        #Plot Results
        plot_results(history, axis, num_fold)
