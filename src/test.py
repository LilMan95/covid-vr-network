import glob
import os
import random
import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.metrics import binary_accuracy

import plot_metrics as pm


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

IMG_WIDTH, IMG_HEIGHT = config["MODEL"]["INPUT_SIZE"] 
NUM_CLASSES = config["MODEL"]["NUM_CLASSES"]
NUM_FOLDS = config["MODEL"]["NUM_FOLDS"]

# paths
SRC_PATH = os.path.abspath(".")
LOG_PATH = os.path.join(SRC_PATH, "..", "logs", DATASET_NAME)
EXPERIMENTS_PATH = os.path.join(SRC_PATH, "..", "experiments", DATASET_NAME)

INPUT_PATH = os.path.join(SRC_PATH, "..", DATA_ROOT_INPUT, DATASET_NAME)
OUTPUT_PATH = os.path.join(SRC_PATH, "..", DATA_ROOT_OUTPUT, DATASET_NAME)


def predictions_by_patient(model, patients, legend_path, axis):
    df_val_all = []
    class_indices = {}
    print(legend_path)
    if os.path.isfile(legend_path):
        class_indices = np.load(legend_path, allow_pickle=True).item()
        class_indices = dict((v,k) for k,v in class_indices.items())
    
    for p in patients:
        images_path = os.path.join(IMAGES_PATH, p, axis)
        files = sorted(os.listdir(images_path))[:]
        df_val = pd.DataFrame(dict(filename=files))
        nb_samples = df_val.shape[0]

        val_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        val_generator = val_gen.flow_from_dataframe(
                df_val, 
                images_path, 
                x_col='filename',
                y_col=None,
                class_mode=None,
                target_size=(IMG_WIDTH, IMG_HEIGHT),
                batch_size=16,
                shuffle=False
        )

        predict = model.predict(val_generator, steps=np.ceil(nb_samples/16))

        if NUM_CLASSES == 2:
            df_val['predicted'] = [ round(pr[0]) for pr in predict]
        else:
            df_val['predicted'] = [np.where(pr == np.max(pr))[0][0] for pr in predict]

        # format to name
        for key in class_indices.keys():
            t = class_indices[key]
            df_val[t] = 0
            df_val.loc[df_val['predicted'] == key, t] = 1
        df_val['patient'] = p
        df_val['axis'] = axis
        df_val_all.append(df_val)

    df_result = pd.DataFrame()
    for _, df_val in enumerate(df_val_all):
        patient_name = df_val['patient'][0]
        df_val['predicted'] = 0
        df_val['patient'] = 0
        df_val = df_val.groupby(['patient', 'predicted'], as_index = False).sum()
        df_val['predicted'] = df_val.idxmax(axis=1)
        df_val['patient'] = patient_name
        df_result = df_result.append(df_val)
        
    return df_result
        

if __name__ == "__main__":
    # Reading label dicts
    temp_data = pd.read_csv(os.path.join(INPUT_PATH, "val", "val1.csv"))
    labels = temp_data["label"].unique()
    labels.sort()
    labels_length = len(labels)
    labels_pos_dict = dict(zip(labels, [i for i in range(labels_length)]))
    legend_path = os.path.join(EXPERIMENTS_PATH, "legend.npy")


    for axis in AXIS_LIST:
        output_path = os.path.join(OUTPUT_PATH, axis)
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        # Predicted class for patient and fold
        df_axis = pd.DataFrame(columns=["fold", "patient", "real", "predicted"])
        confusion_matrix = np.zeros(labels_length*labels_length).reshape(labels_length, labels_length)

        folds = [i+1 for i in range(max(1, NUM_FOLDS))]
        for fold in folds:
            print("{} - Fold: {}".format(axis, fold))
            
            path = os.path.join(INPUT_PATH, "val", "val{}.csv".format(fold))
            val_data = pd.read_csv(path)
            val_dict = dict(zip(val_data["patient"], val_data["label"]))

            path = os.path.join(EXPERIMENTS_PATH, "weights", axis, "fold{}".format(fold), "my_checkpoint")
            model = tf.keras.models.load_model(path)

            df = predictions_by_patient(model, val_dict.keys(), legend_path, axis)
            df.reset_index(drop=True, inplace=True)
            df['fold'] = fold
            df['real'] = ''

            # Populating confusion matrix
            for index, row in df.iterrows():
                label = val_dict[row['patient']]
                df.loc[index, 'real'] = label
                real_label_pos = labels_pos_dict[label]
                pred_label_pos = labels_pos_dict[row['predicted']]
                confusion_matrix[real_label_pos][pred_label_pos] += 1

            df_axis = df_axis.append(df)
            del model

        output_path = os.path.join(OUTPUT_PATH)
        if not os.path.exists(output_path): 
            os.makedirs(output_path, exist_ok=True)

        df_axis.to_csv(os.path.join(output_path, "{}.csv".format(axis)), index=False)

        print(labels_pos_dict)
        print(confusion_matrix)

        pm.plot_labels_metrics(
                    cm=confusion_matrix,
                    normalize=False,
                    labels=labels,
                    show_zero=False,
                    title='Metrics - ' + axis,
                    clear_diagonal=False,
                    figsize=(15, 105),
                    output_file=os.path.join(output_path, "m_{}.png".format(axis))
                )
        pm.plot_confusion_matrix(
                    cm=confusion_matrix,
                    normalize=False,
                    labels=labels,
                    show_zero=False,
                    title="Confusion Matrix - " + axis,
                    clear_diagonal=False,
                    output_file=os.path.join(output_path, "cm_{}.png".format(axis)),
                    figsize=(10, 7)
                )