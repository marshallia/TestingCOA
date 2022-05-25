import sys
import os
import numpy as np
from tqdm.keras import TqdmCallback
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, roc_curve
import json
import yaml

if len(sys.argv) != 4:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython evaluate.py model features model output_plot\n")
    sys.exit(1)
BASE_PATH = '/Users/marshallia/PycharmProjects/training'

params = yaml.safe_load(open(os.path.join(BASE_PATH, "params.yaml")))["evaluate"]
batch_size = params["batch_size"]

test_feature = os.path.join(sys.argv[2], "bottleneck_features_test.npy")
test_label_feature = os.path.join(sys.argv[2], "bottleneck_label_features_test.npy")
model_name = os.path.join(BASE_PATH, sys.argv[1])
plot_output = os.path.join(BASE_PATH, sys.argv[3])
prc_path = os.path.join(plot_output, 'prc.json')
roc_path = os.path.join(plot_output, 'roc.json')


def evaluate():
    model = tf.keras.models.load_model(model_name)
    test_data = np.load(open(test_feature, 'rb'))
    test_label = np.load(open(test_label_feature, 'rb'))

    evaluate = model.evaluate(test_data, test_label, batch_size=batch_size, return_dict=True,
                              callbacks=[TqdmCallback()])
    out_file = "test_result.json"
    with open(out_file, "w") as f:
        json.dump(evaluate, f,
                  indent=4, )
    prediction = model.predict(test_data, callbacks=[TqdmCallback()])

    precision, recall, thresholds = precision_recall_curve(test_label, prediction)
    save_ploting(prc_path, 'prc', precision, recall, thresholds, 
                 ['precision', 'recall', 'threshold'])

    fpr, tpr, thresholds = roc_curve(test_label, prediction)
    save_ploting(roc_path, 'roc', fpr, tpr, thresholds, ['fpr', 'tpr', 'threshold'])


def save_ploting(path, name, a, b, thresholds, keys):
    with open(path, "w") as fd:
        json.dump(
            {
                name: [
                    {keys[0]: str(p), keys[1]: str(r), keys[2]: str(t)}
                    for p, r, t in zip(a, b, thresholds)
                ]
            },
            fd,
            indent=4,
        )


evaluate()
