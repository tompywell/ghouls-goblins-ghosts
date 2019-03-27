from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import data

batch_size = 10
train_steps = 1000
num_layers = 2
num_nodes = 10

# get data from data.py
(train_x, train_y), (test_x, test_y) = data.load_data()

# define the feature columns, simply all columns from training data
feature_columns = [tf.feature_column.numeric_column(key=key) for key in train_x.keys()]

# make the classifier
classifier = tf.estimator.DNNClassifier(
    feature_columns = feature_columns,
    hidden_units = [num_nodes] * num_layers,
    n_classes = len(data.TYPES)
)

# train the classifier
classifier.train(
    input_fn = lambda:data.train_input_fn(train_x, train_y, batch_size),
    steps = train_steps
)

# evaluate the classifier
evaluation = classifier.evaluate(input_fn = lambda:data.eval_input_fn(test_x, test_y, batch_size))

# chekc out the accuracy
print('\naccuracy:', evaluation['accuracy'], '\n')

# hopefully this gets labeled as a ghost
predict_features = {
    'bone_length': [0.0],
    'rotting_flesh': [0.0],
    'hair_length': [0.0],
    'has_soul': [0.0],
    'black': [0.0],
    'blue': [0.0],
    'clear': [1.0],
    'white': [0.0],
    'green': [0.0],
    'blood': [0.0]
}

# make a prediction for the features above
predicted = classifier.predict(
    input_fn = lambda:data.eval_input_fn(predict_features, labels = None, batch_size = batch_size)
)

predicted = list(predicted)
class_id = predicted[0]['class_ids'][0]
probability = predicted[0]['probabilities'][class_id]
type = data.TYPES[class_id]

# is ghost?
print('\npredicted', type, 'with probabilty', probability)
