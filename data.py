import pandas as pd
import tensorflow as tf

# what are the names for our columns?
COLUMN_NAMES = ['id','bone_length','rotting_flesh','hair_length','has_soul','color','type']

# what are the possible values for the labels?
TYPES = ['Ghoul', 'Ghost', 'Goblin']

# turn 'color' features to 6 one-hot features
def to_one_hot_color(data):
    color = data.pop('color')
    data['clear'] = (color == 'clear') * 1.0
    data['green'] = (color == 'green') * 1.0
    data['black'] = (color =='black') * 1.0
    data['white'] = (color == 'white') * 1.0
    data['blue'] = (color == 'blue') * 1.0
    data['blood'] = (color == 'blood') * 1.0
    return data

# function to load all of the data from the csv
def load_data():
    # read data from csv
    data = pd.read_csv('./data/data.csv', names=COLUMN_NAMES, header=0)

    # replace color feature to individual one-hot features
    data = to_one_hot_color(data)

    # replace label strings with ints
    data['type'] = data['type'].apply(lambda x: TYPES.index(x))

    # get rid of id feature, not needed
    data.pop('id')

    # split data into training and testing sets
    train = data.sample(frac=0.8, random_state=44)
    test = data.drop(train.index)
    return (train, train.pop('type')), (test, test.pop('type'))

# simple input function for training
def train_input_fn(features, labels, batch_size):
    return tf.data.Dataset.from_tensor_slices((dict(features), labels)).shuffle(1000).repeat().batch(batch_size)

# simple input function for evaluation
def eval_input_fn(features, labels, batch_size):
    features = dict(features)
    inputs = features if labels is None else (features, labels)
    return tf.data.Dataset.from_tensor_slices(inputs).batch(batch_size)
