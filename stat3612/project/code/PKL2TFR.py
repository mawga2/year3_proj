import pickle
import tensorflow as tf

# Load the pickle file
file_path = 'ehr_preprocessed_seq_by_day_cat_embedding.pkl'
with open(file_path, 'rb') as file:
    ehr_data = pickle.load(file)

# Extract components
feat_dict = ehr_data['feat_dict']
feature_cols = ehr_data['feature_cols']

# Helper function to create a TensorFlow Example for each patient-day record
def create_tf_example(patient_id, day, features):
    feature = {
        'patient_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(patient_id).encode()])),
        'day': tf.train.Feature(int64_list=tf.train.Int64List(value=[day])),
    }
    for feature_name, value in features.items():
        feature[feature_name] = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    return tf.train.Example(features=tf.train.Features(feature=feature))

# Write data to TFRecord
tfrecord_file_path = 'ehr_data.tfrecord'
with tf.io.TFRecordWriter(tfrecord_file_path) as writer:
    for patient_id, patient_data in feat_dict.items():
        for day, daily_data in enumerate(patient_data):
            features = dict(zip(feature_cols, daily_data))
            tf_example = create_tf_example(patient_id, day, features)
            writer.write(tf_example.SerializeToString())

print("Data saved to 'ehr_data.tfrecord'")