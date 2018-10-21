import io
import os
import logging
from PIL import Image
import tensorflow as tf

import utils

flags = tf.app.flags
flags.DEFINE_string('charset_path', 'ocr/charset.txt', 'Path to charset.')
flags.DEFINE_string('list_path', 'ocr/list_train.txt', 'Path to samples list.')
flags.DEFINE_string('data_dir', 'ocr/', 'Root directory to the dataset.')
flags.DEFINE_string('output_path', 'ocr/tfrecords', 'Path to output TFRecord.')
flags.DEFINE_integer('max_sequence_length', 0,
                     'The maximum of sequence length.')
FLAGS = flags.FLAGS


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def get_tf_example(image_path,
                   annotation_path,
                   charset_dict,
                   max_sequence_length,
                   null_code=0):
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    annotation = [line.strip() for line in open(annotation_path)][0]
    label = [null_code for i in range(max_sequence_length)]
    for 
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded': _bytes_feature(encoded_jpg),
                'image/height': _int64_feature([height]),
                'image/width': _int64_feature([width]),
                'image/class': _int64_feature(char_ids_padded),
                'image/text': _bytes_feature(bytes(annotation, 'utf-8')),
            }))
    return example


def main(_):
    if FLAGS.max_sequence_length == 0:
        FLAGS.max_sequence_length = utils.get_max_sequence_length(
            [sample[1] for sample in list_samples], FLAGS.data_dir)
        print('max_sequence_length: {}'.format(FLAGS.max_sequence_length))

    # gen tfrecord
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    charset_dict, null_code = utils.read_charset_index(FLAGS.charset_path)
    list_samples = [line.split() for line in open(FLAGS.list_path)]
    for idx, (image_path, annotation_path) in enumerate(list_samples):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(list_samples))
        image_path = os.path.join(FLAGS.data_dir, image_path)
        annotation_path = os.path.join(FLAGS.data_dir, annotation_path)
        tf_example = get_tf_example(image_path, annotation_path, charset_dict,
                                    max_sequence_length, null_code)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
