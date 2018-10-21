import os
import re
import logging
import tensorflow as tf


def get_max_sequence_length(list_names, data_dir):
    max_sequence_length = 0
    for name in list_names:
        path = os.path.join(data_dir, name)
        annotation = [line.strip() for line in open(path)][0]
        max_sequence_length = max(max_sequence_length, len(annotation))
    return max_sequence_length


def read_charset(filename, null_character=u'\u2591'):
    """Reads a charset definition from a tab separated text file.

    charset file has to have format compatible with the dataset.

    Args:
        filename: a path to the charset file.
        null_character: a unicode character used to replace '<null>' character.
            the default value is a light shade block '░'.

    Returns:
        a dictionary with keys equal to character codes and values - unicode
        characters.
    """
    charset = dict()
    pattern = re.compile(r'(\d+) (.+)')
    with tf.gfile.GFile(filename) as f:
        for i, line in enumerate(f):
            m = pattern.match(line)
            if m is None:
                logging.warning('incorrect charset file. line #%d: %s', i,
                                line)
                continue
            code = int(m.group(1))
            char = m.group(2).decode('utf-8')
            if char == '<nul>':
                char = null_character
            charset[code] = char
    return charset


def read_charset_index(filename, null_character=u'\u2591'):
    """Reads a charset definition from a tab separated text file.

    charset file has to have format compatible with the dataset.

    Args:
        filename: a path to the charset file.
        null_character: a unicode character used to replace '<null>' character.
            the default value is a light shade block '░'.

    Returns:
    a dictionary with keys equal to character values and codes.
    """
    null_code = -1
    charset = dict()
    pattern = re.compile(r'(\d+)\t(.+)')
    with tf.gfile.GFile(filename) as f:
        for i, line in enumerate(f):
            m = pattern.match(line)
            if m is None:
                logging.warning('incorrect charset file. line #%d: %s', i,
                                line)
                continue
            code = int(m.group(1))
            char = m.group(2).decode('utf-8')
            if char == '<nul>':
                char = null_character
                null_code = code
            charset[char] = code
    return charset, null_code
