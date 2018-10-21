import os
import time
import random
from PIL import Image, ImageDraw, ImageFont


def gen_date(time_start=(1900, 1, 1, 0, 0, 0, 0, 0, 0),
             time_end=(2018, 12, 31, 23, 59, 59, 0, 0, 0)):
    time_start = time.mktime(time_start)
    time_end = time.mktime(time_end)
    time_new = random.randint(time_start, time_end)
    time_new = time.localtime(time_new)
    return time.strftime("%Y-%m-%d", time_new)


def draw_text(text,
              font,
              loc=(0, 0),
              size=(256, 128),
              fg_color=(0, 0, 0),
              bg_color=(255, 255, 255)):
    im = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(im)
    draw.text(loc, text, fg_color, font)
    return im


if __name__ == '__main__':
    images_dir = 'images'
    annotations_dir = 'annotations'
    list_train_path = 'list_train.txt'
    list_test_path = 'list_test.txt'
    charset_path = 'charset.txt'
    n_samples = 1000

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)

    # generate images and annotations
    font = ImageFont.truetype('simsun.ttc', 36)
    for i in range(n_samples):
        text = gen_date()
        annotation_path = os.path.join(annotations_dir, '{}.txt'.format(i + 1))
        open(annotation_path, 'w').write(text)
        im = draw_text(text, font,
                       (random.randint(0, 76), random.randint(0, 92)))
        image_path = os.path.join(images_dir, '{}.jpg'.format(i + 1))
        im.save(image_path)

    # generate lists
    n_split = int(.9 * n_samples)
    fd = open(list_train_path, 'w')
    for i in range(n_split):
        image_path = os.path.join(images_dir, '{}.jpg'.format(i + 1))
        annotation_path = os.path.join(annotations_dir, '{}.txt'.format(i + 1))
        fd.write('{} {}\n'.format(image_path, annotation_path))
    fd.close()
    fd = open(list_test_path, 'w')
    for i in range(n_split, n_samples):
        image_path = os.path.join(images_dir, '{}.jpg'.format(i + 1))
        annotation_path = os.path.join(annotations_dir, '{}.txt'.format(i + 1))
        fd.write('{} {}\n'.format(image_path, annotation_path))
    fd.close()

    # generate charset
    fd = open(charset_path, 'w')
    fd.write('0 <nul>\n')
    for i in range(10):
        fd.write('{} {}\n'.format(i + 1, i))
    fd.write('11 -\n')
    fd.close()
