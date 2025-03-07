# For running inference on the TF-Hub module.
import tensorflow as tf

import tensorflow_hub as hub
from tensorflow.keras import datasets, layers, models

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# For measuring the inference time.
import time
from progress.bar import Bar

# Print Tensorflow version
print(tf.__version__)

# Image imports
from zipfile import ZipFile
import matplotlib.pyplot as plt
import pandas as pd

def display_image(image):
  fig = plt.figure(figsize=(20, 15))
  plt.grid(False)
  plt.imshow(image)


def download_and_resize_image(filepath, display=False):
  _, filename = tempfile.mkstemp(suffix=".jpg")
  pil_image = Image.open(filepath)
  pil_image_rgb = pil_image.convert("RGB")
  pil_image_rgb.save(filename, format="JPEG", quality=90)
  #print("Image downloaded to %s." % filename)
  if display:
    display_image(pil_image)
  return filename


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=1,
                               display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin

    
def crop_image(image,
                ymin,
                xmin,
                ymax,
                xmax,
                color,
                font,
                thickness=1,
                display_str_list=()):
  """Crop image around bounding"""

  im_width, im_height = image.size
  
  cropped = image.crop((xmin * im_width, ymin * im_height, xmax * im_width, ymax * im_height))

  return cropped


def draw_boxes(image, boxes, class_names, scores, max_boxes=5, min_score=0.1):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())

  try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                              25)
  except IOError:
    #print("Font not found, using default font.")
    font = ImageFont.load_default()

  ymin, xmin, ymax, xmax = tuple(boxes)
  display_str = "{}: {}%".format(class_names.decode("ascii"),
                                  int(100 * scores))
  #print(display_str)
  color = colors[hash(class_names) % len(colors)]
  image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
  
  crop = crop_image(
      image_pil,
      ymin,
      xmin,
      ymax,
      xmax,
      color,
      font,
      display_str_list=[display_str])
  
  crop = np.array(crop)
      
  return crop


def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img


def run_detector(detector, path):
  img = load_img(path)

  converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  start_time = time.time()
  result = detector(converted_img)
  end_time = time.time()

  result = {key:value.numpy() for key,value in result.items()}


  b = bytes('Person', 'utf-8')

  ind = np.where(result['detection_class_entities'] == b)

  if len(ind[0]) == 0:
    return img.numpy()

  ind = ind[0][0]

  image_with_boxes = draw_boxes(
      img.numpy(), result["detection_boxes"][ind],
      result["detection_class_entities"][ind], result["detection_scores"][ind])
  
  return image_with_boxes
