import io
from google.cloud import vision
from google.cloud.vision import types
from PIL import Image, ImageDraw
import sys
import glob
import os
import json

image_dir = sys.argv[1]
annotation_dir = sys.argv[2]
json_dir = sys.argv[3]

dirlist = []
dirlist = glob.glob(os.path.join(image_dir, '*.jpg'))

client = vision.ImageAnnotatorClient()
for file in dirlist:
    with io.open(file, 'rb') as image_file:
        content = image_file.read()
        image = types.Image(content=content)
        response = client.document_text_detection(image=image)
        document = response.full_text_annotation

        # Make a plain text file for each image that contains the JSON-ish data
        json_out = os.path.join(json_dir, os.path.basename(file).rstrip('.jpg')+'.txt')
        with open(json_out, 'w') as outfile:
            outfile.write(str(document))
            outfile.close()

        # Make a tab-delimited file for each image that contains the OCR data
        out = os.path.join(annotation_dir, os.path.basename(file).rstrip('.jpg')+ '.tsv')
        with open(out, 'w') as outfile:
            outfile.write('\t'.join(['content', 'x', 'y', 'w', 'h', '\n']))
            for page in document.pages:
                for block in page.blocks:
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            first_symbol = word.symbols[0]
                            last_symbol = word.symbols[-1]
                            x = first_symbol.bounding_box.vertices[0].x
                            y = first_symbol.bounding_box.vertices[0].y
                            lower_x = last_symbol.bounding_box.vertices[2].x
                            lower_y = last_symbol.bounding_box.vertices[2].y
                            w = abs(lower_x - x)
                            h = abs(lower_y - y)
                            content = ''.join([i.text for i in word.symbols])
                            if content == '':
                                pass
                            else:
                                outfile.write('\t'.join([content, str(x), str(y), str(w), str(h), '\n']))


'''
bounding_box {
  vertices {
    x: 827 # top left
    y: 186
  }
  vertices {
    x: 906 # top right
    y: 186
  }
  vertices {
    x: 906 # bottom right
    y: 421
  }
  vertices {
    x: 827 # bottom left
    y: 421
  }
}
'''
