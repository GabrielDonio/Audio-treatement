from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image_path = 'path_to_your_image.jpg'  # Replace with your image path
img = Image.open(image_path).convert('L') 

img_array = np.array(img)
spectrogram = img_array / 255.0  