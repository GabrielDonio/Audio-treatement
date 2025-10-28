import Griffin_Lim as glp
import numpy as np
import ImageTospectrogram as its
from PIL import Image
import soundfile as sf

image_path = "/Users/macos/Documents/Audio python/Data audio/canva-ginger-cat-with-paws-raised-in-air-MAGoQJ8-1Kc.jpg"
image_length,image_Height=its.get_image_dimensions(image_path)
spectrogram_image = its.image_to_spectrogram(image_path, image_length, image_Height,log_scale=False,save_path=None)
n_fft = 4096
hop_length = 64
win_length = 128
sr=44100
signal_gl = glp.griffin_lim(spectrogram_image, n_iter=100, hop_length=hop_length, win_length=win_length)
sf.write("output_griffin_lim.wav", signal_gl, sr)
print(f"   ✓ Sauvegardé: output_griffin_lim.wav")