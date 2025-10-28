import numpy as np
from PIL import Image, ImageFilter

def nearest_neighbor_interpolation(image, new_width, new_height):
    """
    Resize an image using nearest-neighbour interpolation.
    This is not about lowering resolution
    """
    # Get the dimensions and number of channels of the input image
    height, width, channels = image.shape

    # Compute the scale factors
    x_scale_factor = width / new_width
    y_scale_factor = height / new_height

    # Create a new image with the desired dimensions
    result = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    # Iterate over the pixels of the new image
    for y in range(new_height):
        for x in range(new_width):
            # Compute the coordinates of the pixel in the original image
            src_x = int(x * x_scale_factor)
            src_y = int(y * y_scale_factor)

            # Copy the pixel from the original image to the new image (i.e. perform nearest-neighbour interpolation)
            result[y, x] = image[src_y, src_x]

    return result

def bilinear_interpolation(image, new_width, new_height):
    # Get the dimensions and number of channels of the input image
    height, width, channels = image.shape

    # Compute the scale factors
    x_scale_factor = width / new_width
    y_scale_factor = height / new_height

    # Create a new image with the desired dimensions
    result = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    # Iterate over the pixels of the new image
    for y in range(new_height):
        for x in range(new_width):
            # Compute the coordinates of the pixel in the original image
            src_x = x * x_scale_factor
            src_y = y * y_scale_factor

            # Compute the coordinates of the four pixels surrounding the pixel in the original image
            x1 = int(src_x)
            y1 = int(src_y)
            x2 = min(x1 + 1, width - 1)
            y2 = min(y1 + 1, height - 1)

            # Compute the interpolation coefficients
            alpha = src_x - x1
            beta = src_y - y1

            # Perform bilinear interpolation
            for c in range(channels):
                result[y, x, c] = (
                    (1 - alpha) * (1 - beta) * image[y1, x1, c]
                    + alpha * (1 - beta) * image[y1, x2, c]
                    + (1 - alpha) * beta * image[y2, x1, c]
                    + alpha * beta * image[y2, x2, c]
                )

    return result

def _rgb_to_luminance(rgb):
    # rgb : HxWx3, valeurs 0-255
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b

def _apply_denoise_mask(norm, threshold=0.02, soft_mask=False):
    """
    norm : array 0-1
    threshold : valeur en [0,1] en dessous de laquelle on attenue / supprime
    soft_mask : si True, on applique une atténuation progressive plutôt que zéro dur
    """
    if threshold <= 0:
        return norm
    if soft_mask:
        # soft attenuation: scale values below threshold by (value/threshold)^2
        mask = np.ones_like(norm)
        low = norm < threshold
        # éviter division par zéro
        scaled = (norm[low] / (threshold + 1e-12))
        mask[low] = scaled * scaled  # intensifie l'atténuation pour les très faibles valeurs
        return norm * mask
    else:
        # hard threshold
        out = norm.copy()
        out[norm < threshold] = 0.0
        return out

def image_to_spectrogram(image_path, new_width, new_height, method='nearest', log_scale=True,
                         denoise=False, threshold=0.02, soft_mask=False, blur_radius=0,
                         save_path=None, return_magnitude=False):
    """
    Charge une image, redimensionne (nearest|bilinear), convertit en luminance,
    optionnellement applique un filtre de denoise (seuillage dur/soft + flou),
    puis renvoie soit une PIL.Image (niveaux de gris) soit un tableau numpy (magnitude 0-1).
    Args:
      denoise: bool, activer le seuillage
      threshold: float [0,1], valeurs en dessous seront atténuées/supprimées
      soft_mask: bool, si True on atténue progressivement au lieu de couper
      blur_radius: float, rayon de flou gaussian pour lisser (0 = pas de flou)
      return_magnitude: si True retourne un ndarray float 2D (H x W) en 0..1
    """
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    if method == 'nearest':
        resized = nearest_neighbor_interpolation(image_np, new_width, new_height)
    elif method == 'bilinear':
        resized = bilinear_interpolation(image_np, new_width, new_height)
    else:
        raise ValueError("Unsupported interpolation method. Use 'nearest' or 'bilinear'.")

    # Convertir en luminance (0-255)
    lum = _rgb_to_luminance(resized)  # shape: (H, W)

    # Normaliser en [0,1]
    norm = lum / 255.0

    # Optionnel : flou pour lisser le bruit avant seuillage
    if blur_radius and blur_radius > 0:
        pil_temp = Image.fromarray((norm * 255).astype(np.uint8), mode='L')
        pil_temp = pil_temp.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        norm = np.array(pil_temp).astype(np.float32) / 255.0

    # Appliquer le filtre de denoise (seuillage dur ou soft)
    if denoise:
        norm = _apply_denoise_mask(norm, threshold=threshold, soft_mask=soft_mask)

    # Petite valeur pour éviter log(0)
    eps = 1e-12
    if log_scale:
        db = 20.0 * np.log10(np.maximum(norm, eps))
        db_min = -80.0
        db_norm = (db - db_min) / (-db_min)
        db_norm = np.clip(db_norm, 0.0, 1.0)
        out_norm = db_norm
    else:
        out_norm = np.clip(norm, 0.0, 1.0)

    # Inverser verticalement pour correspondre au sens classique des spectrogrammes
    out_norm = np.flipud(out_norm)

    if return_magnitude:
        # retourne la magnitude normalisée (0..1) sous forme de ndarray float32
        return out_norm.astype(np.float32)

    # sinon construire image PIL pour affichage/sauvegarde
    out_img = Image.fromarray((out_norm * 255.0).astype(np.uint8), mode='L')
    if save_path:
        out_img.save(save_path)
    return out_img

def get_image_dimensions(image_path):
    """
    Récupérer les dimensions d'une image
    """
    image = Image.open(image_path)
    width, height = image.size
    return width, height
