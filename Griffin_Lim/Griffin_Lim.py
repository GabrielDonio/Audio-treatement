#implementation de l'alogrithme de Griffin-Lim
import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf

def griffin_lim(spectrogram, n_iter=100, hop_length=512, win_length=1024):
    """Griffin-Lim classique"""
    magnitude = np.abs(spectrogram)
    n_fft = (magnitude.shape[0] - 1) * 2
    angles = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
    reconstructed_signal = magnitude * angles
    
    for _ in range(n_iter):
        temp_signal = librosa.istft(reconstructed_signal, hop_length=hop_length, win_length=win_length)
        temp_stft = librosa.stft(temp_signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        reconstructed_signal = magnitude * np.exp(1j * np.angle(temp_stft))
    
    signal = librosa.istft(reconstructed_signal, hop_length=hop_length, win_length=win_length)
    return signal

def griffin_lim_librosa(spectrogram, n_iter=100, hop_length=512, win_length=1024):
    """Utilise l'implémentation librosa"""
    signal = librosa.griffinlim(spectrogram, n_iter=n_iter, hop_length=hop_length, win_length=win_length)
    return signal

def fast_griffin_lim(spectrogram, n_iter=100, hop_length=512, win_length=1024):
    """Fast Griffin-Lim avec accélération FISTA"""
    magnitude = np.abs(spectrogram)
    n_fft = (magnitude.shape[0] - 1) * 2
    angles = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
    c0 = magnitude * angles
    
    # Initialisation t0
    t0_i = librosa.istft(c0, hop_length=hop_length, win_length=win_length)
    t0_s = librosa.stft(t0_i, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    t0 = magnitude * np.exp(1j * np.angle(t0_s))
    
    alpha = 1.0
    
    for _ in range(1, n_iter):
        t1_i = librosa.istft(c0, hop_length=hop_length, win_length=win_length)
        t1_s = librosa.stft(t1_i, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        t1 = magnitude * np.exp(1j * np.angle(t1_s))
        
        c0 = t1 + alpha * (t1 - t0)
        
        #FISTA
        alpha_new = (1 + np.sqrt(1 + 4 * alpha**2)) / 2
        alpha = (alpha - 1) / alpha_new
        
        t0 = t1  
    
    signal = librosa.istft(c0, hop_length=hop_length, win_length=win_length)
    return signal

def accelerate_griffin_lim(spectrogram, n_iter=100, hop_length=512, win_length=1024, alpha=0.99, beta=0.99, gamma=0.5):
    """
    Algorithme Accelerated Griffin-Lim
    
    Args:
        spectrogram: Spectrogramme de magnitude
        n_iter: Nombre d'itérations
        hop_length: Nombre d'échantillons entre frames
        win_length: Longueur de la fenêtre
        alpha, beta, gamma: Paramètres d'accélération (> 0)
    
    Returns:
        signal: Signal audio reconstruit
    """
    # Magnitude (constraint set C1)
    magnitude = np.abs(spectrogram)
    n_fft = (magnitude.shape[0] - 1) * 2
    
    random_phase = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
    c0 = magnitude * random_phase
    

    temp_signal = librosa.istft(c0, hop_length=hop_length, win_length=win_length)
    temp_stft = librosa.stft(temp_signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    t0 = magnitude * np.exp(1j * np.angle(temp_stft))

    d0 = t0.copy()
    
    t_prev = t0
    c_prev = c0
    d_prev = d0
    
    for iteration in range(1, n_iter):
        # PC2(cn-1) = STFT(ISTFT(cn-1))
        temp_signal = librosa.istft(c_prev, hop_length=hop_length, win_length=win_length)
        temp_stft = librosa.stft(temp_signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        # PC1(PC2(cn-1))
        pc1_pc2 = magnitude * np.exp(1j * np.angle(temp_stft))
        
        # tn = (1 - γ)dn-1 + γPC1(PC2(cn-1))
        t_n = (1 - gamma) * d_prev + gamma * pc1_pc2
        
        # cn = tn + α(tn - tn-1)
        c_n = t_n + alpha * (t_n - t_prev)
        
        # dn = tn + β(tn - tn-1)
        d_n = t_n + beta * (t_n - t_prev)
        
        t_prev = t_n
        c_prev = c_n
        d_prev = d_n
    
    signal = librosa.istft(c_n, hop_length=hop_length, win_length=win_length)
    return signal


# Test et comparaison des algorithmes
if __name__ == "__main__":
    # Chargement du fichier audio
    audio_file = "Data audio/705944__josefpres__guitar-tones-005-string-b-22-tone-a57.wav"
    signal_original, sr = librosa.load(audio_file, sr=None, mono=True)
    
    # Paramètres
    n_fft = 2048
    hop_length = 512
    win_length = 1024
    
    # Calculer le spectrogramme de magnitude
    stft_original = librosa.stft(signal_original, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    magnitude = np.abs(stft_original)
    
    print(f"Signal original: {len(signal_original)} échantillons")
    print(f"Spectrogramme: {magnitude.shape}")
    print("\n" + "="*50)
    
    # Test des 4 algorithmes
    print("\n1. Griffin-Lim classique")
    signal_gl = griffin_lim(magnitude, n_iter=100, hop_length=hop_length, win_length=win_length)
    sf.write("output_griffin_lim.wav", signal_gl, sr)
    print(f"   ✓ Sauvegardé: output_griffin_lim.wav")
    
    print("\n2. Griffin-Lim (librosa)")
    signal_gl_lib = griffin_lim_librosa(magnitude, n_iter=100, hop_length=hop_length, win_length=win_length)
    sf.write("output_griffin_lim_librosa.wav", signal_gl_lib, sr)
    print(f"   ✓ Sauvegardé: output_griffin_lim_librosa.wav")
    
    print("\n3. Fast Griffin-Lim")
    signal_fgl = fast_griffin_lim(magnitude, n_iter=100, hop_length=hop_length, win_length=win_length)
    sf.write("output_fast_griffin_lim.wav", signal_fgl, sr)
    print(f"   ✓ Sauvegardé: output_fast_griffin_lim.wav")
    
    print("\n4. Accelerated Griffin-Lim")
    signal_agl = accelerate_griffin_lim(magnitude, n_iter=100, hop_length=hop_length, win_length=win_length)
    sf.write("output_accelerated_griffin_lim.wav", signal_agl, sr)
    print(f"   ✓ Sauvegardé: output_accelerated_griffin_lim.wav")
    
    print("\n" + "="*50)
    print("✓ Tous les fichiers ont été générés avec succès!")
    
    # Visualisation
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 2, 1)
    plt.plot(signal_original[:10000])
    plt.title("Signal original")
    plt.xlabel("Échantillons")
    plt.ylabel("Amplitude")
    
    plt.subplot(3, 2, 2)
    plt.imshow(librosa.amplitude_to_db(magnitude), aspect='auto', origin='lower', cmap='viridis')
    plt.title("Spectrogramme de magnitude")
    plt.xlabel("Frames")
    plt.ylabel("Fréquence")
    plt.colorbar(label='dB')
    
    plt.subplot(3, 2, 3)
    plt.plot(signal_gl[:10000])
    plt.title("Griffin-Lim classique")
    plt.xlabel("Échantillons")
    plt.ylabel("Amplitude")
    
    plt.subplot(3, 2, 4)
    plt.plot(signal_gl_lib[:10000])
    plt.title("Griffin-Lim (librosa)")
    plt.xlabel("Échantillons")
    plt.ylabel("Amplitude")
    
    plt.subplot(3, 2, 5)
    plt.plot(signal_fgl[:10000])
    plt.title("Fast Griffin-Lim")
    plt.xlabel("Échantillons")
    plt.ylabel("Amplitude")
    
    plt.subplot(3, 2, 6)
    plt.plot(signal_agl[:10000])
    plt.title("Accelerated Griffin-Lim")
    plt.xlabel("Échantillons")
    plt.ylabel("Amplitude")
    
    plt.tight_layout()
    plt.show()
