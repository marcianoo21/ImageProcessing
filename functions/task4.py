import numpy as np
import cmath

def DFT(image):
    rows, cols = image.shape
    result = np.zeros((rows, cols), dtype=complex)
    for u in range(rows):
        for v in range(cols):
            sum_pixel = 0
            for x in range(rows):
                for y in range(cols):
                    sum_pixel += image[x, y] * np.exp(-2j * np.pi * ((u * x / rows) + (v * y / cols)))
            result[u, v] = sum_pixel
    return result


def slow_fourier_transform(img):
    M, N = img.shape
    F = np.zeros((M, N), dtype=complex)
    for u in range(M):
        for v in range(N):
            sum_value = 0
            for x in range(M):
                for y in range(N):
                    sum_value += img[x, y] * np.exp(-2j * np.pi * ((u * x / M) + (v * y / N)))
            F[u, v] = sum_value
    return F

def slow_inverse_fourier_transform(F):
    M, N = F.shape
    img = np.zeros((M, N), dtype=complex)
    for x in range(M):
        for y in range(N):
            sum_value = 0
            for u in range(M):
                for v in range(N):
                    sum_value += F[u, v] * np.exp(2j * np.pi * ((u * x / M) + (v * y / N)))
            img[x, y] = sum_value / (M * N)
    return np.abs(img)


def fft_recursive(signal):
    N = len(signal)
    if N <= 1:  
        return signal
    even = fft_recursive(signal[0::2])
    odd = fft_recursive(signal[1::2])
    combined = [0] * N
    for k in range(N // 2):
        t = np.exp(-2j * np.pi * k / N) * odd[k]
        combined[k] = even[k] + t
        combined[k + N // 2] = even[k] - t
    return combined

def ifft_recursive(signal):
    N = len(signal)
    if N <= 1: 
        return signal
    even = ifft_recursive(signal[0::2])
    odd = ifft_recursive(signal[1::2])
    combined = [0] * N
    for k in range(N // 2):
        t = np.exp(2j * np.pi * k / N) * odd[k]
        combined[k] = (even[k] + t) / 2
        combined[k + N // 2] = (even[k] - t) / 2
    return combined


def fft_2d(img):
    rows_fft = np.array([fft_recursive(row) for row in img])
    return np.array([fft_recursive(col) for col in rows_fft.T]).T

def ifft_2d(F):
    rows_ifft = np.array([ifft_recursive(row) for row in F])
    return np.array([ifft_recursive(col) for col in rows_ifft.T]).T
