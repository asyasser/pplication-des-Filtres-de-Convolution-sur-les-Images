import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def load_image(image_path, channel=3):
    """
    Load an image as grayscale (channel=1) or RGB (channel=3)
    with automatic validation
    
    Parameters:
    - image_path: Path to the image file
    - channel: 1 for grayscale, 3 for RGB (default)
    
    Returns:
    - Loaded image as numpy array
    """
    if channel == 1:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        assert image is not None, "Failed to load grayscale image"
        assert len(image.shape) == 2, "Image is not grayscale"
    else:
        image = cv2.imread(image_path)
        assert image is not None, "Failed to load RGB image"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        assert len(image.shape) == 3 and image.shape[2] == 3, "Image is not RGB"
    return image

def convolve_channel(image, kernel):
    """
    Apply convolution to a single channel image
    
    Parameters:
    - image: Single channel image (2D numpy array)
    - kernel: Convolution kernel (2D numpy array)
    
    Returns:
    - Convolved image
    """
    # Validate inputs
    assert isinstance(image, np.ndarray), "Image must be a NumPy array"
    assert isinstance(kernel, np.ndarray), "Kernel must be a NumPy array"
    assert len(kernel.shape) == 2, "Kernel must be 2D"
    assert kernel.shape[0] == kernel.shape[1], "Kernel must be square"
    assert kernel.shape[0] % 2 == 1, "Kernel must have odd dimensions"
    assert image.shape[0] >= kernel.shape[0], "Image too small for kernel height"
    assert image.shape[1] >= kernel.shape[1], "Image too small for kernel width"
    
    # Get padding sizes
    pad_size = kernel.shape[0] // 2
    padded = np.pad(image, pad_size, mode='constant')
    
    # Initialize output
    output = np.zeros_like(image, dtype=np.float32)
    
    # Apply convolution
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            output[i,j] = np.sum(region * kernel)
    
    return output

def apply_convolution(image, kernel):
    """
    Apply convolution to an image (grayscale or RGB)
    
    Parameters:
    - image: Input image (2D or 3D numpy array)
    - kernel: Convolution kernel (2D numpy array)
    
    Returns:
    - Convolved image
    """
    # Validate inputs
    assert isinstance(image, np.ndarray), "Image must be a NumPy array"
    assert isinstance(kernel, np.ndarray), "Kernel must be a NumPy array"
    assert len(kernel.shape) == 2, "Kernel must be 2D"
    assert kernel.shape[0] == kernel.shape[1], "Kernel must be square"
    assert kernel.shape[0] % 2 == 1, "Kernel must have odd dimensions"
    assert len(image.shape) in [2, 3], "Image must be grayscale (2D) or RGB (3D)"
    
    if len(image.shape) == 3:  # Image RGB
        assert image.shape[2] == 3, "RGB image must have 3 channels"
        output = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[2]):
            output[:,:,c] = convolve_channel(image[:,:,c], kernel)
    else:  # Image grayscale
        output = convolve_channel(image, kernel)
    
    # Normalize to [0, 255] and convert to uint8
    return np.clip(output, 0, 255).astype(np.uint8)

def display_images(original, filtered_images, titles):
    """
    Display original image with multiple filtered versions
    
    Parameters:
    - original: Original image
    - filtered_images: List of filtered images
    - titles: List of titles for filtered images
    """
    plt.figure(figsize=(15, 5))
    plt.subplot(1, len(filtered_images)+1, 1)
    plt.imshow(original, cmap='gray' if len(original.shape) == 2 else None)
    plt.title("Original")
    plt.axis('off')
    
    for i, (filt, title) in enumerate(zip(filtered_images, titles)):
        plt.subplot(1, len(filtered_images)+1, i+2)
        plt.imshow(filt, cmap='gray' if len(filt.shape) == 2 else None)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def create_output_directory():
    """Create output directory if it doesn't exist"""
    if not os.path.exists('output'):
        os.makedirs('output')

def main():
    # Create output directory
    create_output_directory()
    
    # Define all kernels
    kernels = {
        # Standard 3x3 kernels
        'Blur 3x3': np.array([
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9]
        ]),
        'Sobel Horizontal': np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]),
        'Sobel Vertical': np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]),
        'Sharpen': np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]),
        'Edge Detect': np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ]),
        'Emboss': np.array([
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ]),
        # Larger kernels
        'Blur 5x5': np.ones((5,5)) / 25,
        'Sobel H 5x5': np.array([
            [-1, -2, 0, 2, 1],
            [-2, -3, 0, 3, 2],
            [-3, -5, 0, 5, 3],
            [-2, -3, 0, 3, 2],
            [-1, -2, 0, 2, 1]
        ]),
        # Random kernel with normalization
        'Random': (np.random.rand(3,3) - 0.5) / np.sum(np.abs(np.random.rand(3,3) - 0.5))
    }
    
    # Load images (replace with your image path)
    image_path = 'baboon.jpg'  # Example image name
    try:
        gray_image = load_image(image_path, channel=1)
        rgb_image = load_image(image_path, channel=3)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Process grayscale image
    gray_results = []
    gray_titles = []
    for name, kernel in kernels.items():
        try:
            filtered = apply_convolution(gray_image, kernel)
            gray_results.append(filtered)
            gray_titles.append(name)
            
            # Save grayscale results
            cv2.imwrite(f'output/gray_{name.lower().replace(" ", "_")}.jpg', filtered)
        except Exception as e:
            print(f"Error processing grayscale with {name}: {e}")
    
    # Process RGB image
    rgb_results = []
    rgb_titles = []
    for name, kernel in kernels.items():
        try:
            filtered = apply_convolution(rgb_image, kernel)
            rgb_results.append(filtered)
            rgb_titles.append(name)
            
            # Save RGB results (convert back to BGR for OpenCV)
            rgb_save = cv2.cvtColor(filtered, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'output/rgb_{name.lower().replace(" ", "_")}.jpg', rgb_save)
        except Exception as e:
            print(f"Error processing RGB with {name}: {e}")
    
    # Display results (showing first 5 filters to avoid crowding)
    display_images(gray_image, gray_results[:5], gray_titles[:5])
    display_images(rgb_image, rgb_results[:5], rgb_titles[:5])
    
    print("Processing complete. Results saved in 'output' directory.")

if __name__ == "__main__":
    main()
