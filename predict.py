import os
import cv2
import numpy as np
import pandas as pd
import joblib
from skimage.feature import hog, daisy, local_binary_pattern
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

# Constants (match your training script)
TEST_IMAGES_DIR = 'test_ims/'
TEST_CSV = 'test.csv'
IMG_SIZE = (32, 32)

class LDA_encoder:
    """LDA-based feature encoder (same as in training)."""
    def __init__(self, n_components=9):
        self.lda = LinearDiscriminantAnalysis(n_components=n_components)
    
    def fit(self, X, y):
        self.lda.fit(X, y)
        self.class_centers = np.array([
            np.mean(self.lda.transform(X[y == i]), axis=0)
            for i in np.unique(y)
        ])
    
    def encode(self, X):
        return 1 / np.array([
            np.linalg.norm(self.lda.transform(X) - center, axis=1)
            for center in self.class_centers
        ]).T

# Load saved model and scalers
print("Loading saved models...")
svm = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
lda_encoder = joblib.load("lda_encoder.pkl")

# Feature extraction functions (must match training)
def blur(img, kernel_size=4):
    """Blur image by averaging kernel_size x kernel_size blocks."""
    h, w, c = img.shape
    blurred = np.mean(img.reshape(h//kernel_size, kernel_size, 
                                w//kernel_size, kernel_size, c), 
                    axis=(1, 3))
    return blurred.flatten()

def eoh(img, num_bins=9, cell_size=8):
    """Edge Orientation Histograms."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    angle = np.arctan2(sobely, sobelx) % (2 * np.pi)
    
    num_cells_x = gray.shape[1] // cell_size
    num_cells_y = gray.shape[0] // cell_size
    histograms = []
    
    for i in range(num_cells_y):
        for j in range(num_cells_x):
            cell_mag = magnitude[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell_angle = angle[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            hist, _ = np.histogram(cell_angle, bins=num_bins, range=(0, 2*np.pi), weights=cell_mag)
            histograms.append(hist)
    
    return np.concatenate(histograms)

def extract_features(img):
    """Extract features for a single image (must match training)."""
    # Ensure proper shape
    if len(img.shape) == 3 and img.shape[2] == 3:
        pass
    else:
        img = img.reshape(IMG_SIZE[0], IMG_SIZE[1], 3)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # HOG
    hog_feat = hog(gray, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1))
    
    # LBP
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist_lbp = np.histogram(lbp, bins=np.arange(0, 10), range=(0, 9))[0]
    
    # Daisy
    daisy_feat = daisy(gray, step=7, radius=7, rings=1, histograms=6, orientations=8).flatten()
    
    # EOH
    eoh_feat = eoh(img)
    
    # Color stats
    color_feat = []
    for channel in cv2.split(img):  # BGR
        color_feat.extend([np.mean(channel), np.std(channel)])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for channel in cv2.split(hsv):  # HSV
        color_feat.extend([np.mean(channel), np.std(channel)])
    
    # Blurred
    blurred_feat = blur(img)
    
    return np.hstack([hog_feat, hist_lbp, daisy_feat, eoh_feat, color_feat, blurred_feat])

def predict_test_images():
    """Main prediction function."""
    # Load test data
    test_df = pd.read_csv(TEST_CSV)
    test_images = []
    
    print("Loading test images...")
    for img_name in tqdm(test_df['im_name']):
        img_path = os.path.join(TEST_IMAGES_DIR, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, IMG_SIZE)
        test_images.append(img)
    
    # Extract features
    print("\nExtracting features...")
    test_features = []
    for img in tqdm(test_images):
        features = extract_features(img)
        test_features.append(features)
    test_features = np.array(test_features)
    
    # Preprocess (scale + LDA)
    print("\nPreprocessing features...")
    test_scaled = scaler.transform(test_features)
    test_encoded = np.hstack([test_scaled, lda_encoder.encode(test_scaled)])
    
    # Predict
    print("\nMaking predictions...")
    predictions = svm.predict(test_encoded)
    
    # Save results
    test_df['label'] = predictions
    test_df.to_csv('predictions.csv', index=False)
    print("\nPredictions saved to predictions.csv")

if __name__ == "__main__":
    predict_test_images()