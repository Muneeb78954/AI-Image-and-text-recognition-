Module 1 — OCR Text Recognition

Uses pytesseract (Tesseract 5 engine) + Pillow for image preprocessing
Runs on 3 sample images: a business card, a handwritten note, and a code snippet
Reports confidence score, word/line counts, and extracted text per image

Module 2 — Handwritten Digit Classification

Trains an RBF-SVM (Support Vector Machine) on sklearn's built-in Digits dataset (1,797 samples of 8×8 pixel images, digits 0–9)
Achieves 98.89% accuracy in under 0.3 seconds
Displays each digit as ASCII art in the terminal with predicted label + confidence + top-3 probabilities

Module 3 — Image Feature Analysis (OpenCV)

No model needed — uses classical computer vision
Detects edges (Canny), counts contours, measures brightness and dominant color channel
Classifies scene type (text-heavy, mixed, minimal)

How to run

# Install dependencies (most likely already present)
pip install pytesseract pillow scikit-learn opencv-python

# Run the project
cd recognition_project
python3 recognition.py

The samples/ folder contains the 3 pre-generated test images. You can swap in your own images for the OCR module — just update the paths in the images dict inside run_ocr_demo().
