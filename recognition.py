"""
=============================================================
  AI Recognition Project — Image OCR + Digit Classification
  Uses: pytesseract, OpenCV, scikit-learn, Pillow
=============================================================
"""

import os
import time
import textwrap
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
#  ANSI color helpers for terminal output
# ─────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA= "\033[95m"
    RED    = "\033[91m"
    BLUE   = "\033[94m"
    WHITE  = "\033[97m"
    DIM    = "\033[2m"

def banner(text, color=C.CYAN):
    width = 62
    line  = "─" * width
    print(f"\n{color}{C.BOLD}┌{line}┐")
    print(f"│  {text:<{width-1}}│")
    print(f"└{line}┘{C.RESET}")

def section(title, color=C.BLUE):
    print(f"\n{color}{C.BOLD}  ◆ {title}{C.RESET}")
    print(f"{C.DIM}  {'─'*55}{C.RESET}")

def result_box(label, value, color=C.GREEN):
    print(f"  {color}▸ {C.BOLD}{label:<20}{C.RESET} {value}")


# ═══════════════════════════════════════════════════════════
#  MODULE 1 — TEXT RECOGNITION (OCR)
# ═══════════════════════════════════════════════════════════

def preprocess_for_ocr(image_path: str) -> Image.Image:
    """Apply preprocessing pipeline to improve OCR accuracy."""
    img = Image.open(image_path).convert("RGB")

    # Resize if small
    w, h = img.size
    if w < 300:
        img = img.resize((w * 2, h * 2), Image.LANCZOS)

    # Enhance contrast and sharpness
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = ImageEnhance.Sharpness(img).enhance(2.0)

    return img


def extract_text(image_path: str) -> dict:
    """Run OCR and return structured results."""
    img = preprocess_for_ocr(image_path)

    # pytesseract configs
    configs = {
        "Standard":   "--psm 3",        # Fully automatic page segmentation
        "Single Block":"--psm 6",       # Assume single uniform block of text
    }

    best_text = ""
    best_conf = 0.0

    for mode_name, cfg in configs.items():
        raw = pytesseract.image_to_string(img, config=cfg).strip()
        # Confidence via data
        data = pytesseract.image_to_data(img, config=cfg,
                                         output_type=pytesseract.Output.DICT)
        confs = [c for c in data["conf"] if isinstance(c, (int, float)) and c > 0]
        avg_conf = sum(confs) / len(confs) if confs else 0

        if avg_conf > best_conf:
            best_conf = avg_conf
            best_text = raw

    words = [w for w in best_text.split() if w.strip()]
    lines = [l for l in best_text.splitlines() if l.strip()]

    return {
        "text":       best_text,
        "words":      words,
        "lines":      lines,
        "word_count": len(words),
        "confidence": round(best_conf, 1),
    }


def run_ocr_demo(samples_dir: str):
    banner("MODULE 1 — TEXT RECOGNITION (OCR)", C.CYAN)
    print(f"{C.DIM}  Engine: Tesseract 5 via pytesseract + Pillow preprocessing{C.RESET}")

    images = {
        "Business Card": "business_card.png",
        "Handwritten Note": "note.png",
        "Code Snippet":  "code_snippet.png",
    }

    for label, fname in images.items():
        path = os.path.join(samples_dir, fname)
        if not os.path.exists(path):
            print(f"  {C.RED}✗ Missing sample: {fname}{C.RESET}")
            continue

        section(label)
        t0 = time.time()
        res = extract_text(path)
        elapsed = time.time() - t0

        result_box("Confidence",  f"{res['confidence']}%",  C.GREEN)
        result_box("Words found", str(res["word_count"]),    C.GREEN)
        result_box("Lines found", str(len(res["lines"])),    C.GREEN)
        result_box("Time taken",  f"{elapsed:.3f}s",         C.DIM)

        print(f"\n  {C.YELLOW}{C.BOLD}Extracted Text:{C.RESET}")
        for line in res["lines"]:
            clean = line.strip()
            if clean:
                print(f"    {C.WHITE}{clean}{C.RESET}")


# ═══════════════════════════════════════════════════════════
#  MODULE 2 — HANDWRITTEN DIGIT CLASSIFICATION
# ═══════════════════════════════════════════════════════════

def train_digit_classifier():
    """Train an SVM on sklearn's 8×8 digit dataset."""
    digits = load_digits()
    X, y   = digits.data, digits.target

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = SVC(kernel="rbf", C=10, gamma=0.001, probability=True)
    clf.fit(X_train, y_train)

    y_pred   = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred,
                                     target_names=[str(i) for i in range(10)])

    return clf, scaler, digits, X_test, y_test, y_pred, accuracy, report


def render_digit(pixel_array_8x8: np.ndarray) -> str:
    """Render a digit as ASCII art in the terminal."""
    chars = " ░▒▓█"
    lines = []
    for row in pixel_array_8x8.reshape(8, 8):
        line = ""
        for val in row:
            idx = int(val / 16 * (len(chars) - 1))
            line += chars[idx] * 2
        lines.append("    " + line)
    return "\n".join(lines)


def run_digit_demo():
    banner("MODULE 2 — HANDWRITTEN DIGIT RECOGNITION (SVM)", C.MAGENTA)
    print(f"{C.DIM}  Model: RBF-SVM trained on sklearn Digits dataset (1797 samples, 8×8 px){C.RESET}")

    section("Training SVM classifier…", C.MAGENTA)
    t0 = time.time()
    clf, scaler, digits, X_test, y_test, y_pred, accuracy, report = train_digit_classifier()
    elapsed = time.time() - t0

    result_box("Training time",  f"{elapsed:.2f}s",       C.GREEN)
    result_box("Test accuracy",  f"{accuracy*100:.2f}%",  C.GREEN)
    result_box("Test samples",   str(len(y_test)),        C.GREEN)
    result_box("Classes",        "0 – 9 (10 digits)",     C.GREEN)

    # Show 10 individual predictions (one per digit class)
    section("Sample Predictions", C.MAGENTA)
    shown = set()
    raw_X  = digits.data   # original unscaled for display
    raw_Xs = scaler.transform(raw_X)

    for i in range(len(digits.target)):
        true_label = digits.target[i]
        if true_label in shown:
            continue

        proba   = clf.predict_proba(raw_Xs[i:i+1])[0]
        pred    = clf.predict(raw_Xs[i:i+1])[0]
        conf    = proba[pred] * 100
        correct = "✓" if pred == true_label else "✗"
        color   = C.GREEN if pred == true_label else C.RED

        print(f"\n  {C.BOLD}Digit sample — True label: {true_label}{C.RESET}")
        print(render_digit(digits.images[i]))
        print(f"\n  {color}{correct}  Predicted: {pred}   Confidence: {conf:.1f}%{C.RESET}")

        # Top 3 probabilities
        top3 = np.argsort(proba)[::-1][:3]
        print(f"  {C.DIM}Top-3: " +
              " | ".join(f"{t}→{proba[t]*100:.1f}%" for t in top3) +
              C.RESET)

        shown.add(true_label)
        if len(shown) == 10:
            break

    # Per-class summary
    section("Classification Report (per digit class)", C.MAGENTA)
    for line in report.strip().splitlines():
        if line.strip():
            print(f"  {C.DIM}{line}{C.RESET}")


# ═══════════════════════════════════════════════════════════
#  MODULE 3 — REAL-TIME IMAGE FEATURE DETECTION (OpenCV)
# ═══════════════════════════════════════════════════════════

def run_feature_demo(samples_dir: str):
    banner("MODULE 3 — IMAGE FEATURE ANALYSIS (OpenCV)", C.YELLOW)
    print(f"{C.DIM}  Analyzes edges, contours, and color histograms without a neural network{C.RESET}")

    images = {
        "Business Card": "business_card.png",
        "Handwritten Note": "note.png",
    }

    for label, fname in images.items():
        path = os.path.join(samples_dir, fname)
        if not os.path.exists(path):
            continue

        section(label, C.YELLOW)
        img     = cv2.imread(path)
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w    = gray.shape

        # Edge detection
        edges   = cv2.Canny(gray, 50, 150)
        n_edges = int(np.sum(edges > 0))
        edge_pct= round(n_edges / (h * w) * 100, 2)

        # Contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Brightness
        mean_bright = round(float(np.mean(gray)), 1)

        # Dominant color channel
        means = cv2.mean(img)[:3]  # B, G, R
        ch_names = ["Blue", "Green", "Red"]
        dominant = ch_names[int(np.argmax(means))]

        result_box("Dimensions",     f"{w} × {h} px",       C.YELLOW)
        result_box("Edge pixels",    f"{n_edges} ({edge_pct}%)", C.YELLOW)
        result_box("Contours",       str(len(contours)),    C.YELLOW)
        result_box("Mean brightness",f"{mean_bright}/255",  C.YELLOW)
        result_box("Dominant channel", dominant,            C.YELLOW)

        # Simple scene classification based on features
        if edge_pct > 5:
            scene = "Text-heavy / high contrast"
        elif edge_pct > 2:
            scene = "Mixed content"
        else:
            scene = "Low contrast / minimal content"

        result_box("Scene type",     scene,                 C.GREEN)


# ═══════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════

def main():
    SAMPLES = os.path.join(os.path.dirname(__file__), "samples")

    print(f"\n{C.BOLD}{C.CYAN}")
    print("  ██████╗ ███████╗ ██████╗ ██████╗  ██████╗ ")
    print("  ██╔══██╗██╔════╝██╔════╝██╔═══██╗██╔════╝ ")
    print("  ██████╔╝█████╗  ██║     ██║   ██║██║  ███╗")
    print("  ██╔══██╗██╔══╝  ██║     ██║   ██║██║   ██║")
    print("  ██║  ██║███████╗╚██████╗╚██████╔╝╚██████╔╝")
    print("  ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═════╝  ╚═════╝ ")
    print(f"\n     AI Image & Text Recognition Project{C.RESET}")
    print(f"  {C.DIM}  Python · OpenCV · pytesseract · scikit-learn{C.RESET}")

    run_ocr_demo(SAMPLES)
    run_digit_demo()
    run_feature_demo(SAMPLES)

    banner("ALL MODULES COMPLETE", C.GREEN)
    print(f"""
  {C.BOLD}Summary of techniques used:{C.RESET}
  {C.DIM}┌─────────────────────────────────────────────────────┐
  │  Module 1 │ OCR via Tesseract + Pillow preprocessing  │
  │  Module 2 │ SVM classifier on sklearn Digits dataset  │
  │  Module 3 │ Edge/contour/color analysis via OpenCV    │
  └─────────────────────────────────────────────────────┘{C.RESET}
""")


if __name__ == "__main__":
    main()
