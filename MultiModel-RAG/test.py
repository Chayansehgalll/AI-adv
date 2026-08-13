import os
import pytesseract

# Set paths
os.environ["PATH"] += r";C:\Program Files\Tesseract-OCR"
os.environ["PATH"] += r";C:\Program Files\poppler\Library\bin"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Test imports
try:
    from unstructured.partition.pdf import partition_pdf
    print("✅ unstructured imported successfully")
except Exception as e:
    print(f"❌ unstructured failed: {e}")

try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
except Exception as e:
    print(f"❌ PyTorch failed: {e}")

try:
    import cv2
    print(f"✅ OpenCV version: {cv2.__version__}")
except Exception as e:
    print(f"❌ OpenCV failed: {e}")

try:
    import pytesseract
    print(f"✅ Tesseract version: {pytesseract.get_tesseract_version()}")
except Exception as e:
    print(f"❌ Tesseract failed: {e}")

print("\n✅ All checks complete!")