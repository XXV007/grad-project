"""Extract and analyze PDF detection results"""
import sys
try:
    import pypdf
    print("Using pypdf library")
    with open(r"c:\Users\vishn\Downloads\Detection Results 1.pdf", 'rb') as f:
        reader = pypdf.PdfReader(f)
        print(f"\nPDF Pages: {len(reader.pages)}\n")
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            print(f"{'='*70}")
            print(f"Page {i+1}")
            print(f"{'='*70}")
            if text:
                print(text)
            else:
                print("[No extractable text]")
            print()
except ImportError:
    print("pypdf not installed, trying pdfplumber...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "pypdf", "-q"])
    import pypdf
    with open(r"c:\Users\vishn\Downloads\Detection Results 1.pdf", 'rb') as f:
        reader = pypdf.PdfReader(f)
        print(f"\nPDF Pages: {len(reader.pages)}\n")
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            print(f"{'='*70}")
            print(f"Page {i+1}")
            print(f"{'='*70}")
            print(text[:1000] if text else "[No text]")
            print()
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
