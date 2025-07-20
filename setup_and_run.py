#!/usr/bin/env python
"""setup_and_run.py

Helper script to test the RAG system setup and guide through the process.
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed."""
    required_packages = [
        'streamlit',
        'openai', 
        'unstructured',
        'numpy',
        'tqdm',
        'PIL'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package}")
    
    return missing

def check_environment():
    """Check environment setup."""
    print("\n🔧 Checking environment...")
    
    # Check OpenAI API key
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OPENAI_API_KEY is set")
    else:
        print("❌ OPENAI_API_KEY not found in environment")
        return False
    
    # Check PDF directory
    pdf_dir = Path("/Users/juan/Desktop/pdfs")
    if pdf_dir.exists():
        pdf_files = list(pdf_dir.glob("*.pdf"))
        print(f"✅ PDF directory found with {len(pdf_files)} PDF files")
    else:
        print(f"❌ PDF directory not found: {pdf_dir}")
        return False
    
    return True

def main():
    print("🚀 RAG System Setup Checker")
    print("=" * 50)
    
    print("\n📦 Checking required packages...")
    missing = check_requirements()
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return
    
    if not check_environment():
        print("\n❌ Environment setup incomplete")
        print("\nTo fix:")
        print("1. Set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        print("2. Make sure your PDFs are in /Users/juan/Desktop/pdfs/")
        return
    
    print("\n✅ All checks passed!")
    print("\n📋 Next steps:")
    print("1. Run: python process_pdfs_llamaparse.py")
    print("   (This will process your PDFs and create embeddings.pkl)")
    print("2. Run: streamlit run streamlit_app.py")
    print("   (This will start the RAG chat interface)")
    
    # Check if embeddings file exists
    if Path("embeddings.pkl").exists():
        print("\n🎉 embeddings.pkl found! You can skip step 1 and go directly to step 2.")

if __name__ == "__main__":
    main() 