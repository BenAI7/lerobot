"""
Test that all Phase 1 dependencies are installed.
Run: python sail/test_dependencies.py
"""

def test_imports():
    print("=" * 60)
    print("SAIL Phase 1 - Dependency Check")
    print("=" * 60)
    
    all_passed = True
    
    # PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  VRAM: {vram:.1f} GB")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        all_passed = False
    
    # Transformers
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers: {e}")
        print("  Install with: pip install transformers>=4.36.0")
        all_passed = False
    
    # TIMM
    try:
        import timm
        print(f"✓ TIMM {timm.__version__}")
    except ImportError as e:
        print(f"✗ TIMM: {e}")
        print("  Install with: pip install timm>=0.9.12")
        all_passed = False
    
    # PIL
    try:
        from PIL import Image
        import PIL
        print(f"✓ Pillow (PIL) {PIL.__version__}")
    except ImportError as e:
        print(f"✗ Pillow: {e}")
        print("  Install with: pip install Pillow>=10.0.0")
        all_passed = False
    
    # Accelerate
    try:
        import accelerate
        print(f"✓ Accelerate {accelerate.__version__}")
    except ImportError as e:
        print(f"✗ Accelerate: {e}")
        print("  Install with: pip install accelerate>=0.25.0")
        all_passed = False
    
    # BitsAndBytes (optional on Windows)
    try:
        import bitsandbytes
        print(f"✓ BitsAndBytes (for 4-bit quantization)")
    except ImportError as e:
        print(f"⚠️  BitsAndBytes: {e}")
        print("  Note: BitsAndBytes may not work on Windows. VLM will use FP16 instead.")
        print("  This is OK - 4-bit is optional.")
    
    # Numpy
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        all_passed = False
    
    # LeRobot
    try:
        import lerobot
        print(f"✓ LeRobot {lerobot.__version__}")
    except ImportError as e:
        print(f"✗ LeRobot: {e}")
        all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("✅ All required dependencies installed!")
        print("\nNext step: python sail/test_vip.py")
        return 0
    else:
        print("❌ Some dependencies missing. Install them and re-run.")
        print("\nInstall all at once:")
        print("pip install transformers>=4.36.0 timm>=0.9.12 Pillow>=10.0.0 accelerate>=0.25.0")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(test_imports())

