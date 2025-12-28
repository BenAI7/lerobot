"""
Phase 0 Pre-Flight Verification Script
Checks all requirements before starting SAIL baseline training
"""

import sys
import subprocess

def check_lerobot_installation():
    """Verify LeRobot is installed and get version."""
    try:
        import lerobot
        print(f"✓ LeRobot installed: version {lerobot.__version__}")
        return True
    except ImportError as e:
        print(f"✗ LeRobot not installed: {e}")
        return False

def check_torch_cuda():
    """Check PyTorch and CUDA availability."""
    try:
        import torch
        print(f"✓ PyTorch installed: version {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("✗ CUDA not available")
            return False
    except ImportError as e:
        print(f"✗ PyTorch not installed: {e}")
        return False

def check_cameras():
    """Find available OpenCV cameras."""
    try:
        import cv2
        print("\nChecking available cameras:")
        found_cameras = []
        for i in range(5):  # Check first 5 indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    print(f"✓ Camera {i}: {w}x{h}")
                    found_cameras.append(i)
                cap.release()
        
        if found_cameras:
            print(f"\nFound {len(found_cameras)} camera(s): {found_cameras}")
            return found_cameras
        else:
            print("✗ No cameras found")
            return []
    except Exception as e:
        print(f"✗ Error checking cameras: {e}")
        return []

def check_serial_ports():
    """Find available COM ports for robot connection."""
    try:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        print("\nAvailable COM ports:")
        if ports:
            for port in ports:
                print(f"  {port.device}: {port.description}")
            return [p.device for p in ports]
        else:
            print("✗ No COM ports found")
            return []
    except Exception as e:
        print(f"✗ Error checking ports: {e}")
        return []

def check_huggingface_cli():
    """Check if logged into Hugging Face."""
    try:
        result = subprocess.run(
            ["huggingface-cli", "whoami"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            username = result.stdout.strip()
            print(f"\n✓ Logged into Hugging Face as: {username}")
            return username
        else:
            print("\n✗ Not logged into Hugging Face")
            print("  Run: huggingface-cli login")
            return None
    except Exception as e:
        print(f"\n✗ Hugging Face CLI check failed: {e}")
        return None

def check_feetech_motors():
    """Check if Feetech motor SDK is available."""
    try:
        from lerobot.motors.feetech import FeetechMotorsBus
        print("✓ Feetech motor SDK available")
        return True
    except ImportError:
        print("✗ Feetech motor SDK not installed")
        print("  Install with: pip install lerobot[feetech]")
        return False

def main():
    print("=" * 60)
    print("SAIL Phase 0 Pre-Flight Verification")
    print("=" * 60)
    
    checks = []
    
    # Core requirements
    checks.append(("LeRobot", check_lerobot_installation()))
    checks.append(("PyTorch + CUDA", check_torch_cuda()))
    checks.append(("Feetech Motors", check_feetech_motors()))
    
    # Hardware
    cameras = check_cameras()
    checks.append(("Cameras", len(cameras) > 0))
    
    ports = check_serial_ports()
    checks.append(("Serial Ports", len(ports) > 0))
    
    # Hugging Face
    hf_user = check_huggingface_cli()
    checks.append(("Hugging Face", hf_user is not None))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = all(check[1] for check in checks)
    
    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All checks passed! Ready to proceed with Phase 0.")
        
        # Generate environment setup info
        print("\n" + "=" * 60)
        print("CONFIGURATION FOR PHASE 0")
        print("=" * 60)
        if hf_user:
            print(f"Hugging Face User: {hf_user}")
        if cameras:
            print(f"Suggested camera: {cameras[0]}")
        if ports:
            print(f"Available ports: {', '.join(ports)}")
            if len(ports) >= 2:
                print(f"  Leader arm (suggestion): {ports[0]}")
                print(f"  Follower arm (suggestion): {ports[1]}")
        
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix issues before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

