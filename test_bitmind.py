"""
Quick test script to verify BitMind API integration
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.bitmind_detector import BitMindDetector

# Test API key
API_KEY = "bitmind-ca6693f0-38f1-11f1-ac47-015f92f4a374:529e01d0"

print("="*70)
print("BitMind API Integration Test")
print("="*70)

# Initialize detector
print(f"\n✓ Initializing BitMind detector with API key...")
detector = BitMindDetector(API_KEY)

# Test health check
print(f"✓ Testing BitMind API health...")
is_healthy = detector.is_healthy()
print(f"  BitMind API Status: {'🟢 ONLINE' if is_healthy else '🔴 OFFLINE'}")

if is_healthy:
    print("\n" + "="*70)
    print("✓ BitMind API Integration is Working!")
    print("="*70)
    print("\nYou can now:")
    print("  1. Restart the Flask app: python app.py")
    print("  2. Upload videos at http://localhost:5000")
    print("  3. Videos will be analyzed using BitMind API")
else:
    print("\n⚠️  BitMind API is unreachable")
    print("Please check:")
    print("  - Internet connection")
    print("  - API key is correct")
    print("  - BitMind service status")
