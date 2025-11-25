import cv2
import time

def quick_rtsp_test():
    """
    Super simple RTSP connection test.
    Just modify the RTSP URL below and run.
    """
    
    # 🔧 MODIFY THIS URL WITH YOUR CAMERA DETAILS
    rtsp_url = "rtsp://admin:123456@192.168.1.13:554/stream1"
    
    print("🔗 Testing RTSP connection...")
    print(f"📡 URL: {rtsp_url}")
    print("-" * 40)
    
    try:
        # Create video capture
        cap = cv2.VideoCapture(rtsp_url)
        
        if not cap.isOpened():
            print("❌ FAILED: Could not connect to RTSP stream")
            print("💡 Check:")
            print("   - Camera IP address")
            print("   - Username/password") 
            print("   - RTSP enabled in camera settings")
            return False
        
        print("✅ SUCCESS: Connected to RTSP stream!")
        
        # Get stream info
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"📊 Stream Info:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        
        # Test reading frames
        print("\n🎬 Testing frame capture...")
        success_count = 0
        
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                success_count += 1
                print(f"✅ Frame {i+1}: OK")
            else:
                print(f"❌ Frame {i+1}: Failed")
            time.sleep(0.5)
        
        cap.release()
        
        if success_count >= 3:
            print(f"\n🎉 RTSP TEST PASSED! ({success_count}/5 frames)")
            print("🚀 Your camera is ready for head detection!")
            return True
        else:
            print(f"\n⚠️ RTSP TEST PARTIAL ({success_count}/5 frames)")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_common_urls():
    """Test common RTSP URL patterns for TP-Link cameras."""
    
    # 🔧 MODIFY THESE VALUES
    camera_ip = "192.168.1.100"  # Change to your camera's IP
    username = "admin"           # Change if different
    password = "123456"           # Change if different
    
    urls_to_test = [
        f"rtsp://{username}:{password}@{camera_ip}:554/stream1",
        f"rtsp://{username}:{password}@{camera_ip}:554/stream2", 
        f"rtsp://{username}:{password}@{camera_ip}:554/live/ch00_0",
        f"rtsp://{username}:{password}@{camera_ip}:554/live/ch00_1",
        f"rtsp://{username}:{password}@{camera_ip}:554/",
    ]
    
    print("🔍 Testing common TP-Link RTSP URLs...")
    print(f"📡 Camera IP: {camera_ip}")
    print(f"👤 Credentials: {username}:{password}")
    print("=" * 50)
    
    working_urls = []
    
    for i, url in enumerate(urls_to_test, 1):
        print(f"\n🧪 Test {i}: {url}")
        
        try:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print("✅ SUCCESS!")
                    working_urls.append(url)
                else:
                    print("⚠️ Connected but no frames")
            else:
                print("❌ Failed to connect")
            cap.release()
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(1)
    
    print("\n" + "=" * 50)
    if working_urls:
        print(f"🎉 Found {len(working_urls)} working URL(s):")
        for url in working_urls:
            print(f"   📡 {url}")
        print(f"\n🚀 Use this URL: {working_urls[0]}")
        return working_urls[0]
    else:
        print("❌ No working URLs found")
        return None

if __name__ == "__main__":
    print("🎥 Simple RTSP Tester")
    print("Choose an option:")
    print("1. Test specific URL (modify rtsp_url in code)")
    print("2. Test common URLs (modify camera_ip/credentials in code)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        quick_rtsp_test()
    elif choice == "2":
        test_common_urls()
    else:
        print("Invalid choice. Running quick test...")
        quick_rtsp_test()