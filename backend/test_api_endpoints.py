#!/usr/bin/env python3
"""
test_api_endpoints.py

Test script to verify that the Flask API endpoints are working correctly,
including the new bear movement detection endpoints.
"""

import requests
from datetime import datetime, timezone

def test_health_endpoint(base_url):
    """Test the health check endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        print("✓ Health endpoint working")
        return True
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}")
        return False

def test_bears_endpoint(base_url):
    """Test the original bears endpoint."""
    print("Testing bears endpoint...")
    try:
        response = requests.get(f"{base_url}/api/bears")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Check structure of first bear record
        bear = data[0]
        required_fields = ["bearId", "pose", "timestamp"]
        for field in required_fields:
            assert field in bear, f"Missing field: {field}"
        
        print(f"✓ Bears endpoint working ({len(data)} records)")
        return True
    except Exception as e:
        print(f"❌ Bears endpoint failed: {e}")
        return False

def test_bear_movements_endpoint(base_url):
    """Test the bear movements endpoint."""
    print("Testing bear movements endpoint...")
    try:
        response = requests.get(f"{base_url}/api/bear-movements")
        
        if response.status_code == 503:
            print("⚠️  Bear movement detection system not available (expected if no data)")
            return True
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            print(f"✓ Bear movements endpoint working ({len(data)} records)")
            
            # If we have data, test the structure
            if data:
                movement = data[0]
                required_fields = ["id", "frame_number", "movement_state", "bear_id"]
                for field in required_fields:
                    assert field in movement, f"Missing field: {field}"
                print("✓ Movement record structure valid")
            
            return True
        else:
            print(f"❌ Bear movements endpoint returned {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Bear movements endpoint failed: {e}")
        return False

def test_bear_movements_summary_endpoint(base_url):
    """Test the bear movements summary endpoint."""
    print("Testing bear movements summary endpoint...")
    try:
        response = requests.get(f"{base_url}/api/bear-movements/summary")
        
        if response.status_code == 503:
            print("⚠️  Bear movement detection system not available (expected if no data)")
            return True
        
        if response.status_code == 200:
            data = response.json()
            required_fields = ["total_detections", "moving_count", "resting_count", "avg_confidence"]
            for field in required_fields:
                assert field in data, f"Missing field: {field}"
            
            print(f"✓ Bear movements summary working (total: {data['total_detections']})")
            return True
        else:
            print(f"❌ Bear movements summary returned {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Bear movements summary failed: {e}")
        return False

def test_openapi_documentation(base_url):
    """Test that OpenAPI documentation is accessible."""
    print("Testing OpenAPI documentation...")
    try:
        response = requests.get(f"{base_url}/docs/")
        assert response.status_code == 200
        print("✓ OpenAPI docs accessible")
        
        # Test OpenAPI JSON spec
        response = requests.get(f"{base_url}/docs/openapi.json")
        assert response.status_code == 200
        spec = response.json()
        
        # Check for our endpoints
        paths = spec.get("paths", {})
        expected_paths = ["/api/bears", "/api/bear-movements", "/api/bear-movements/summary"]
        
        for path in expected_paths:
            if path in paths:
                print(f"✓ Found {path} in OpenAPI spec")
            else:
                print(f"⚠️  Missing {path} in OpenAPI spec")
        
        return True
    except Exception as e:
        print(f"❌ OpenAPI documentation failed: {e}")
        return False

def run_comprehensive_test():
    """Run all API tests."""
    base_url = "http://localhost:5000"
    
    print("Bear Data Viewer API Test Suite")
    print("=" * 50)
    print(f"Testing API at: {base_url}")
    print(f"Test time: {datetime.now(timezone.utc).isoformat()}")
    print()
    
    # Test if server is reachable
    try:
        requests.get(base_url, timeout=5)
        print("✓ Server is reachable")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running?")
        print("Start server with: python run.py")
        return False
    except Exception as e:
        print(f"❌ Server connection error: {e}")
        return False
    
    print()
    
    # Run all tests
    tests = [
        test_health_endpoint,
        test_bears_endpoint,
        test_bear_movements_endpoint,
        test_bear_movements_summary_endpoint,
        test_openapi_documentation,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func(base_url)
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed or had warnings")
        return False

def main():
    """Main test runner."""
    success = run_comprehensive_test()
    
    if not success:
        print("\nTroubleshooting:")
        print("1. Ensure Flask server is running: python run.py")
        print("2. Check that all dependencies are installed")
        print("3. Verify .env configuration if testing movement endpoints")
        print("4. Run individual component tests: python test_bear_movement.py")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
