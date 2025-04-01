import sys
import os
import requests

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def test_api_endpoints(base_url='http://localhost:5000'):
    print("=== API Endpoint Tester ===")
    
    endpoints = [
        ('/api/species/', 'GET'),
        ('/api/images/', 'GET'),
        ('/api/annotations/', 'GET'),
        ('/', 'GET'),
        ('/health', 'GET')
    ]
    
    for endpoint, method in endpoints:
        try:
            full_url = f"{base_url}{endpoint}"
            
            if method == 'GET':
                response = requests.get(full_url)
            else:
                print(f"Unsupported method {method} for {endpoint}")
                continue
            
            print(f"\nEndpoint: {endpoint}")
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                print("Response JSON:")
                print(response.json())
            else:
                print(f"Error: {response.text}")
        
        except requests.exceptions.ConnectionError:
            print(f"\n❌ Cannot connect to {endpoint}. Is the server running?")
        except Exception as e:
            print(f"\n❌ Error testing {endpoint}: {e}")

if __name__ == '__main__':
    test_api_endpoints()