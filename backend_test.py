import requests
import sys
import json
from datetime import datetime

class PredictiveMaintenanceAPITester:
    def __init__(self, base_url="https://predictiverails.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.machine_ids = []

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=30):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if not endpoint.startswith('http') else endpoint
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=timeout)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    if isinstance(response_data, dict) and len(str(response_data)) < 500:
                        print(f"   Response: {response_data}")
                    elif isinstance(response_data, list):
                        print(f"   Response: List with {len(response_data)} items")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text[:200]}")
                return False, {}

        except requests.exceptions.Timeout:
            print(f"❌ Failed - Request timeout after {timeout}s")
            return False, {}
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_api_status(self):
        """Test API operational status"""
        return self.run_test("API Status", "GET", "", 200)

    def test_seed_demo_data(self):
        """Test seeding demo data"""
        success, response = self.run_test("Seed Demo Data", "POST", "seed-demo", 200, timeout=60)
        if success and 'machine_ids' in response:
            self.machine_ids = response['machine_ids']
            print(f"   Created machines: {len(self.machine_ids)}")
        return success

    def test_get_machines(self):
        """Test getting machine list"""
        success, response = self.run_test("Get Machines", "GET", "machines", 200)
        if success and isinstance(response, list):
            print(f"   Found {len(response)} machines")
            if response:
                machine = response[0]
                print(f"   Sample machine: {machine.get('name')} - Health: {machine.get('health_score')}%")
        return success

    def test_get_machine_readings(self, machine_id):
        """Test getting sensor readings for a machine"""
        success, response = self.run_test(
            f"Get Readings for {machine_id[:8]}...", 
            "GET", 
            f"machines/{machine_id}/readings?limit=10", 
            200
        )
        if success and isinstance(response, list):
            print(f"   Found {len(response)} readings")
            if response:
                reading = response[0]
                print(f"   Sample reading: Temp={reading.get('temperature')}°C, Vib={reading.get('vibration')}mm/s")
        return success

    def test_get_sensor_graph(self, machine_id):
        """Test getting sensor correlation graph"""
        success, response = self.run_test(
            f"Get Sensor Graph for {machine_id[:8]}...", 
            "GET", 
            f"machines/{machine_id}/sensor-graph", 
            200
        )
        if success and isinstance(response, dict):
            nodes = response.get('nodes', [])
            links = response.get('links', [])
            print(f"   Graph: {len(nodes)} nodes, {len(links)} links")
        return success

    def test_run_prediction(self, machine_id):
        """Test running multimodal prediction"""
        success, response = self.run_test(
            f"Run Prediction for {machine_id[:8]}...", 
            "POST", 
            f"machines/{machine_id}/predict", 
            200,
            timeout=45
        )
        if success and isinstance(response, dict):
            rul = response.get('remaining_useful_life_days')
            confidence = response.get('confidence_score')
            failure_type = response.get('failure_type')
            print(f"   Prediction: RUL={rul} days, Confidence={confidence}, Type={failure_type}")
        return success

    def test_create_maintenance_log(self, machine_id):
        """Test creating maintenance log with NLP analysis"""
        log_data = {
            "machine_id": machine_id,
            "log_text": "Abnormal bearing noise detected during routine inspection. Excessive vibration observed near motor housing.",
            "technician": "Test Engineer",
            "severity": "warning"
        }
        success, response = self.run_test(
            f"Create Maintenance Log for {machine_id[:8]}...", 
            "POST", 
            "maintenance-logs", 
            200,
            data=log_data
        )
        if success and isinstance(response, dict):
            keywords = response.get('risk_keywords', [])
            similarity = response.get('embedding_similarity')
            print(f"   NLP Analysis: Keywords={keywords}, Risk Score={similarity}")
        return success

    def test_get_maintenance_logs(self, machine_id):
        """Test getting maintenance logs"""
        success, response = self.run_test(
            f"Get Maintenance Logs for {machine_id[:8]}...", 
            "GET", 
            f"machines/{machine_id}/maintenance-logs", 
            200
        )
        if success and isinstance(response, list):
            print(f"   Found {len(response)} maintenance logs")
        return success

    def test_dashboard_summary(self):
        """Test dashboard summary endpoint"""
        success, response = self.run_test("Dashboard Summary", "GET", "dashboard/summary", 200)
        if success and isinstance(response, dict):
            total = response.get('total_machines')
            healthy = response.get('healthy')
            warning = response.get('warning')
            critical = response.get('critical')
            print(f"   Summary: {total} total, {healthy} healthy, {warning} warning, {critical} critical")
        return success

def main():
    print("🚀 Starting Multimodal Predictive Maintenance API Tests")
    print("=" * 60)
    
    tester = PredictiveMaintenanceAPITester()
    
    # Test basic API status
    if not tester.test_api_status():
        print("❌ API is not accessible, stopping tests")
        return 1

    # Seed demo data
    if not tester.test_seed_demo_data():
        print("❌ Failed to seed demo data, stopping tests")
        return 1

    # Test machine endpoints
    if not tester.test_get_machines():
        print("❌ Failed to get machines")
        return 1

    # Test dashboard summary
    tester.test_dashboard_summary()

    # Test machine-specific endpoints if we have machine IDs
    if tester.machine_ids:
        test_machine_id = tester.machine_ids[0]
        print(f"\n📊 Testing machine-specific endpoints with ID: {test_machine_id}")
        
        # Test sensor readings
        tester.test_get_machine_readings(test_machine_id)
        
        # Test sensor graph
        tester.test_get_sensor_graph(test_machine_id)
        
        # Test prediction (this might take longer)
        tester.test_run_prediction(test_machine_id)
        
        # Test maintenance logs
        tester.test_create_maintenance_log(test_machine_id)
        tester.test_get_maintenance_logs(test_machine_id)
    else:
        print("⚠️  No machine IDs available for machine-specific tests")

    # Print final results
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {tester.tests_passed}/{tester.tests_run} tests passed")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"❌ {tester.tests_run - tester.tests_passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())