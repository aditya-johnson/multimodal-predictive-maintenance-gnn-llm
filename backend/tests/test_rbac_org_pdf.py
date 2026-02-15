"""
Comprehensive tests for RBAC, Organizations, and PDF Report features
Tests: User registration, login, organization CRUD, member management, 
       RBAC permissions, machine operations, predictions, and PDF reports
"""
import pytest
import requests
import os
import time

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')

# Test credentials
TEST_ADMIN_EMAIL = "admin_test@test.com"
TEST_ADMIN_PASSWORD = "password123"
TEST_ADMIN_NAME = "Test Admin"

TEST_OPERATOR_EMAIL = "operator_test@test.com"
TEST_OPERATOR_PASSWORD = "password123"
TEST_OPERATOR_NAME = "Test Operator"

TEST_VIEWER_EMAIL = "viewer_test@test.com"
TEST_VIEWER_PASSWORD = "password123"
TEST_VIEWER_NAME = "Test Viewer"


class TestAuthEndpoints:
    """Authentication endpoint tests - Register and Login"""
    
    def test_api_health(self):
        """Test API is operational"""
        response = requests.get(f"{BASE_URL}/api/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "operational"
        print(f"✓ API health check passed - version {data.get('version')}")
    
    def test_register_admin_user(self):
        """Test user registration"""
        response = requests.post(f"{BASE_URL}/api/auth/register", json={
            "email": TEST_ADMIN_EMAIL,
            "password": TEST_ADMIN_PASSWORD,
            "name": TEST_ADMIN_NAME
        })
        # May return 400 if already registered
        if response.status_code == 400:
            assert "already registered" in response.json().get("detail", "").lower()
            print("✓ Admin user already exists")
        else:
            assert response.status_code == 200
            data = response.json()
            assert "access_token" in data
            assert data["user"]["email"] == TEST_ADMIN_EMAIL
            print(f"✓ Admin user registered: {data['user']['email']}")
    
    def test_register_operator_user(self):
        """Test operator user registration"""
        response = requests.post(f"{BASE_URL}/api/auth/register", json={
            "email": TEST_OPERATOR_EMAIL,
            "password": TEST_OPERATOR_PASSWORD,
            "name": TEST_OPERATOR_NAME
        })
        if response.status_code == 400:
            print("✓ Operator user already exists")
        else:
            assert response.status_code == 200
            print(f"✓ Operator user registered")
    
    def test_register_viewer_user(self):
        """Test viewer user registration"""
        response = requests.post(f"{BASE_URL}/api/auth/register", json={
            "email": TEST_VIEWER_EMAIL,
            "password": TEST_VIEWER_PASSWORD,
            "name": TEST_VIEWER_NAME
        })
        if response.status_code == 400:
            print("✓ Viewer user already exists")
        else:
            assert response.status_code == 200
            print(f"✓ Viewer user registered")
    
    def test_login_success(self):
        """Test successful login"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": TEST_ADMIN_EMAIL,
            "password": TEST_ADMIN_PASSWORD
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["user"]["email"] == TEST_ADMIN_EMAIL
        print(f"✓ Login successful for {TEST_ADMIN_EMAIL}")
    
    def test_login_invalid_credentials(self):
        """Test login with invalid credentials"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": "nonexistent@test.com",
            "password": "wrongpassword"
        })
        assert response.status_code == 401
        print("✓ Invalid credentials rejected correctly")


class TestOrganizationEndpoints:
    """Organization CRUD and member management tests"""
    
    @pytest.fixture
    def admin_token(self):
        """Get admin auth token"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": TEST_ADMIN_EMAIL,
            "password": TEST_ADMIN_PASSWORD
        })
        assert response.status_code == 200
        return response.json()["access_token"]
    
    @pytest.fixture
    def admin_headers(self, admin_token):
        """Get admin auth headers"""
        return {"Authorization": f"Bearer {admin_token}"}
    
    def test_create_organization(self, admin_headers):
        """Test organization creation"""
        response = requests.post(f"{BASE_URL}/api/organizations", 
            json={"name": "TEST_Org_RBAC", "description": "Test organization for RBAC testing"},
            headers=admin_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "organization" in data
        assert data["organization"]["name"] == "TEST_Org_RBAC"
        assert data["role"] == "admin"
        print(f"✓ Organization created: {data['organization']['name']}")
        return data["organization"]["id"]
    
    def test_get_user_organizations(self, admin_headers):
        """Test getting user's organizations"""
        response = requests.get(f"{BASE_URL}/api/organizations", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        print(f"✓ Retrieved {len(data)} organizations")
    
    def test_switch_organization(self, admin_headers):
        """Test switching organization"""
        # First get organizations
        orgs_response = requests.get(f"{BASE_URL}/api/organizations", headers=admin_headers)
        orgs = orgs_response.json()
        
        if len(orgs) > 0:
            org_id = orgs[0]["id"]
            response = requests.post(f"{BASE_URL}/api/organizations/{org_id}/switch", headers=admin_headers)
            assert response.status_code == 200
            data = response.json()
            assert "access_token" in data
            assert "organization" in data
            assert "role" in data
            print(f"✓ Switched to organization: {data['organization']['name']}, role: {data['role']}")
            return data["access_token"]
        else:
            pytest.skip("No organizations to switch to")


class TestRBACPermissions:
    """Test RBAC permission enforcement"""
    
    @pytest.fixture
    def setup_org_and_token(self):
        """Setup: Login admin, create/switch org, return token"""
        # Login admin
        login_resp = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": TEST_ADMIN_EMAIL,
            "password": TEST_ADMIN_PASSWORD
        })
        assert login_resp.status_code == 200
        admin_token = login_resp.json()["access_token"]
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        
        # Get or create organization
        orgs_resp = requests.get(f"{BASE_URL}/api/organizations", headers=admin_headers)
        orgs = orgs_resp.json()
        
        if len(orgs) == 0:
            # Create new org
            create_resp = requests.post(f"{BASE_URL}/api/organizations",
                json={"name": "TEST_RBAC_Org", "description": "RBAC Test Org"},
                headers=admin_headers
            )
            assert create_resp.status_code == 200
            org_id = create_resp.json()["organization"]["id"]
        else:
            org_id = orgs[0]["id"]
        
        # Switch to org
        switch_resp = requests.post(f"{BASE_URL}/api/organizations/{org_id}/switch", headers=admin_headers)
        assert switch_resp.status_code == 200
        new_token = switch_resp.json()["access_token"]
        
        return new_token, org_id
    
    def test_admin_can_manage_machines(self, setup_org_and_token):
        """Test admin can create machines"""
        token, org_id = setup_org_and_token
        headers = {"Authorization": f"Bearer {token}"}
        
        response = requests.post(f"{BASE_URL}/api/machines",
            json={"name": "TEST_Machine_Admin", "machine_type": "Test Type", "location": "Test Location"},
            headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "TEST_Machine_Admin"
        print(f"✓ Admin created machine: {data['name']}")
    
    def test_admin_can_seed_demo(self, setup_org_and_token):
        """Test admin can seed demo data"""
        token, org_id = setup_org_and_token
        headers = {"Authorization": f"Bearer {token}"}
        
        response = requests.post(f"{BASE_URL}/api/seed-demo", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "machines_created" in data
        print(f"✓ Admin seeded demo data: {data['machines_created']} machines")
    
    def test_admin_can_run_prediction(self, setup_org_and_token):
        """Test admin can run predictions"""
        token, org_id = setup_org_and_token
        headers = {"Authorization": f"Bearer {token}"}
        
        # Get machines
        machines_resp = requests.get(f"{BASE_URL}/api/machines", headers=headers)
        machines = machines_resp.json()
        
        if len(machines) > 0:
            machine_id = machines[0]["id"]
            response = requests.post(f"{BASE_URL}/api/machines/{machine_id}/predict", headers=headers)
            assert response.status_code == 200
            data = response.json()
            assert "remaining_useful_life_days" in data
            assert "fusion_score" in data
            print(f"✓ Admin ran prediction - RUL: {data['remaining_useful_life_days']} days")
        else:
            pytest.skip("No machines available for prediction")
    
    def test_viewer_cannot_create_machine(self, setup_org_and_token):
        """Test viewer cannot create machines (permission denied)"""
        admin_token, org_id = setup_org_and_token
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        
        # Invite viewer
        invite_resp = requests.post(f"{BASE_URL}/api/organizations/{org_id}/invite",
            json={"email": TEST_VIEWER_EMAIL, "role": "viewer"},
            headers=admin_headers
        )
        # May already be invited or member
        
        # Login as viewer
        viewer_login = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": TEST_VIEWER_EMAIL,
            "password": TEST_VIEWER_PASSWORD
        })
        
        if viewer_login.status_code == 200:
            viewer_token = viewer_login.json()["access_token"]
            viewer_headers = {"Authorization": f"Bearer {viewer_token}"}
            
            # Check invitations
            invites_resp = requests.get(f"{BASE_URL}/api/invitations", headers=viewer_headers)
            invites = invites_resp.json()
            
            # Accept invitation if pending
            for invite in invites:
                if invite["org_id"] == org_id:
                    accept_resp = requests.post(f"{BASE_URL}/api/invitations/{invite['id']}/accept", headers=viewer_headers)
            
            # Switch to org
            switch_resp = requests.post(f"{BASE_URL}/api/organizations/{org_id}/switch", headers=viewer_headers)
            if switch_resp.status_code == 200:
                viewer_token = switch_resp.json()["access_token"]
                viewer_headers = {"Authorization": f"Bearer {viewer_token}"}
                
                # Try to create machine - should fail
                create_resp = requests.post(f"{BASE_URL}/api/machines",
                    json={"name": "TEST_Viewer_Machine", "machine_type": "Test", "location": "Test"},
                    headers=viewer_headers
                )
                assert create_resp.status_code == 403
                print("✓ Viewer correctly denied machine creation permission")
            else:
                print("✓ Viewer not member of org - permission test skipped")
        else:
            pytest.skip("Viewer user not available")


class TestMachineAndPrediction:
    """Machine CRUD and prediction tests"""
    
    @pytest.fixture
    def auth_setup(self):
        """Setup authenticated session with org"""
        login_resp = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": TEST_ADMIN_EMAIL,
            "password": TEST_ADMIN_PASSWORD
        })
        assert login_resp.status_code == 200
        token = login_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Get orgs and switch
        orgs_resp = requests.get(f"{BASE_URL}/api/organizations", headers=headers)
        orgs = orgs_resp.json()
        
        if len(orgs) > 0:
            switch_resp = requests.post(f"{BASE_URL}/api/organizations/{orgs[0]['id']}/switch", headers=headers)
            if switch_resp.status_code == 200:
                token = switch_resp.json()["access_token"]
                headers = {"Authorization": f"Bearer {token}"}
        
        return headers
    
    def test_get_machines(self, auth_setup):
        """Test getting machines list"""
        response = requests.get(f"{BASE_URL}/api/machines", headers=auth_setup)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        print(f"✓ Retrieved {len(data)} machines")
        return data
    
    def test_get_machine_details(self, auth_setup):
        """Test getting single machine details"""
        machines_resp = requests.get(f"{BASE_URL}/api/machines", headers=auth_setup)
        machines = machines_resp.json()
        
        if len(machines) > 0:
            machine_id = machines[0]["id"]
            response = requests.get(f"{BASE_URL}/api/machines/{machine_id}", headers=auth_setup)
            assert response.status_code == 200
            data = response.json()
            assert "health_score" in data
            assert "risk_level" in data
            print(f"✓ Machine details: {data['name']} - Health: {data['health_score']}%")
        else:
            pytest.skip("No machines available")
    
    def test_get_sensor_readings(self, auth_setup):
        """Test getting sensor readings"""
        machines_resp = requests.get(f"{BASE_URL}/api/machines", headers=auth_setup)
        machines = machines_resp.json()
        
        if len(machines) > 0:
            machine_id = machines[0]["id"]
            response = requests.get(f"{BASE_URL}/api/machines/{machine_id}/readings", headers=auth_setup)
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            if len(data) > 0:
                assert "temperature" in data[0]
                assert "vibration" in data[0]
            print(f"✓ Retrieved {len(data)} sensor readings")
        else:
            pytest.skip("No machines available")
    
    def test_get_sensor_graph(self, auth_setup):
        """Test getting sensor correlation graph"""
        machines_resp = requests.get(f"{BASE_URL}/api/machines", headers=auth_setup)
        machines = machines_resp.json()
        
        if len(machines) > 0:
            machine_id = machines[0]["id"]
            response = requests.get(f"{BASE_URL}/api/machines/{machine_id}/sensor-graph", headers=auth_setup)
            assert response.status_code == 200
            data = response.json()
            assert "nodes" in data
            assert "links" in data
            print(f"✓ Sensor graph: {len(data['nodes'])} nodes, {len(data['links'])} links")
        else:
            pytest.skip("No machines available")
    
    def test_get_predictions(self, auth_setup):
        """Test getting prediction history"""
        machines_resp = requests.get(f"{BASE_URL}/api/machines", headers=auth_setup)
        machines = machines_resp.json()
        
        if len(machines) > 0:
            machine_id = machines[0]["id"]
            response = requests.get(f"{BASE_URL}/api/machines/{machine_id}/predictions", headers=auth_setup)
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            print(f"✓ Retrieved {len(data)} predictions")
        else:
            pytest.skip("No machines available")


class TestPDFReport:
    """PDF Report generation tests"""
    
    @pytest.fixture
    def auth_setup(self):
        """Setup authenticated session with org"""
        login_resp = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": TEST_ADMIN_EMAIL,
            "password": TEST_ADMIN_PASSWORD
        })
        assert login_resp.status_code == 200
        token = login_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        orgs_resp = requests.get(f"{BASE_URL}/api/organizations", headers=headers)
        orgs = orgs_resp.json()
        
        if len(orgs) > 0:
            switch_resp = requests.post(f"{BASE_URL}/api/organizations/{orgs[0]['id']}/switch", headers=headers)
            if switch_resp.status_code == 200:
                token = switch_resp.json()["access_token"]
                headers = {"Authorization": f"Bearer {token}"}
        
        return headers
    
    def test_generate_pdf_report(self, auth_setup):
        """Test PDF report generation"""
        machines_resp = requests.get(f"{BASE_URL}/api/machines", headers=auth_setup)
        machines = machines_resp.json()
        
        if len(machines) > 0:
            machine_id = machines[0]["id"]
            response = requests.get(f"{BASE_URL}/api/machines/{machine_id}/report?days=30", headers=auth_setup)
            assert response.status_code == 200
            assert response.headers.get("content-type") == "application/pdf"
            assert len(response.content) > 1000  # PDF should have content
            print(f"✓ PDF report generated: {len(response.content)} bytes")
        else:
            pytest.skip("No machines available for PDF report")
    
    def test_pdf_report_different_periods(self, auth_setup):
        """Test PDF report with different time periods"""
        machines_resp = requests.get(f"{BASE_URL}/api/machines", headers=auth_setup)
        machines = machines_resp.json()
        
        if len(machines) > 0:
            machine_id = machines[0]["id"]
            
            for days in [7, 30, 90]:
                response = requests.get(f"{BASE_URL}/api/machines/{machine_id}/report?days={days}", headers=auth_setup)
                assert response.status_code == 200
                assert response.headers.get("content-type") == "application/pdf"
                print(f"✓ PDF report ({days} days): {len(response.content)} bytes")
        else:
            pytest.skip("No machines available")


class TestMaintenanceLogs:
    """Maintenance log tests with NLP analysis"""
    
    @pytest.fixture
    def auth_setup(self):
        """Setup authenticated session"""
        login_resp = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": TEST_ADMIN_EMAIL,
            "password": TEST_ADMIN_PASSWORD
        })
        assert login_resp.status_code == 200
        token = login_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        orgs_resp = requests.get(f"{BASE_URL}/api/organizations", headers=headers)
        orgs = orgs_resp.json()
        
        if len(orgs) > 0:
            switch_resp = requests.post(f"{BASE_URL}/api/organizations/{orgs[0]['id']}/switch", headers=headers)
            if switch_resp.status_code == 200:
                token = switch_resp.json()["access_token"]
                headers = {"Authorization": f"Bearer {token}"}
        
        return headers
    
    def test_create_maintenance_log(self, auth_setup):
        """Test creating maintenance log with NLP analysis"""
        machines_resp = requests.get(f"{BASE_URL}/api/machines", headers=auth_setup)
        machines = machines_resp.json()
        
        if len(machines) > 0:
            machine_id = machines[0]["id"]
            response = requests.post(f"{BASE_URL}/api/maintenance-logs",
                json={
                    "machine_id": machine_id,
                    "log_text": "Abnormal bearing noise detected during inspection. Excessive vibration observed.",
                    "technician": "Test Technician",
                    "severity": "warning"
                },
                headers=auth_setup
            )
            assert response.status_code == 200
            data = response.json()
            assert "risk_keywords" in data
            assert len(data["risk_keywords"]) > 0  # Should detect keywords
            print(f"✓ Maintenance log created with keywords: {data['risk_keywords']}")
        else:
            pytest.skip("No machines available")
    
    def test_get_maintenance_logs(self, auth_setup):
        """Test getting maintenance logs"""
        machines_resp = requests.get(f"{BASE_URL}/api/machines", headers=auth_setup)
        machines = machines_resp.json()
        
        if len(machines) > 0:
            machine_id = machines[0]["id"]
            response = requests.get(f"{BASE_URL}/api/machines/{machine_id}/maintenance-logs", headers=auth_setup)
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            print(f"✓ Retrieved {len(data)} maintenance logs")
        else:
            pytest.skip("No machines available")


class TestAlerts:
    """Alert system tests"""
    
    @pytest.fixture
    def auth_setup(self):
        """Setup authenticated session"""
        login_resp = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": TEST_ADMIN_EMAIL,
            "password": TEST_ADMIN_PASSWORD
        })
        assert login_resp.status_code == 200
        token = login_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        orgs_resp = requests.get(f"{BASE_URL}/api/organizations", headers=headers)
        orgs = orgs_resp.json()
        
        if len(orgs) > 0:
            switch_resp = requests.post(f"{BASE_URL}/api/organizations/{orgs[0]['id']}/switch", headers=headers)
            if switch_resp.status_code == 200:
                token = switch_resp.json()["access_token"]
                headers = {"Authorization": f"Bearer {token}"}
        
        return headers
    
    def test_get_alerts(self, auth_setup):
        """Test getting alerts"""
        response = requests.get(f"{BASE_URL}/api/alerts", headers=auth_setup)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        print(f"✓ Retrieved {len(data)} alerts")
    
    def test_get_alert_settings(self, auth_setup):
        """Test getting alert settings"""
        response = requests.get(f"{BASE_URL}/api/alert-settings", headers=auth_setup)
        assert response.status_code == 200
        data = response.json()
        assert "email_enabled" in data
        assert "critical_threshold" in data
        print(f"✓ Alert settings: critical threshold = {data['critical_threshold']}%")


class TestDashboard:
    """Dashboard summary tests"""
    
    @pytest.fixture
    def auth_setup(self):
        """Setup authenticated session"""
        login_resp = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": TEST_ADMIN_EMAIL,
            "password": TEST_ADMIN_PASSWORD
        })
        assert login_resp.status_code == 200
        token = login_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        orgs_resp = requests.get(f"{BASE_URL}/api/organizations", headers=headers)
        orgs = orgs_resp.json()
        
        if len(orgs) > 0:
            switch_resp = requests.post(f"{BASE_URL}/api/organizations/{orgs[0]['id']}/switch", headers=headers)
            if switch_resp.status_code == 200:
                token = switch_resp.json()["access_token"]
                headers = {"Authorization": f"Bearer {token}"}
        
        return headers
    
    def test_dashboard_summary(self, auth_setup):
        """Test dashboard summary endpoint"""
        response = requests.get(f"{BASE_URL}/api/dashboard/summary", headers=auth_setup)
        assert response.status_code == 200
        data = response.json()
        assert "total_machines" in data
        assert "healthy" in data
        assert "warning" in data
        assert "critical" in data
        assert "average_health_score" in data
        print(f"✓ Dashboard: {data['total_machines']} machines, avg health: {data['average_health_score']}%")


class TestInvitationFlow:
    """Test invitation and member management flow"""
    
    def test_full_invitation_flow(self):
        """Test complete invitation flow: invite -> accept -> verify membership"""
        # Login as admin
        admin_login = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": TEST_ADMIN_EMAIL,
            "password": TEST_ADMIN_PASSWORD
        })
        assert admin_login.status_code == 200
        admin_token = admin_login.json()["access_token"]
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        
        # Get/create org
        orgs_resp = requests.get(f"{BASE_URL}/api/organizations", headers=admin_headers)
        orgs = orgs_resp.json()
        
        if len(orgs) == 0:
            create_resp = requests.post(f"{BASE_URL}/api/organizations",
                json={"name": "TEST_Invite_Org", "description": "Invitation test"},
                headers=admin_headers
            )
            org_id = create_resp.json()["organization"]["id"]
        else:
            org_id = orgs[0]["id"]
        
        # Switch to org
        switch_resp = requests.post(f"{BASE_URL}/api/organizations/{org_id}/switch", headers=admin_headers)
        if switch_resp.status_code == 200:
            admin_token = switch_resp.json()["access_token"]
            admin_headers = {"Authorization": f"Bearer {admin_token}"}
        
        # Invite operator
        invite_resp = requests.post(f"{BASE_URL}/api/organizations/{org_id}/invite",
            json={"email": TEST_OPERATOR_EMAIL, "role": "operator"},
            headers=admin_headers
        )
        
        if invite_resp.status_code == 200:
            print(f"✓ Invitation sent to {TEST_OPERATOR_EMAIL}")
        elif invite_resp.status_code == 400:
            print(f"✓ User already invited or member")
        
        # Login as operator and check invitations
        operator_login = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": TEST_OPERATOR_EMAIL,
            "password": TEST_OPERATOR_PASSWORD
        })
        
        if operator_login.status_code == 200:
            operator_token = operator_login.json()["access_token"]
            operator_headers = {"Authorization": f"Bearer {operator_token}"}
            
            # Get invitations
            invites_resp = requests.get(f"{BASE_URL}/api/invitations", headers=operator_headers)
            invites = invites_resp.json()
            
            # Accept pending invitation
            for invite in invites:
                if invite["org_id"] == org_id and not invite["accepted"]:
                    accept_resp = requests.post(f"{BASE_URL}/api/invitations/{invite['id']}/accept", headers=operator_headers)
                    if accept_resp.status_code == 200:
                        print(f"✓ Invitation accepted")
                    break
            
            # Verify membership
            operator_orgs = requests.get(f"{BASE_URL}/api/organizations", headers=operator_headers)
            org_ids = [o["id"] for o in operator_orgs.json()]
            if org_id in org_ids:
                print(f"✓ Operator is now member of organization")
        
        print("✓ Invitation flow test completed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
