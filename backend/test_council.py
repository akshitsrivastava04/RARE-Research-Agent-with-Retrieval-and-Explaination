import requests
import json

# URL of the backend
API_URL = "http://localhost:8000"

def test_council_chat():
    print("Testing /council_chat endpoint...")
    
    # We need a token if auth is enabled.
    # Let's try to login first (assuming a test user exists or register one)
    # Actually, for this quick test, I'll assume I can just hit it if I have a valid token or make a temp user.
    # Let's register a temp user.
    
    username = "test_council_user"
    password = "password123"
    
    # Register
    try:
        requests.post(f"{API_URL}/register", json={"username": username, "password": password})
    except:
        pass # Maybe already exists
        
    # Login
    login_res = requests.post(f"{API_URL}/login", data={"username": username, "password": password})
    if login_res.status_code != 200:
        print("Login failed:", login_res.text)
        return
        
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test Council
    payload = {
        "question": "What are the benefits of functional programming?"
    }
    
    # Set a timeout because models take time
    try:
        res = requests.post(f"{API_URL}/council_chat", json=payload, headers=headers, timeout=120)
        
        if res.status_code == 200:
            data = res.json()
            print("\n--- COUNCIL CHAT SUCCESS ---")
            print("Session ID:", data.get("session_id"))
            print("Final Answer Length:", len(data.get("final_answer", "")))
            print("Individual Responses:", len(data.get("individual_responses", [])))
            for i, r in enumerate(data.get("individual_responses", [])):
                print(f"  Agent {i+1} ({r.get('agent_name')}): {len(r.get('response', ''))} chars")
        else:
            print("Council Chat Failed:", res.status_code, res.text)
            
    except Exception as e:
        print("Request Error:", e)

if __name__ == "__main__":
    test_council_chat()
