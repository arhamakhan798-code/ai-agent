"""
One-time Gmail OAuth2 Authorization Script
Run this script to generate gmail_token.json
"""
import os
import sys
import json

# Ensure required libraries
import subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "google-auth-oauthlib", "google-api-python-client", "--quiet"], check=True)

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIENT_SECRET_FILE = os.path.join(BASE_DIR, "client_secret.json")
TOKEN_OUTPUT_FILE  = os.path.join(BASE_DIR, "gmail_token.json")

SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
]

print("=" * 60)
print("  TechMart Gmail - One-Time Authorization")
print("=" * 60)

if not os.path.exists(CLIENT_SECRET_FILE):
    print(f"\n[ERROR] Missing credentials file: {CLIENT_SECRET_FILE}")
    sys.exit(1)

with open(CLIENT_SECRET_FILE, "r") as f:
    raw_config = json.load(f)

# Support both "web" and "installed" (Desktop app) configs
if "web" in raw_config:
    web = raw_config["web"]
    app_config = {
        "installed": {
            "client_id": web["client_id"],
            "client_secret": web["client_secret"],
            "auth_uri": web.get("auth_uri", "https://accounts.google.com/o/oauth2/auth"),
            "token_uri": web.get("token_uri", "https://oauth2.googleapis.com/token"),
            "redirect_uris": ["http://localhost:8080/"]
        }
    }
else:
    app_config = raw_config

print("\nStarting local authentication server on port 8080...")
print("If a browser does not open automatically, copy and paste the link below.\n")

flow = InstalledAppFlow.from_client_config(app_config, SCOPES)

try:
    creds = flow.run_local_server(
        host="localhost",
        port=8080,
        authorization_prompt_message="Open this URL in your browser to authorize:\n{url}\n",
        success_message="Authentication complete! You can close this window now.",
        open_browser=True,
        prompt="consent",
        access_type="offline"
    )
except Exception as e:
    print(f"\n[ERROR] Failed to start local server on port 8080: {e}")
    print("If port 8080 is blocked, make sure no other program is using it.")
    sys.exit(1)

token_data = {
    "token": creds.token,
    "refresh_token": creds.refresh_token,
    "token_uri": creds.token_uri,
    "client_id": creds.client_id,
    "client_secret": creds.client_secret,
    "scopes": list(creds.scopes),
}

with open(TOKEN_OUTPUT_FILE, "w") as f:
    json.dump(token_data, f, indent=2)

print("\n" + "=" * 60)
print("[OK] Authorization successful!")
print(f"[OK] Token saved to: {TOKEN_OUTPUT_FILE}")

service = build("gmail", "v1", credentials=creds)
profile = service.users().getProfile(userId="me").execute()
print(f"[OK] Connected Gmail: {profile['emailAddress']}")
print(f"[OK] Total messages : {profile.get('messagesTotal', 0)}")
print("=" * 60)
