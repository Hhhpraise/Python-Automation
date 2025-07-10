#!/usr/bin/env python3
"""
Enhanced YouTube Video Uploader with Progress Reporting
-----------------------------------------------------
Improvements:
1. Better progress reporting with print statements
2. More robust upload handling with retries
3. Detailed error reporting
4. Connection timeout handling
"""

import os
import re
import json
import time
import logging
import datetime
import socket
from typing import Dict, List, Optional
import googleapiclient.discovery
import googleapiclient.errors
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("youtube_uploader.log"),
        logging.StreamHandler()
    ]
)

# YouTube API constants
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'
CLIENT_SECRETS_FILE = 'D:/youtube/client_secret.json'
TOKEN_FILE = 'token.json'

# Upload settings
MAX_RETRIES = 3
RETRY_DELAY = 60  # seconds
CHUNK_SIZE = 1024 * 1024 * 10  # 10MB chunks for better progress reporting

# Quota management
UPLOAD_QUOTA_COST = 1600
DAILY_QUOTA_LIMIT = 10000


class YouTubeUploader:
    def __init__(self, videos_dir: str):
        """Initialize the uploader with enhanced progress tracking."""
        self.videos_dir = videos_dir
        self.youtube = self._authenticate()
        self._init_history()

    def _authenticate(self) -> googleapiclient.discovery.Resource:
        """Authenticate with YouTube API with better error handling."""
        print("Authenticating with YouTube API...")
        credentials = None

        if os.path.exists(TOKEN_FILE):
            try:
                credentials = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
                print("Loaded existing credentials")
            except Exception as e:
                print(f"Error loading credentials: {e}")

        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                print("Refreshing expired credentials...")
                credentials.refresh(Request())
            else:
                print("Starting new OAuth flow...")
                flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
                credentials = flow.run_local_server(port=0)

            with open(TOKEN_FILE, 'w') as token:
                token.write(credentials.to_json())
            print("Credentials saved")

        return googleapiclient.discovery.build(
            API_SERVICE_NAME, API_VERSION, credentials=credentials,
            static_discovery=False)  # Avoids discovery cache issues

    def _init_history(self):
        """Initialize or load upload history."""
        self.history_file = 'upload_history.json'
        try:
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
            print("Loaded upload history")
        except (FileNotFoundError, json.JSONDecodeError):
            self.history = {
                "uploaded_videos": [],
                "quota_used": 0,
                "last_reset": datetime.datetime.now().isoformat()
            }
            print("Created new upload history")

    def _save_history(self):
        """Save history with error handling."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")

    def _get_next_video(self) -> Optional[str]:
        """Find the next video to upload with progress reporting."""
        print("\nSearching for next video to upload...")
        for file in os.listdir(self.videos_dir):
            if file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                filepath = os.path.join(self.videos_dir, file)
                print(f"Found video: {file}")
                return filepath
        print("No videos found in directory")
        return None

    def _check_quota(self) -> bool:
        """Check quota with reset handling."""
        now = datetime.datetime.now()
        last_reset = datetime.datetime.fromisoformat(self.history.get("last_reset", now.isoformat()))

        if (now - last_reset).days >= 1:
            print("Resetting daily quota")
            self.history["quota_used"] = 0
            self.history["last_reset"] = now.isoformat()
            self._save_history()

        remaining = DAILY_QUOTA_LIMIT - self.history["quota_used"]
        print(f"Remaining quota: {remaining}/{DAILY_QUOTA_LIMIT}")
        return remaining >= UPLOAD_QUOTA_COST

    def upload_video(self, filepath: str) -> bool:
        """
        Enhanced video upload with detailed progress reporting.

        Returns:
            bool: True if upload succeeded, False otherwise
        """
        if not os.path.exists(filepath):
            print(f"Error: File not found - {filepath}")
            return False

        if not self._check_quota():
            print("Insufficient quota for upload")
            return False

        filename = os.path.basename(filepath)
        title = os.path.splitext(filename)[0]
        filesize = os.path.getsize(filepath)
        print(f"\nPreparing to upload: {filename} ({filesize / 1024 / 1024:.2f} MB)")

        body = {
            "snippet": {
                "title": title,
                "description": "Uploaded by automated YouTube uploader",
                "categoryId": "22"
            },
            "status": {
                "privacyStatus": "private"
            }
        }

        try:
            media = MediaFileUpload(
                filepath,
                chunksize=CHUNK_SIZE,
                resumable=True,
                mimetype='video/mp4'
            )

            request = self.youtube.videos().insert(
                part=",".join(body.keys()),
                body=body,
                media_body=media
            )

            print("Starting upload...")
            response = None
            retry = 0

            while response is None and retry < MAX_RETRIES:
                try:
                    status, response = request.next_chunk()
                    if status:
                        progress = status.progress() * 100
                        print(f"Upload progress: {progress:.1f}%")

                except socket.timeout:
                    retry += 1
                    print(f"Timeout occurred, retry {retry}/{MAX_RETRIES}")
                    time.sleep(RETRY_DELAY)
                    continue
                except ConnectionError as e:
                    retry += 1
                    print(f"Connection error: {e}, retry {retry}/{MAX_RETRIES}")
                    time.sleep(RETRY_DELAY)
                    continue
                except googleapiclient.errors.HttpError as e:
                    print(f"HTTP Error: {e}")
                    return False

            if response:
                video_id = response.get("id")
                print(f"Upload complete! Video ID: {video_id}")

                # Update history
                self.history["quota_used"] += UPLOAD_QUOTA_COST
                self.history["uploaded_videos"].append({
                    "video_id": video_id,
                    "title": title,
                    "timestamp": datetime.datetime.now().isoformat()
                })
                self._save_history()

                # Delete original file
                try:
                    os.remove(filepath)
                    print(f"Deleted original file: {filename}")
                except Exception as e:
                    print(f"Warning: Could not delete file - {e}")

                return True

            print("Upload failed after maximum retries")
            return False

        except Exception as e:
            print(f"Upload failed with error: {e}")
            return False

    def run(self):
        """Main execution method with user feedback."""
        print("\n" + "=" * 50)
        print("YouTube Auto Uploader")
        print("=" * 50)

        while True:
            video_path = self._get_next_video()
            if not video_path:
                print("No more videos to upload")
                break

            if self.upload_video(video_path):
                print("Upload successful, processing next video...")
            else:
                print("Upload failed, waiting before retry...")
                time.sleep(RETRY_DELAY)

        print("\nUpload session complete")


def main():
    """Main function with error handling."""
    try:
        uploader = YouTubeUploader("D:/youtube/processed")
        uploader.run()
    except KeyboardInterrupt:
        print("\nUpload interrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        logging.exception("Uploader crashed")


if __name__ == "__main__":
    main()