# src/integrations/whatsapp_service.py
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class WhatsAppService:
    def __init__(self):
        self.api_url = f"https://graph.facebook.com/v20.0/{os.getenv('META_PHONE_NUMBER_ID')}"
        self.headers = {
            "Authorization": f"Bearer {os.getenv('META_ACCESS_TOKEN')}",
            "Content-Type": "application/json",
        }

    def send_text_message(self, to_number: str, message: str):
        """Sends a simple text message to a user."""
        payload = {
            "messaging_product": "whatsapp",
            "to": to_number,
            "type": "text",
            "text": {"body": message},
        }
        try:
            response = requests.post(f"{self.api_url}/messages", headers=self.headers, json=payload)
            response.raise_for_status()
            print(f"Message sent to {to_number}")
        except requests.exceptions.RequestException as e:
            print(f"Error sending message: {e.response.text}")

    def get_media_url(self, media_id: str) -> str | None:
        """Retrieves the download URL for a given media ID."""
        try:
            response = requests.get(f"https://graph.facebook.com/v20.0/{media_id}", headers=self.headers)
            response.raise_for_status()
            return response.json().get("url")
        except requests.exceptions.RequestException as e:
            print(f"Error getting media URL: {e.response.text}")
            return None

    def download_media(self, media_url: str) -> bytes | None:
        """Downloads the media content from a URL."""
        try:
            auth_header = {"Authorization": f"Bearer {os.getenv('META_ACCESS_TOKEN')}"}
            response = requests.get(media_url, headers=auth_header)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            print(f"Error downloading media: {e.response.text}")
            return None
