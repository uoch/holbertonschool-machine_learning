#!/usr/bin/env python3
"""api module"""
import requests
import sys
import time


def get_user_location(api_url):
    """get user location from api"""
    try:
        response = requests.get(api_url)
        if response.status_code == 404:
            print("Not found")
        elif response.status_code == 403:
            reset_time = int(response.headers['X-Ratelimit-Reset'])
            current_time = int(time.time())
            reset_in_seconds = reset_time - current_time
            reset_in_minutes = reset_in_seconds // 60
            print("Reset in {} min".format(reset_in_minutes))
        elif response.status_code == 200:
            user_data = response.json()
            location = user_data.get("location")
            if location:
                print(location)
            else:
                print("Location not available")
        else:
            print("Unexpected status code: {}".format(response.status_code))

    except requests.RequestException as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    """main"""
    if len(sys.argv) != 2:
        print("Usage: python script.py <user_api_url>")
        sys.exit(1)

    user_api_url = sys.argv[1]
    get_user_location(user_api_url)
