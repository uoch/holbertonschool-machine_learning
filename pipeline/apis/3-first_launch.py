#!/usr/bin/env python3
"""api module"""
import requests

if __name__ == '__main__':
    # Define the API endpoints
    base_url = 'https://api.spacexdata.com/v4'
    launch_url = f'{base_url}/launches/upcoming'
    rockets_url = f'{base_url}/rockets'
    launchpads_url = f'{base_url}/launchpads'

    # Fetch data for upcoming launches, rockets, and launchpads
    response_launches = requests.get(launch_url).json()
    response_rockets = requests.get(rockets_url).json()
    response_launchpads = requests.get(launchpads_url).json()

    # Extract data for the next launch
    next_launch = min(response_launches, key=lambda x: x['date_unix'])
    next_launch_name = next_launch["name"]
    date_launch = next_launch["date_local"]
    rocket_id = next_launch["rocket"]
    launchpad_id = next_launch["launchpad"]

    # Retrieve rocket and launchpad details using dictionary lookup
    rocket = next((r for r in response_rockets if r['id'] == rocket_id), None)
    launchpad = next((lp for lp in response_launchpads if lp['id'] == launchpad_id), None)

    # Extract relevant information
    rocket_name = rocket["name"] if rocket else "Unknown Rocket"
    launchpad_name = launchpad["name"] if launchpad else "Unknown Launchpad"
    launchpad_locality = launchpad["locality"] if launchpad else "Unknown Locality"

    # Print the formatted information
    print(
        f"{next_launch_name} {date_launch} {rocket_name} {launchpad_name} ({launchpad_locality})"
    )
