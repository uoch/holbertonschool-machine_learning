#!/usr/bin/env python3
"""api module"""
import requests

if __name__ == '__main__':
    base_url = 'https://api.spacexdata.com/v4'
    launch_url = '{}/launches/upcoming'.format(base_url)
    rockets_url = '{}/rockets'.format(base_url)
    launchpads_url = '{}/launchpads'.format(base_url)

    response_launches = requests.get(launch_url).json()
    response_rockets = requests.get(rockets_url).json()
    response_launchpads = requests.get(launchpads_url).json()

    next_launch = min(response_launches, key=lambda x: x['date_unix'])
    next_launch_name = next_launch["name"]
    date_launch = next_launch["date_local"]
    rocket_id = next_launch["rocket"]
    launchpad_id = next_launch["launchpad"]

    rocket = next((r for r in response_rockets if r['id'] == rocket_id), None)
    launchpad = next(
        (lp for lp in response_launchpads if lp['id'] == launchpad_id), None)

    rocket_name = rocket["name"] if rocket else "Unknown Rocket"
    launchpad_name = launchpad["name"] if launchpad else "Unknown Launchpad"
    lpad_locality = launchpad["locality"] if launchpad else "Unknown Locality"

    print(
        "{} ({}) {} - {} ({})".format(next_launch_name,
                                      date_launch,
                                      rocket_name,
                                      launchpad_name,
                                      lpad_locality)
    )
