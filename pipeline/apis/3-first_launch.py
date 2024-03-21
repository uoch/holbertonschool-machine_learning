#!/usr/bin/env python3
"""api module"""
import requests
import sys
import time






if __name__=='__main__':
    url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(url)
    data = response.json()
    rockets = {}
    for launch in data:
        rocket_id = launch["rocket"]
        rocket_url = "https://api.spacexdata.com/v4/rockets/" + rocket_id
        rocket_response = requests.get(rocket_url)
        rocket_data = rocket_response.json()
        rocket_name = rocket_data["name"]
        if rocket_name in rockets:
            rockets[rocket_name] += 1
        else:
            rockets[rocket_name] = 1
    sorted_rockets = sorted(rockets.items(), key=lambda x: x[1], reverse=True)
    for rocket in sorted_rockets:
        print("{}: {}".format(rocket[0], rocket[1]))
    sys.stdout.flush()
    time.sleep(10)
    sys.exit(0)