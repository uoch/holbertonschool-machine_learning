#!/usr/bin/env python3
"""api module"""
import requests


if __name__ == '__main__':
    launch_url = 'https://api.spacexdata.com/v4/launches/upcoming'
    rocker_url = 'https://api.spacexdata.com/v4/rockets'
    launchpad_url = 'https://api.spacexdata.com/v4/launchpads'

    response = requests.get(launch_url).json()

    dates = [x["date_unix"] for x in response]
    dates.sort()
    next_launch = [x for x in response if x["date_unix"] == dates[0]]
    next_launch_name = next_launch[0]["name"]
    date_launch = next_launch[0]["date_local"]
    rocker_id = next_launch[0]["rocket"]
    launchpad_id = next_launch[0]["launchpad"]
    rocket_name = [x["name"] for x in requests.get(
        rocker_url).json() if x["id"] == rocker_id][0]
    launchpad_name = [x["name"] for x in requests.get(
        launchpad_url).json() if x["id"] == launchpad_id][0]
    launchpad_locality = [x["locality"] for x in requests.get(
        launchpad_url).json() if x["id"] == launchpad_id][0]
    print(
        "{} {} {} {} ({})".format(next_launch_name,
                                  date_launch,
                                  rocket_name,
                                  launchpad_name,
                                  launchpad_locality))
