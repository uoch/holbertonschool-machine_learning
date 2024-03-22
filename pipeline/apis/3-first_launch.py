#!/usr/bin/env python3
"""api module"""
import requests


if __name__ == '__main__':
    launches = requests.get(
        'https://api.spacexdata.com/v4/launches/upcoming').json()
    unix_dates = [launch['date_unix'] for launch in launches]
    min_idx = unix_dates.index(min(unix_dates))
    upcoming_launch = launches[min_idx]

    rocket = requests.get(
        'https://api.spacexdata.com/v4/rockets/{}'.format(
            upcoming_launch['rocket'])).json()
    launchpad = requests.get(
        'https://api.spacexdata.com/v4/launchpads/{}'.format(
            upcoming_launch['launchpad'])).json()

    print('{} ({}) {} - {} ({})'.format(upcoming_launch['name'],
                                        upcoming_launch['date_local'],
                                        rocket['name'],
                                        launchpad['name'],
                                        launchpad['locality']))
