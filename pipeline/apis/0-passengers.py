#!/usr/bin/env python3
import requests as rq
import json
"""api module"""


def availableShips(passengerCount):
    url = "https://swapi-api.hbtn.io/api/starships/"
    ships = []
    while url is not None:
        response = rq.get(url)
        data = response.json()
        result = data["results"]
        for ship in result:
            passengers = ship["passengers"]
            if passengers.isnumeric() and int(passengers) >= passengerCount:
                ships.append(ship["name"])
        url = data["next"]
    return ships
