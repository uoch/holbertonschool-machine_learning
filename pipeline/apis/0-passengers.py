#!/usr/bin/env python3
"""api module"""
import requests as rq
import json


def availableShips(passengerCount):
    """count the number of passengers in the starships"""
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
