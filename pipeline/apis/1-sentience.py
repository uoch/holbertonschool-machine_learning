#!/usr/bin/env python3
"""api module"""
import requests as rq


def sentientPlanets():
    """ count the number of sentient planets """
    url = "https://swapi-api.hbtn.io/api/species"
    planets = []
    while url is not None:
        response = rq.get(url)
        data = response.json()
        result = data["results"]
        for planet in result:
            classification = planet["classification"]
            designation = planet["designation"]
            if classification in ["sentient",] or designation in ["sentient"]:
                url2 = planet["homeworld"]
                if url2 is not None:
                    response2 = rq.get(url2)
                    data2 = response2.json()
                    planets.append(data2["name"])
        url = data["next"]
    return planets
