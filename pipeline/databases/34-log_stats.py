#!/usr/bin/env python3
"""database module"""
from pymongo import MongoClient


if __name__ == "__main__":
    client = MongoClient()
    db = client["logs"]
    collection = db["nginx"]
    tot_logs = collection.count_documents({})
    print("{} logs".format(tot_logs))
    http_methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    print("Methods:")
    for method in http_methods:
        count = collection.count_documents(
            {"method": method})
        print("\tmethod {}: {}".format(method, count))
    status_count = collection.count_documents(
        {"method": "GET", "path": "/status"})
    print("{} status check".format(status_count))
