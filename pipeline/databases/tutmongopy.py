from pymongo import MongoClient

client = MongoClient()
db = client.rptutorials

tutorial1= {
    "title": "Working With JSON Data in Python",
    "author": "Lucas",
    "contributors": [
        "Aldren",
        "Diana"
    ],  
    "url": "https://realpython.com/python-json/"
}
tutorial = db.tutorial
result = tutorial.insert_one(tutorial1)
print(f"One tutorial: {result.inserted_id}")