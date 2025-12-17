import json


with open("runner.json", mode="r") as f:
    data = json.loads(f.read())


print(data)


