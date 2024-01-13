import json
import pandas as pd

with open("requests.json", 'r') as read_file:
    existing_data = json.load(read_file)

frames = []
for _ in range(5000):
    if 'artists' in existing_data[_]:
        if 'artist' in existing_data[_]['artists']:
            frames.append(pd.DataFrame(existing_data[_]['artists']['artist']))
            artist_counts = len(existing_data[_]['artists']['artist'])
artists = pd.concat(frames)
artists = artists.drop(['image'], axis=1)

print(artists.head())
print(artists.info())
print(artists.describe())

print("how many records")
print(pd.Series(artist_counts).value_counts()) 


artists = artists.drop_duplicates().reset_index(drop=True)
print(artists.describe()) 



