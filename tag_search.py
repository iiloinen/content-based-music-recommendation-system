import json
import pandas as pd


def lookup_tags(artist):
    with open("requests.json", 'r') as read_file:
        artist_data = json.load(read_file)
    for index in range(len(artist_data)):
        if 'artists' in artist_data[index]:
            if 'artist' in artist_data[index]['artists']:
                for artist_dict in artist_data[index]["artists"]["artist"]:
                    if artist_dict["name"] == artist:
                        artist_data_frame = pd.DataFrame({
                            "name" : artist_dict["name"],
                            "playcount":artist_dict["playcount"],
                            "listeners":artist_dict["listeners"] 
                                                          }, index = [0])
    
    return artist_data_frame

print(lookup_tags("")) # artist name


