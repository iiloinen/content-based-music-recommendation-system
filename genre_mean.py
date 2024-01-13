import json
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import numpy as np
import os

with open('tag_list.json', 'r') as tag_file:
    tag_list = json.loads(tag_file.read())

# list of a 100 genres
music_genres = [
    "Acid Jazz",
    "Alternative",
    "Alternative pop",
    "Ambient",
    "Blues",
    "Bollywood",
    "Classical",
    "Country",
    "Dance",
    "Dance pop"
    "Disco",
    "Dubstep",
    "EDM",
    "Electronic",
    "Folk",
    "Funk",
    "Gospel",
    "Hip Hop",
    "Hip-hop",
    "House",
    "Indie",
    "Jazz",
    "Latin",
    "Metal",
    "New Age",
    "Opera",
    "Pop",
    "Punk",
    "Rap",
    "Reggae",
    "Rock",
    "Salsa",
    "Samba",
    "Ska",
    "Soul",
    "Swing",
    "Techno",
    "Trance",
    "Trap",
    "World",
    "Alternative Rock",
    "Ambient Pop",
    "Art Rock",
    "Baroque Pop",
    "Bluegrass",
    "Britpop",
    "Chamber Music",
    "Chillout",
    "Classic Rock",
    "Country Pop",
    "Cumbia",
    "Death Metal",
    "Dream Pop",
    "Drum and Bass",
    "Easy Listening",
    "Electro",
    "Emo",
    "Experimental",
    "Fado",
    "Flamenco",
    "Folk Rock",
    "Folk Pop",
    "Freestyle",
    "Fusion",
    "Glam Rock",
    "Garage",
    "Glitch",
    "Grunge",
    "Hard Rock",
    "Hardcore",
    "Heavy Metal",
    "Indie Pop",
    "Indie Rock",
    "Instrumental",
    "Jazz Fusion",
    "K-Pop",
    "Merengue",
    "Lofi",
    "Industrial",
    "Metalcore",
    "Motown",
    "New Wave",
    "Nu Metal",
    "Orchestral",
    "Pop",
    "Rock",
    "Rap",
    "Pop Rock",
    "Post-Punk",
    "Progressive",
    "Progressive Metal",
    "Psychedelic",
    "R&B",
    "Rock and Roll",
    "Rockabilly",
    "Samba Rock",
    "Ska Punk",
    "Soft Rock",
    "Southern Rock",
    "Surf Rock",
    "Symphonic Metal",
    "Synthpop",
    "Tango",
    "Tap",
    "Trap rap",
    "Jazz rap",
    "Trap Metal",
    "Tropical House",
    "UK Garage",
    "Vaporwave",
    "Worldbeat"
]

def make_lowercase(lst):
    modified_list = []
    for item in lst:
        if item and item[0].isupper():
            modified_list.append(item[0].lower() + item[1:])
        else:
            modified_list.append(item)
    return modified_list

list_of_genres = make_lowercase(music_genres)
tag_list_lower = make_lowercase(tag_list)

tags_from_both = [tag  for tag in list_of_genres if tag in tag_list_lower]


genres_with_artists = {}
with open('tags.json', 'r') as file:
    artist_tags = json.loads(file.read())


for dictionary in artist_tags:
    artists_name = dictionary["toptags"]["@attr"]["artist"]
    for tag in dictionary["toptags"]["tag"]:
        if tag["name"] in tags_from_both:
                if artists_name in genres_with_artists:
                    genres_with_artists[artists_name].append(tag["name"])
                else:
                    genres_with_artists[artists_name] = list()
                    genres_with_artists[artists_name].append(tag["name"])

for key in genres_with_artists:
    genres_with_artists[key] = genres_with_artists[key][:5]

i = 0

for key, value in genres_with_artists.items():
    
    i+=1


max_length = max(len(lst) for lst in genres_with_artists.values())
for key in genres_with_artists:
    genres_with_artists[key] += [None] * (max_length - len(genres_with_artists[key]))

df_2 = pd.DataFrame(genres_with_artists, columns = list(genres_with_artists.keys()))


def get_genre_mean():
    artists = []
    genres = []

    for key, values in genres_with_artists.items():
        genres.append(key)
        for value in values:
            artists.append(value)
    label_encoder = LabelEncoder()
    encoded_genres = label_encoder.fit_transform(artists)
    

    for key, value in genres_with_artists.items():
        genres_with_artists[key] = list(label_encoder.transform(value[:5])*10)
        
    def save_to_csv():
        cnt = 0
        lenf = len(list(genres_with_artists.keys()))
        pd_list = list()
    
        for key , value in genres_with_artists.items():
            pd_list.append([key] + value+[round((value[0]*10+value[1]*2+value[2]+value[3]*.5+value[4]*.25)/13.75, 2)])
            
            cnt+=1
            
            print((cnt/lenf)*100)
        df = pd.DataFrame(data = pd_list,  columns = ["name" , "genre_1st","genre_2nd","genre_3rd","genre_4th","genre_5th","weighted_mean"])
        df.to_csv("artists_and_their_five_genres.csv", index = False)


    def add_features():
        file_features= os.listdir("C:\\Users\\Olga\\Desktop\\minbd_projekt\\features")
        base_data_frame = pd.DataFrame([])

        for ind, _ in enumerate(file_features):
            file_path = os.path.join('C:\\Users\\Olga\\Desktop\\minbd_projekt\\features',_)
            with open(file_path, 'r') as read_file:
                
                try:
                    file_dictionary = json.loads(read_file.read())
                    column_list = []
                    file_list = [[]]
                    for key, value in file_dictionary.items():
                        column_list.append(key)
                        file_list[0].append(value)
                       
                    file_pd = pd.DataFrame(data = file_list , columns= column_list)
                
                    base_data_frame = pd.concat([base_data_frame , file_pd], ignore_index = False )
                except:
                    pass
        base_data_frame.to_csv("artists_with_mean_and_features.csv", index = False)

    add_features()


get_genre_mean()



