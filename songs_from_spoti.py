import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json

client_credentials_manager = SpotifyClientCredentials(client_id='', client_secret='')
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

file_name_list = os.listdir('') # directory
cnt = 0
for name in file_name_list:
    file_path = os.path.join('', name) # directory
    with open(file_path, 'r') as read_file:
        file_dictionary = json.loads(read_file.read())
    read_file.close()
    for playlists_dictionary in file_dictionary["playlists"]:
        for track in playlists_dictionary["tracks"]:
    
            track_uri = track["track_uri"]
            artist_name = track["artist_name"]
            track_name = track["track_name"]
            try:
                audio_features = sp.audio_features(track_uri)[0]
                audio_features = audio_features
                audio_features["ARTIST"] = artist_name
                audio_features["NAME"] = track_name
                audio_features = json.dumps(audio_features)
                with open(f'{artist_name}-{track_name}.json', 'w') as write_file:
                    write_file.write(audio_features)
                write_file.close()
                print(f"loaded >> {cnt} << {artist_name}||{track_name}||{track_uri}")
            except KeyboardInterrupt:
                break
            except:
                print(f"skipped >> {track_uri}")
