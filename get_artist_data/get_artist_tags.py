import json
import time
import requests


api_key = '' # personal key
user_name = ''

 
def lastfm_get(data):

    headers = {'user-name': user_name}
    url = 'https://ws.audioscrobbler.com/2.0/'
    data['api_key'] = api_key
    data['format'] = 'json'
    response = requests.get(url, headers, data)
    
    return response


def see_tags(tags):
    
    response = lastfm_get({'method': 'artist.getTopTags','artist':  tags["name"]})
    
    if response.status_code != 200:
        return None

    dictionary_with_tags_and_artist ={"tags":[t['name'] for t in response.json()['toptags']['tag'][:5]],"artist_info":tags}
    
    with open("3_Tag_file.json" , 'r') as read_file:
        existing_data = json.loads(read_file.read())

    existing_data.append(dictionary_with_tags_and_artist)

    with open("3_Tag_file.json" , 'w') as write_file:
        write_file.write( json.dumps(existing_data, indent=4))
    if not getattr(response, 'from_cache', False):
        time.sleep(0.25)

    return " "


def get_tags():
    with open("requests.json" , 'r') as read_file:
        all_artists_info = json.loads(read_file.read())
    artist_info = []

    for index in range(len(all_artists_info)):
        for artist_list_unit in all_artists_info[index]["artists"]["artist"]:
            artist_info.append({"name":artist_list_unit["name"],"playcount":artist_list_unit["playcount"],"listeners":artist_list_unit["listeners"]})
   
    for index,info in enumerate(artist_info):
        see_tags(info)
        print(f"completed >>{(index/len(artist_info))*100}%")
      
    # page += 1

def main():
    get_tags()

if __name__ =="__main__":
    main()