import json


with open('tags.json', 'r') as tag_file:
    tag_json = json.loads(tag_file.read())

with open('tag_list.json', 'w') as written_file:
    genre_names_list = []

    cnt =0
    for dictionary in tag_json:
         for tag in dictionary["toptags"]["tag"]:
            if tag["name"] not in genre_names_list:
                genre_names_list.append(tag["name"])
                print(cnt)
            cnt+=1

    print(genre_names_list)
    
    written_file.write(json.dumps(genre_names_list))



