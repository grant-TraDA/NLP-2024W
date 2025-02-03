import json
from datetime import datetime
import os



def prepro_json(path):
    # Read the JSON file
    with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    
    filtered_messages = []
    for message in data['messages']:
        if 'content' in message and message['content']:
            filtered_message = {key: message[key] for key in ("sender_name", "timestamp_ms", "content")}
            filtered_messages.append(filtered_message)
    return filtered_messages
    
def join_jsons(path):
    files = os.listdir(path)
    json_files = [file for file in files if file.endswith('.json')]
    json_files.sort()
    filtered_messages = []
    for file in json_files:
        filtered_messages += prepro_json(os.path.join(path, file))
    
    with open(os.path.join(path, file), 'r', encoding='utf-8') as file:
            data = json.load(file)

    return {"participants": data["participants"], "messages": filtered_messages}

def get_month_year(timestamp_ms):
    dt = datetime.fromtimestamp(timestamp_ms / 1000.0)
    return dt.strftime('%Y-%m')

def groupby_month(messages):
    grouped_messages = {}
    for message in messages:
        month_year = get_month_year(message['timestamp_ms'])
        if month_year not in grouped_messages:
            grouped_messages[month_year] = []
        grouped_messages[month_year].append(message)
    return grouped_messages


def prepro_folder(path, new_folder_name):
    items = os.listdir(path)
    folders = [item for item in items if os.path.isdir(os.path.join(path, item))]
    os.makedirs(new_folder_name, exist_ok=True)

    for folder in folders:
        new_name = folder.split('_')[0]
        person_folder_path = os.path.join(new_folder_name, new_name)
        os.makedirs(person_folder_path, exist_ok=True)

        joined_jsons = join_jsons(os.path.join(path, folder))
        with open(os.path.join(person_folder_path, f'{new_name}.json'), 'w', encoding='utf-8') as file:
            json.dump(joined_jsons, file, ensure_ascii=True, indent=2)
            
        grouped_messages = groupby_month(joined_jsons["messages"])
        grouped_json = {"participants": joined_jsons["participants"], 'messages': grouped_messages}
        with open(os.path.join(person_folder_path, f'{new_name}_grouped.json'), 'w', encoding='utf-8') as file:
            json.dump(grouped_json, file, ensure_ascii=True, indent=2)

def stats(json_path):
    pass

def prepro(path):
    prepro_folder(os.path.join(path, "e2ee_cutover"), "private_chats")
    prepro_folder(os.path.join(path, "inbox"), "group_chats")

    

    

if __name__=="__main__":
    path = r""
    prepro(path)