import json
import os

def group_conv(messages):
  convs = []
  n = len(messages)
  prev_timestamp = messages[-1]['timestamp_ms']
  convs.append([messages[-1]])
  for i in range(n-2, -1, -1):
    timestamp = messages[i]['timestamp_ms']
    if (timestamp - prev_timestamp) < 28800000:
      convs[-1].append(messages[i])
    else:
      convs.append([messages[i]])
    prev_timestamp = timestamp
  return convs

def make_json_convs(data):
  participants = [part['name'] for part in data["participants"]]
  convs = group_conv(data['messages'])
  results = {"conversations": []}
  for i in range(len(convs)):
    conv = {"start_timestamp_ms": convs[i][0]['timestamp_ms'],
            "end_timestamp_ms": convs[i][-1]['timestamp_ms']}
    for part in participants:
      conv[part] = " ".join([msg['content'] for msg in convs[i] if msg['sender_name'] == part])
      conv[f"{part}_translated"] = " ".join([msg['content_translated'] for msg in convs[i] if msg['sender_name'] == part])
    results["conversations"].append(conv)
    results["participants"] = participants
  return results

def process_folder(folder_path):
    folder_name = os.path.basename(folder_path)
    input_file = os.path.join(folder_path, f"{folder_name}_translated.json")
    output_file = os.path.join(folder_path, f"{folder_name}_conversations.json")
    
    if os.path.exists(input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = make_json_convs(data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=True, indent=2)
    else:
        print(f"File {input_file} does not exist.")

if __name__ == "__main__":
    base_dir = r""
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            process_folder(folder_path)