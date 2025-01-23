import os
import json
from text_analyzer import TextAnalyzer
import spacy 


nlp = spacy.load("pl_core_news_sm")


def analyze_single_book(file_path, output_folder):
    """
    Analyze a single .txt book and save the results to a JSON file.
    
    Args:
        file_path (str): Path to the .txt file.
        output_folder (str): Path to the folder where JSON file will be saved.
        
    Returns:
        dict: Dictionary containing the analysis results for the book.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the content of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Analyze the book using TextAnalyzer
    analyzer = TextAnalyzer(content)
    analysis = analyzer.analyze()
    analysis["title"] = os.path.basename(file_path).split('.')[0]  # Use file name (without extension) as title

    # Save the analysis to a JSON file
    output_file = os.path.join(output_folder, f"{analysis['title']}_analysis.json")
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(analysis, json_file, ensure_ascii=False, indent=4)

    return analysis


def analyze_books_in_directory(data_folder, output_folder):
    """
    Analyze all .txt books in a directory and save the results to JSON files.
    
    Args:
        data_folder (str): Path to the folder containing the .txt files.
        output_folder (str): Path to the folder where JSON files will be saved.
        
    Returns:
        list: A list of dictionaries containing the analysis results for each book.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of all .txt files in the folder
    txt_files = [f for f in os.listdir(data_folder) if f.endswith('.txt')]
    if not txt_files:
        raise ValueError("No .txt files found in the specified folder.")

    book_analyses = []
    for file_name in txt_files:
        print(f'Current analysis: {file_name}')
        file_path = os.path.join(data_folder, file_name)
        analysis = analyze_single_book(file_path, output_folder)
        book_analyses.append(analysis)

    return book_analyses


def load_json_files_from_folder(folder_path):
    """
    Sequentially open every JSON file in the given folder.

    Args:
        folder_path (str): Path to the folder containing JSON files.

    Returns:
        list: A list of dictionaries loaded from the JSON files.
    """
    json_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):  # Only process JSON files
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                json_data.append(data)
    return json_data


def create_directories_from_titles(data_list, parent_directory):
    """
    Create directories in the given parent directory based on the 'title' key in each dictionary.

    Args:
        data_list (list): List of dictionaries, each containing a 'title' key.
        parent_directory (str): Path to the parent directory where subdirectories will be created.

    Returns:
        None
    """
    # Ensure the parent directory exists
    os.makedirs(parent_directory, exist_ok=True)

    for data in data_list:
        if 'title' in data:
            # Sanitize the title to make it a valid directory name
            title = data['title'].replace(" ", "_").replace("/", "_")
            dir_path = os.path.join(parent_directory, title)
            os.makedirs(dir_path, exist_ok=True)  # Create the directory


