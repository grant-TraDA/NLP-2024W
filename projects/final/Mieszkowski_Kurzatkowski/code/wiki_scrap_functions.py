import vector_database
import embedding
import wikipediaapi
import pandas as pd
import numpy as np
import copy
from itertools import compress
import requests
from bs4 import BeautifulSoup

CONST_CHUNK_SIZE = 60
CONST_CHUNK_OVERLAP = 10
CONST_ROWS_IN_CHUNKS = 4

def get_articles_by_categories(included_categories, 
                               wiki, 
                               topic_keywords, 
                               exclude_keywords, 
                               year_range=(2023, 2024)):
    """
    Fetches Wikipedia articles from multiple specified categories and their subcategories.
    Filters articles for relevance to politics, conflicts, and elections in Russia, as well as time constraints.
    
    Args:
        included_categories (list): A list of root categories to search articles in.
        year_range (tuple): A range of years for relevant articles (inclusive).

    Returns:
        list: A list of relevant Wikipedia article titles.
    """
    relevant_articles = []
    
    '''
    # Keywords to focus on specific topics
    topic_keywords = [
        "Russia", "Ukraine", "war", "conflict", "elections", "politics",
        "government", "diplomacy", "Putin", "Crimea", "sanctions", "NATO", "relations", "China", "summit"
    ]
    
    # Keywords to exclude irrelevant topics
    exclude_keywords = ["football", "soccer", "FIFA", "UEFA", "Wrestling", "Cup", "Sport", "Sports"]
    '''

    def get_subcategories(category):
        num_of_articles = 0
        """Recursively explore subcategories to find articles."""
        for subcat in category.categorymembers.values():
            if subcat.ns == 14:  # Namespace 14 is for categories
                print(f"Exploring subcategory: {subcat.title}")
                get_subcategories(subcat)
            elif subcat.ns == 0:  # Namespace 0 is for articles
                if is_relevant_article(subcat.title):
                    relevant_articles.append(subcat.title)
                    num_of_articles += 1
                    if (num_of_articles % 100) == 0:
                        print("Number of found articles: ", num_of_articles)

    def is_relevant_article(title):
        """Determines if an article is relevant based on title."""
        title_lower = title.lower()
        
        # Check for topic relevance
        if not any(keyword.lower() in title_lower for keyword in topic_keywords):
            return False

        # Exclude irrelevant topics
        if any(exclude.lower() in title_lower for exclude in exclude_keywords):
            return False

        # Ensure the year is within the specified range
        if not any(str(year) in title for year in range(year_range[0], year_range[1] + 1)):
            return False

        return True

    for category_name in included_categories:
        category = wiki.page(f"Category:{category_name}")
        
        # Check if the category exists
        if not category.exists():
            print(f"Warning: The category '{category_name}' does not exist.")
            continue
        
        try:
            get_subcategories(category)
        except KeyError as e:
            print(f"Error while processing category '{category_name}': {e}")
            print("This may indicate an empty category or unexpected API response.")

    return relevant_articles

def get_text_from_metadata(metadata_singular, wiki):
    if metadata_singular['page_title'] != 0:
        level = wiki.page(metadata_singular['page_title'])
        if metadata_singular['section_title'] != 0:
            level = level.section_by_title(metadata_singular['section_title'])
            if metadata_singular['subsection_title']:
                level = level.section_by_title(metadata_singular['subsection_title'])
                if metadata_singular['subsubsection_title'] != '':
                    level = level.section_by_title(metadata_singular['subsubsection_title'])
    return level.text

def get_section_from_metadata(metadata_singular, wiki):
    if metadata_singular['page_title'] != 0:
        level = wiki.page(metadata_singular['page_title'])
        if metadata_singular['section_title'] != 0:
            level = level.section_by_title(metadata_singular['section_title'])
            if metadata_singular['subsection_title']:
                level = level.section_by_title(metadata_singular['subsection_title'])
                if metadata_singular['subsubsection_title'] != '':
                    level = level.section_by_title(metadata_singular['subsubsection_title'])
    return level
    

def split_into_chunks(text, chunk_size, overlap):
    text = text.split(' ')
    tmp = []
    #print(len(text))
    for i in range(0, len(text), chunk_size-overlap):
        #print(i)
        tmp.append(' '.join(text[i:i + chunk_size]))
    return tmp

def remove_unrelevant_sections(metadata):
    sections_to_remove = [
        "References", 
        "External links", 
        "See also", 
        "Further reading", 
        "Notes",
        "Sources",
        "Gallery",
        "Citations"
    ]
    mask = [True for _ in range(len(metadata))]
    i = 0
    for dic in metadata:
        #print(dic['section_title'])
        #print(any(sect_to_rem in dic['section_title'] for sect_to_rem in sections_to_remove))
        mask[i] = (not any(sect_to_rem in dic['section_title'] for sect_to_rem in sections_to_remove))
        i += 1

    metadata2 = list(compress(metadata, mask))
    for dic in metadata2:
        print(dic)
    print(len(metadata2))
    return metadata2

def section_contains_table(article_title, section_title, subsection_title='', subsubsection_title=''):
    base_url = "https://en.wikipedia.org/wiki/"
    url = base_url + article_title.replace(" ", "_")

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch the article '{article_title}'. HTTP Status: {response.status_code}")
        return False

    soup = BeautifulSoup(response.content, "html.parser")
    current_section = ''
    current_subsection = ''
    current_subsubsection = ''

    # Iterate through all tags to find headers and tables
    for tag in soup.find_all(['h2', 'h3', 'h4', 'table']):
        if tag.name.startswith('h'):
            header_text = tag.get_text(strip=True).replace("[edit]", "")
            if tag.name == 'h2':
                current_section = header_text
                current_subsection = ''
                current_subsubsection = ''
            elif tag.name == 'h3':
                current_subsection = header_text
                current_subsubsection = ''
            elif tag.name == 'h4':
                current_subsubsection = header_text
        
        elif tag.name == 'table':  # Check for tables under the current section
            if (current_section == section_title and
                (subsection_title == '' or current_subsection == subsection_title) and
                (subsubsection_title == '' or current_subsubsection == subsubsection_title)):
                return True

    return False

def get_table_from_section(article_title, section_title, subsection_title='', subsubsection_title=''):
    base_url = "https://en.wikipedia.org/wiki/"
    url = base_url + article_title.replace(" ", "_")

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch the article '{article_title}'. HTTP Status: {response.status_code}")
        return ''

    soup = BeautifulSoup(response.content, "html.parser")
    current_section = ''
    current_subsection = ''
    current_subsubsection = ''

    # Iterate through all tags to find headers and tables
    for tag in soup.find_all(['h2', 'h3', 'h4', 'table']):
        if tag.name.startswith('h'):
            header_text = tag.get_text(strip=True).replace("[edit]", "")
            if tag.name == 'h2':
                current_section = header_text
                current_subsection = ''
                current_subsubsection = ''
            elif tag.name == 'h3':
                current_subsection = header_text
                current_subsubsection = ''
            elif tag.name == 'h4':
                current_subsubsection = header_text
        
        elif tag.name == 'table':  # Check for tables under the current section
            if (current_section == section_title and
                (subsection_title == '' or current_subsection == subsection_title) and
                (subsubsection_title == '' or current_subsubsection == subsubsection_title)):
                # Extract table data
                table_data = []
                for row in tag.find_all('tr'):
                    cells = row.find_all(['th', 'td'])
                    row_data = [cell.get_text(strip=True) for cell in cells]
                    table_data.append(row_data)
                return table_data

    return ''