import wikipediaapi
import requests
from bs4 import BeautifulSoup
import wiki_scrap_functions as fn

def get_section_text(page_title: str, section_title: str, subsection_title: str, subsubsection_title: str, part) -> str:
    """
    This function returns the text of the page from the title.
    :param title: str: The title of the page.
    :return: str: The text of the page.
    """
    part_of_text = part
    wiki_wiki = wikipediaapi.Wikipedia(language='en', user_agent="NLP WUT 2024")
    page = wiki_wiki.page(page_title)
    #print("##### PAGE #####")
    #print(page)
    section = page.section_by_title(section_title)
    if subsection_title != '':
        section = section.section_by_title(subsection_title)
        if subsubsection_title != '':
            section = section.section_by_title(subsubsection_title)
    
    #print("##### SECTION #####")
    #print(section)

    ############################
    merged_title = " ".join([page_title, section_title, subsection_title, subsubsection_title])
    table = fn.section_contains_table(page_title, 
                                    section_title, 
                                    subsection_title, 
                                    subsubsection_title)
    
    table_chunks = []
    if table:
        table_data = fn.get_table_from_section(page_title, 
                                            section_title, 
                                            subsection_title, 
                                            subsubsection_title)
        head = merged_title + " Start of table data. \n" + "\t".join([word for word in table_data[0]]) + "\n"
        tmp = head
        for j in range(1, len(table_data)):
            tmp += ( "\t".join([word for word in table_data[j]]) )
            tmp += "\n"
            if j % fn.CONST_ROWS_IN_CHUNKS == (fn.CONST_ROWS_IN_CHUNKS - 1) or j == (len(table_data) - 1):
                tmp += "End of table data."
                table_chunks.append(tmp)
                tmp = head

    section_text = fn.split_into_chunks(section.text, fn.CONST_CHUNK_SIZE, fn.CONST_CHUNK_OVERLAP)
    if section_text[0] == '':
        section_text = []
    else:
        for k in range(len(section_text)):
            section_text[k] = " ".join([merged_title, section_text[k]])

    section = section_text + table_chunks
    #print(section)
    #print(len(section))
    #print(part_of_text[0])
    tmp = section[part_of_text[0]]
    ############################

    return tmp