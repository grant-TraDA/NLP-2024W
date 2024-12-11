import os
from bs4 import BeautifulSoup

input_folder = "data/raw/mini_page"
output_folder = "data/processed/mini_page"


def parse_html_to_text(input_path, output_path):
    """
    Parse HTML file to extract the body text and save it as a text file.
    """
    with open(input_path, "r", encoding="utf-8") as html_file:
        soup = BeautifulSoup(html_file, "html.parser")

    main_content_div = soup.find("div", class_="et_pb_extra_column_main")
    if not main_content_div:
        print(f"File {input_path} - No div with class 'et_pb_extra_column_main' found.")
        return None

    content = []

    for element in main_content_div.descendants:
        if element.name in [
            "p",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "ul",
            "ol",
            "li",
            "em",
        ]:
            text = (
                element.get_text(separator=" ", strip=True)
                .replace("(kropka)", ".")
                .replace("(at)", "@")
            )
            if element.find("a"):
                a_text = element.find("a").get_text(separator=" ", strip=True)
                if a_text:
                    text = text.split(a_text)[0]
            if text:
                content.append(text)
        elif element.name == "a":
            link_text = (
                element.get_text(separator=" ", strip=True)
                .replace("(kropka)", ".")
                .replace("(at)", "@")
            )
            href = element.get("href")
            if href and content and link_text != content[-1]:
                content.append(f"{link_text} (link: {href})")
            elif href:
                content.append(f"(link: {href})")
        elif element.name == "table":
            table_content = []
            for row in element.find_all("tr"):
                row_content = [
                    cell.get_text(separator=" ", strip=True)
                    for cell in row.find_all(["td", "th"])
                ]
                table_content.append(" | ".join(row_content))
            content.append("\n".join(table_content))

    clean_content = "\n".join(content)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as text_file:
        text_file.write(clean_content)


def process_directory(input_dir, output_dir):
    """
    Recursively process all HTML files in the input directory and save parsed text in the output directory.
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".html"):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(
                    output_dir, relative_path, os.path.splitext(file)[0] + ".txt"
                )

                parse_html_to_text(input_path, output_path)


if __name__:
    process_directory(input_folder, output_folder)
