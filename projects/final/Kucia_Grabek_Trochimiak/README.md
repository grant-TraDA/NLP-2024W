# Artistic-Chatbot
Project for 15-aniversary of faculty of Media Arts and NLP project


For data processing:
- Marker https://github.com/VikParuchuri/marker?tab=readme-ov-file  but it returns Markdown, but then to txt should be ok
  - example single use:  marker_single raw/mysl-sztuka-net.pdf xd --batch_multiplier 2 --max_pages 10 --langs Polish
  - multiple use: marker /path/to/input/folder /path/to/output/folder --workers 4 --max 10

And then maybe using pip install strip-markdown  or https://gist.githubusercontent.com/lorey/eb15a7f3338f959a78cc3661fbc255fe/raw/1cdd946a1b62452b6277a8358819b947187bb480/markdown_to_text.py
- https://medium.com/@harikrishnank497/how-to-convert-a-pdf-file-to-a-txt-file-locally-using-python-bc82c1403749  great source pdf to txt
```{python}
import fitz
def pdf_to_text(pdf_path, txt_path):
 # Open the PDF
 pdf_document = fitz.open(pdf_path)
 
 # Create a text file to store the extracted text
 with open(txt_path, "w", encoding="utf-8") as text_file:
 for page_number in range(len(pdf_document)):
 page = pdf_document.load_page(page_number)
 text = page.get_text()
 text_file.write(text)
 
 # Close the PDF
 pdf_document.close()
# Example usage
pdf_path = "/path/to/your/input.pdf"
txt_path = "/path/to/your/output.txt"
pdf_to_text(pdf_path, txt_path)
```
