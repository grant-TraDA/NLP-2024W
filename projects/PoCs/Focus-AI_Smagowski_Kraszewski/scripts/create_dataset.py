import os
import pandas as pd
from pathlib import Path
from html2markdown import convert

def process_document_files(folder_path='data'):
    file_data = []
    path = Path(folder_path)
    
    # Files to ignore
    IGNORE_FILES = {'.DS_Store', '.gitignore', 'Thumbs.db'}
    
    # Process each category
    categories = ['Bands', 'BioMedical', 'Goats', 'Sheep']
    
    for category in categories:
        category_path = path / category
        if not category_path.exists():
            print(f"Category directory not found: {category}")
            continue
            
        print(f"\nProcessing category: {category}")
        
        # Process each file in the category
        for file_path in category_path.glob('*'):
            # Skip if it's not a file or if it's in ignore list
            if not file_path.is_file() or file_path.name in IGNORE_FILES:
                continue
                
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # Basic check if content is HTML
                if not ('<html' in content.lower() or '<title' in content.lower() or '<body' in content.lower()):
                    continue
                    
                # Convert HTML to Markdown
                markdown_content = convert(content)
                
                file_data.append({
                    'filename': file_path.name,
                    'category': category,
                    'original_content': content,
                    'markdown_content': markdown_content,
                    'file_size': file_path.stat().st_size,
                    'last_modified': pd.Timestamp.fromtimestamp(file_path.stat().st_mtime)
                })
                print(f"Processed: {file_path.name}")
                
            except UnicodeDecodeError:
                print(f"Skipping non-text file: {file_path.name}")
            except Exception as e:
                print(f"Error processing {file_path.name}: {str(e)}")
    
    # Create DataFrame
    df = pd.DataFrame(file_data)
    
    # Print summary
    print(f"\nProcessed files summary:")
    print(f"Files successfully processed: {len(df)}")
    if not df.empty:
        print("\nFiles per category:")
        print(df['category'].value_counts())
    
    return df

if __name__ == "__main__":
    df = process_document_files()
    if not df.empty:
        # Save DataFrame to CSV file
        output_file = 'processed-data/processed-files.csv'
        df.to_csv(output_file, index=False)
        print(f"\nDataFrame saved to {output_file}")