import os
from tqdm import tqdm


def clean_text_files(input_folder, output_folder):
    """
    Processes text files in a folder to keep only the content after
    "Proszę, napisz zwięzłą odpowiedź." until the next line with "===".
    Each response is separated by "\n===\n" in the output.
    """
    os.makedirs(output_folder, exist_ok=True)
    input_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]

    for filename in tqdm(input_files, desc="Cleaning files", unit="file"):
        try:
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with open(input_path, "r", encoding="utf-8") as infile:
                lines = infile.readlines()

            cleaned_lines = []
            capture = False
            temp_response = []

            for line in lines:
                if "Proszę, napisz zwięzłą odpowiedź." in line:
                    capture = True
                    temp_response = [line.split("Proszę, napisz zwięzłą odpowiedź.", 1)[-1].strip()]
                elif "===" in line:
                    if capture and temp_response:
                        cleaned_lines.append(" ".join(temp_response).strip() + "\n===\n")
                        temp_response = []
                    capture = False
                elif capture:
                    temp_response.append(line.strip())

            # Add any leftover response if file does not end with ===
            if capture and temp_response:
                cleaned_lines.append(" ".join(temp_response).strip() + "\n===\n")

            with open(output_path, "w", encoding="utf-8") as outfile:
                outfile.writelines(cleaned_lines)

            print(f"Cleaned and saved: {filename}")
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            
            
if __name__ == "__main__":
    try:
        input_folder = "../data_processing_scripts/q_a/answers_mistral"
        output_folder = "../data_processing_scripts/q_a/answers_mistral_clean"

        print("Starting cleaning process...")
        clean_text_files(input_folder, output_folder)

        print(f"All cleaned files saved to '{output_folder}'.")
    except Exception as e:
        print(f"Critical error: {e}")
