# Skrypt: Zastąpienie co 10. słowa na [GAP] i zapisanie wyniku do nowego pliku

def replace_with_gap(input_file, output_file):
    try:
        # Wczytanie zawartości pliku wejściowego
        with open(input_file, 'r', encoding='utf-8') as file:
            content = file.read()

        # Tokenizacja tekstu na słowa
        words = content.split()

        # Zamiana co 10. słowa na [GAP]
        gap_count = 0
        for i in range(9, len(words), 10):
            words[i] = '[GAP]'
            gap_count += 1

        # Połączenie słów z powrotem w tekst
        modified_content = ' '.join(words)

        # Zapisanie zmodyfikowanego tekstu do pliku wyjściowego
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(modified_content)

        print(f"Zamiana zakończona! Wynik zapisano w pliku: {output_file}")
        print(f"Liczba zastąpionych słów: {gap_count}")
    except Exception as e:
        print(f"Wystąpił błąd: {e}")



# Przykład użycia:
input_file = '../data_processing_scripts/blanks_new/blanks_new_answer/balladyna.txt'  # Plik wejściowy z oryginalnym tekstem
output_file = 'output.txt'  # Plik wyjściowy z zamienionymi słowami

replace_with_gap(input_file, output_file)