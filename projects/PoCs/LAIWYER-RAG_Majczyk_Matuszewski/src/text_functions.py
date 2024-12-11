import textwrap

def format_context(retrieved_articles):
    context_lines = []
    for article in retrieved_articles:
        chapter, chapter_title, article_number, article_text, score = article
        context_lines.append(
            f"Rodział {chapter}: {chapter_title}\nArtykuł {article_number}:\n{article_text}"
        )
    return "\n\n".join(context_lines)

def custom_print(text, width=80):
    # Split the input string by existing newlines, wrap each part, and rejoin with newlines
    wrapped_lines = []
    for line in text.splitlines():
        # Wrap each line individually
        wrapped_lines.extend(textwrap.wrap(line, width=width) if line else [""])

    # Join the wrapped lines with newlines
    wrapped_text = "\n".join(wrapped_lines)
    print(wrapped_text)