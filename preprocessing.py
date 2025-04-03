import pdfplumber
import re



def extract_text_and_tables(pdf_path):
    extracted_text = []
    extracted_tables = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract tables first
            tables = page.extract_tables()
            for table in tables:
                extracted_tables.append(table)

            # Extract text while keeping table titles
            text = page.extract_text()
            if text:
                text_lines = text.split("\n")  # Split into lines for better control
                filtered_lines = []

                # Identify table rows to exclude
                table_rows = set()
                for table in tables:
                    for row in table:  # Skip header row
                        table_rows.add(" ".join([str(cell) for cell in row if cell]))  # Combine row cells into string

                # Keep text lines that are not found in extracted tables
                for line in text_lines:
                    if line.strip() and line.strip() not in table_rows:  # Only exclude exact matches
                        filtered_lines.append(line.strip())

                cleaned_text = "\n".join(filtered_lines)
                extracted_text.append(cleaned_text)

    return extracted_text, extracted_tables




def preprocess_text(text):
    """Splits text into sentences."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return [sentence.strip().replace('\n', ' ') for sentence in sentences]




def preprocess_table(table):
    """Formats table rows into structured text."""
    formatted_rows = []
    if table and len(table) > 1:
        headers = table[0]  # Column headers
        for row in table[1:]:
            formatted_row = [f"{headers[i]}: {row[i]}" for i in range(len(headers)) if i < len(row)]
            formatted_rows.append(" | ".join(formatted_row))
    return formatted_rows





def preprocess_pdf(pdf_path):
    """Gives a list of chunks"""
    text_data, table_data = extract_text_and_tables(pdf_path)
    # Process extracted data
    processed_text = []
    for text in text_data:
        processed_text.extend(preprocess_text(text)) 
    # list of lines 
    processed_tables = []
    for table in table_data:
        processed_tables.extend(preprocess_table(table)) 
    
    return processed_text + processed_tables