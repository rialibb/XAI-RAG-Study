import pdfplumber
import re



def extract_text_and_tables(pdf_path):
    """
    Extracts cleaned text and tables from a PDF document.

    Args:
        pdf_path (str): Path to the input PDF file.

    Returns:
        tuple:
            - extracted_text (list of str): Cleaned textual content from each page, excluding exact table rows.
            - extracted_tables (list of lists): List of tables, each represented as a list of rows.

    Process:
        - Opens the PDF file using pdfplumber.
        - Iterates through each page and extracts:
            - Tables as raw lists of rows (added to `extracted_tables`).
            - Text, line by line, excluding lines that match any table row exactly.
        - Cleans and stores the filtered text per page.
    """

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
    """
    Splits a block of text into clean, individual sentences.

    Args:
        text (str): Raw input text to be segmented.

    Returns:
        list of str: List of cleaned sentences extracted from the input text.

    Process:
        - Uses regular expressions to split the text into sentences based on punctuation (period or question mark),
          while avoiding false splits on abbreviations and initials.
        - Strips leading/trailing whitespace from each sentence.
        - Replaces any newline characters within sentences with spaces.
    """

    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return [sentence.strip().replace('\n', ' ') for sentence in sentences]





def preprocess_table(table):
    """
    Formats a table into readable, structured text rows using column headers.

    Args:
        table (list of lists): A table where the first row contains headers and the following rows contain data.

    Returns:
        list of str: List of formatted text rows, each representing one table row with header-value pairs.

    Process:
        - Checks if the table has more than one row (i.e., includes both headers and data).
        - Iterates through each data row (excluding the header).
        - For each cell, pairs it with the corresponding header as 'Header: Value'.
        - Joins all header-value pairs in the row using ' | ' separator.
        - Returns the list of these formatted rows.
    """

    formatted_rows = []
    if table and len(table) > 1:
        headers = table[0]  # Column headers
        for row in table[1:]:
            formatted_row = [f"{headers[i]}: {row[i]}" for i in range(len(headers)) if i < len(row)]
            formatted_rows.append(" | ".join(formatted_row))
    return formatted_rows






def preprocess_pdf(pdf_path):
    """
    Processes a PDF file and returns a combined list of sentence and table-based text chunks.

    Args:
        pdf_path (str): Path to the input PDF file.

    Returns:
        list of str: A combined list of text chunks, including individual sentences and formatted table rows.

    Process:
        - Extracts raw text and tables from the PDF using `extract_text_and_tables`.
        - Splits the extracted text into clean sentences using `preprocess_text`.
        - Converts each extracted table into a list of readable rows using `preprocess_table`.
        - Merges the processed sentences and table rows into one unified list of context chunks.
    """

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