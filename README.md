# Installation and Running Instructions
1. **Install the required dependency**:
```bash
   pip install pymupdf
```

2. **Run the extract_text_from_pdf.py script**
```bash
python extract_text_from_pdf.py
```

3. **Copy the extracted text:**:
After running the script, the text will be saved in SAMPLE_TEXT_FOR_BENCH.txt. Open this file and copy its content.

4. **Update the constant in _etl_a9number_v3.py script**:
Paste the copied content from SAMPLE_TEXT_FOR_BENCH.txt into the SAMPLE_TEXT_FOR_BENCH constant in the Python
```python
# TODO 1: the text extracted from the PDF needs to be added inside this constant
SAMPLE_TEXT_FOR_BENCH = "..."  # Paste the extracted content here
```

5. **Run the PDF extraction script:**
Once you've completed the necessary modifications, run the tests using the following command:
```bash
python -m pytest _etl_a9number_v3.py
```
   
