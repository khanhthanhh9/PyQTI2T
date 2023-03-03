import pytesseract
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'  # path for tesseract. 
import pandas as pd 
import re

def run_ocr(img):
    # Run OCR on the image and return the result as a DataFrame
    custom_config = r'--oem 3 --psm 6 -l eng'
    d = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT)
    return pd.DataFrame(d)

def remove_redundant_spaces(string):
# Remove redundant spaces while keeping newlines normally
    pattern = r"(?m)^(?:\S+\s+)+\S+$|\s+"
    string = re.sub(pattern, lambda match: match.group().replace(" ", "") if len(match.group().split()) > 1 else match.group(), string)
    return string

def parse_ocr_result(df):
    # Clean up OCR result and extract text
    df1 = df[(df.conf != '-1') & (df.text != ' ') & (df.text != '')]
    sorted_blocks = df1.groupby('block_num').first().sort_values('top').index.tolist()
    text = ''
    for block in sorted_blocks:
        curr = df1[df1['block_num'] == block] # CURRENT
        sel = curr[curr.text.str.len() >= 1] # get word 
        char_w = (sel.width / sel.text.str.len()).mean()
        prev_par, prev_line, prev_left = 0, 0, 0
        for ix, ln in curr.iterrows():
            if prev_par != ln['par_num']:
                text += '\n'
                prev_par = ln['par_num']
                prev_line = ln['line_num']
                prev_left = 0
            elif prev_line != ln['line_num']:
                text += '\n'
                prev_line = ln['line_num']
                prev_left = 0

            added = 0
            if ln['left'] / char_w > prev_left + 1:
                added = int(ln['left'] / char_w) - prev_left
                text += ' ' * added
            text += ln['text'] + ' '
            prev_left += len(ln['text']) + added + 1
        text = remove_redundant_spaces(text)
        text += '\n'
    return text