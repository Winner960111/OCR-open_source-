from paddleocr import PaddleOCR, draw_ocr
from ast import literal_eval
import json
import cv2
import re
from dateutil import parser
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Extract contents from Passport images
paddleocr = PaddleOCR(lang="en",ocr_version="PP-OCRv4",show_log = False,use_gpu=True)
receipt_image_array = cv2.imread('EID-1.jpg')

def paddle_scan(paddleocr,img_path_or_nparray):
    result = paddleocr.ocr(img_path_or_nparray,cls=True)
    result = result[0]
    boxes = [line[0] for line in result]       #boundign box 
    txts = [line[1][0] for line in result]     #raw text
    scores = [line[1][1] for line in result]   # scores
    return  txts, result

# perform ocr scan
receipt_texts, receipt_boxes = paddle_scan(paddleocr,receipt_image_array)
print(50*"--","\ntext only:\n",receipt_texts)

# receipt_texts =  ['LHn', 'IDNumber/', '784-1980-7549106-2', 'Name:Swapan Ghosh Rabindranath Ghosh', 'Date of Birth', '01/01/1980', 'Nationality.Bangladesh', 'Issuing Date/Jy', '05/02/2024', 'S', 'Signature/', 'Expiry Date/yl', 'Sex.', 'M', '03/02/2026']

# Load the model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
model_name = "Trelis/Llama-2-7b-chat-hf-function-calling-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Define your function metadata
function_metadata = {
    "name": "classify_text",
    "description": "Extract the essential information from the text below and return it in a json format.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": f"Give me the full name of the person.",
            },
            "ID_number": {
                "type": "string",
                "description": f"Give me the ID number of the person.",
            },
            "date_of_birth": {
                "type": "string",
                "description": f"Give me the date of birth of the person.",
            },
            "nationality": {
                "type": "string",
                "description": f"Give me the nationality of the person.",
            },
            "issuing_date": {
                "type": "string",
                "description": f"Give me the issuing date of the person.",
            },
            "expiry_date": {
                "type": "string",
                "description": f"Give me the expiry date of the person.",
            },
            "issuing_place": {
                "type": "string",
                "description": f"Give me the issuing place of the person.",
            },
            "gender": {
                "type": "string",
                "description": f"Give me the gender of the person.",
            },
            "occupation": {
                "type": "string",
                "description": f"Give me the occupation of the person.",
            },
            "employer":{
                "type": "string",
                "description": f"Give me the employer of the person. If this information is not available, return 'N/A'.",
            }

        },
        "required": ["name", "ID_number", "date_of_birth", "nationality", "issuing_date", "issuing_place", "gender", "occupation", "employer"],
    }
}

# Format your prompt with function metadata included
function_metadata_str = json.dumps(function_metadata)
user_prompt = f"Plz extract essential information from the text below and return it in JSON format: {receipt_texts}"
formatted_prompt = f"<FUNCTIONS>{function_metadata_str}</FUNCTIONS>\n\n[INST]{user_prompt}[/INST]"

# Encode input
inputs = tokenizer(formatted_prompt, return_tensors="pt")

# Generate response
outputs = model.generate(**inputs)
response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Parse JSON response
try:
    response_json = json.loads(response_text)
    print("Function Call:", response_json)
except json.JSONDecodeError:
    print("Response is not valid JSON:", response_text)