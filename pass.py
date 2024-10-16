from paddleocr import PaddleOCR, draw_ocr
from ast import literal_eval
import json
import cv2
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Extract contents from ID images
paddleocr = PaddleOCR(lang="en",ocr_version="PP-OCRv4",show_log = False,use_gpu=True)
receipt_image_array = cv2.imread('Passport.jpg')

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

# receipt_texts =  ['UNHEDARAB EMIRAIES', 'Passport', 'Type', 'Country Code', 'Passport No', 'P', 'ARE', 'YG5306428', 'Names', 'Esblljlayeolaye', 'GHARIB SULAIMAN GHARIB SULAIMAN ALMUTAWA', 'Nationality', 'United Arab Emirates', 'y', 'Will', 'Date of Birth', 'Sex', 'M', '26/11/1990', 'Place of Birth', 'DUBAI', 'Date of Expiry', 'Date of Issue', '13/10/2025', '14/10/2020', 'Issuing Authorityy', 'Holders Signature ', 'DUBAI', '5', 'P<AREALMUTAWA<<GHARIB<SULAIMAN<GHARIB<SULAIM', 'YG53064280ARE9011263M2510136<<<<<2']

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
            "first_name": {
                "type": "string",
                "description": f"Give me the first name of the person.",
            },
            "last_name": {
                "type": "string",
                "description": f"Give me the last name of the person.",
            },
            "country_code": {
                "type": "string",
                "description": f"Give me the country code in the given text.",
            },
            "passport_no": {
                "type": "string",
                "description": f"Give me the passport number in the given text.",
            },
            "date_of_birth": {
                "type": "string",
                "description": f"Give me the date of birth of the person.",
            },
            "nationality": {
                "type": "string",
                "description": f"Give me the nationality of the person.",
            },
            "date_of_issue": {
                "type": "string",
                "description": f"Give me the date of issue in the given text.",
            },
            "date_of_expiry": {
                "type": "string",
                "description": f"Give me the date of expiry in the given text.",
            },
            "issuing_authority": {
                "type": "string",
                "description": f"Give me the issuing authority in the given text.",
            },
            "gender": {
                "type": "string",
                "description": f"Give me the gender of the person.",
            },
            "passport_type": {
                "type": "string",
                "description": f"Give me the passport type in the given text.",
            },

        },
        "required": ["first_name", "last_name", "country_code", "passport_no", "date_of_birth", "nationality", "date_of_expiry", "date_of_issue", "issuing_authority", "gender", "passport_type"],
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