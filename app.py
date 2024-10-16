
from paddleocr import PaddleOCR, draw_ocr
import json
import cv2
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import base64
from flask_cors import CORS
# Using flask to make an api 
# import necessary libraries and functions 
from flask import Flask, jsonify, request 

# creating a Flask app 
app = Flask(__name__) 
CORS(app, origins = '*')
# on the terminal type: curl http://127.0.0.1:5000/ 
# returns hello world when we use GET. 
# returns the data that we send when we use POST. 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
paddleocr = PaddleOCR(lang="en",ocr_version="PP-OCRv4",show_log = False,use_gpu=True)

def paddle_scan(paddleocr,img_path_or_nparray):
    result = paddleocr.ocr(img_path_or_nparray,cls=True)
    result = result[0]
    boxes = [line[0] for line in result]       #boundign box 
    txts = [line[1][0] for line in result]     #raw text
    scores = [line[1][1] for line in result]   # scores
    return  txts, result

def base64_to_image(base64_string, output_image_path):
    # Decode the Base64 string
    image_data = base64.b64decode(base64_string)
    
    # Write the binary data to an image file
    with open(output_image_path, "wb") as image_file:
        image_file.write(image_data)

def classification(text, format):
    # Load the model and tokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.is_available())

        model_name = "Trelis/Llama-2-7b-chat-hf-function-calling-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        # Format your prompt with function metadata included
        function_metadata_str = json.dumps(format)
        user_prompt = f"Plz extract essential information from the text below and return it in JSON format: {text}"
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
            return(response_json)
        except json.JSONDecodeError:
            print("Response is not valid JSON:", response_text)
            return("Response is not valid JSON:", response_text)

@app.route('/id', methods = ['POST']) 
def id(): 
    if(request.method == 'POST'): 

        base64_code = request.json['base64_code']
        base64_to_image(base64_code, 'id.jpg')

        # Extract contents from passport images
        receipt_image_array = cv2.imread('id.jpg')

        # perform ocr scan
        receipt_texts, receipt_boxes = paddle_scan(paddleocr,receipt_image_array)
        print(50*"--","\ntext only:\n",receipt_texts)

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
        result = classification(receipt_texts, function_metadata)
        return jsonify({'data': result})
    
    else:
        return jsonify({'data': 'hello world'})
  
  
# A simple function to calculate the square of a number 
# the number to be squared is sent in the URL when we use GET 
# on the terminal type: curl http://127.0.0.1:5000 / home / 10 
# this returns 100 (square of 10) 
@app.route('/passport', methods = ['POST']) 
def passport(): 
    if (request.method == 'POST'):
        base64_code = request.json['base64_code']
        base64_to_image(base64_code, 'passport.jpg')

        # Extract contents from passport images
        receipt_image_array = cv2.imread('passport.jpg')

        # perform ocr scan
        receipt_texts, receipt_boxes = paddle_scan(paddleocr,receipt_image_array)
        print(50*"--","\ntext only:\n",receipt_texts)

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

        result = classification(receipt_texts, function_metadata)
        return jsonify({'data': result})
    else:
        return jsonify({'data': 'hello world'})
  
  
# driver function 
if __name__ == '__main__': 
  
    app.run(debug = True) 
