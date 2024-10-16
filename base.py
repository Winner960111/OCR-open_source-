import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        # Read the image file and encode it to base64
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# Usage
image_path = "PP-AE-2.jpg"  # Replace with your image file path
base64_code = image_to_base64(image_path)

with open("base64_code.txt", "w") as file:
    file.write(base64_code)

