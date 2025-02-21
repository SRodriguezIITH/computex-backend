# Ensure the parent directories are accessible
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
app = Flask(__name__)
# CORS(app, supports_credentials=True)
CORS(app, resources={r"/api/*": {"origins": "https://compute-x-tau.vercel.app"}})


import torch
from Preprocessing.Preprocessing import preprocess_image
from model.architecture import CNN
from PIL import Image
import io
import os
import cv2
import torch
import shutil
from Preprocessing.Preprocessing import preprocess_image_from_input
from model.architecture import CNN
from post_processing import format_equation, correct_equation
from compute_equation import compute_equation


class CustomDataset():
    def __init__(self, root_dir, transform=None):
        # print("\n\nInitializing Custom Dataset")
        self.root_dir = root_dir
        # print("Root Directory:", root_dir)
        self.image_paths = []
        
        class_folder = os.path.join(root_dir)
        # print("Class Folder:", class_folder)
        if os.path.isdir(class_folder):
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    # print("Image Path:", img_path)
                    if img_path.endswith(('.jpg', '.png', '.jpeg')):  # Ensure valid image formats
                        self.image_paths.append(img_path)
                        
                # print("Image Processed")

    def __len__(self):
        return len(self.image_paths)

    def getitem(self):
        """ Get an image and its corresponding label """
        images = []
        self.image_paths = sorted(self.image_paths, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        for i in self.image_paths:
            # print("Processing Image Path:", i)
            image = cv2.imread(i, cv2.IMREAD_GRAYSCALE) # Do not convert to RGB
            image = torch.tensor(image).unsqueeze(0)
            images.append(image)
            
        # print("\n\nImages: ", len(images))

        return images
num_classes = 157725
# Load PyTorch model
model = CNN(num_classes)
model.load_state_dict(torch.load(os.path.join("models", "model.pth")))
model.eval()

inverted_class_mapping = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "+",
    11: "-",
    12: "forward_slash",
    13: "(",
    14: ")",
    15: "div",
    16: "times",
    17: "x",
    18: "y",
}
from flask import send_file

@app.route("/g", methods=["POST"])
@cross_origin()
def process():
    file = request.files["file"]
    print("\n\nFile: ", file)
    if not file:
        response = jsonify({"error": "No file provided"})
        response.headers.add("Access-Control-Allow-Origin", "*")  # ✅ Fix CORS in response
        return response, 400
    
    image = Image.open(io.BytesIO(file.read())).convert("L")
    image_path = os.path.join(os.getcwd(), "test.png")
    image.save(image_path)

    if not image_path:
        return jsonify({"error": "Invalid file uploaded"}), 400

    preprocess_image_from_input(image_path)

    test_dataset = CustomDataset(root_dir=os.path.join(os.getcwd(), "cache/final_letters"))
    test_dataset_x = test_dataset.getitem()
    test_dataset_x = torch.stack(test_dataset_x).to(torch.float32)

    print("Prediction Commences")
    pred_str = ""

    try:
        prediction = model(test_dataset_x)
        result = torch.argmax(prediction, dim=1)
        
        pred = [inverted_class_mapping[p.item()] for p in result]
        print("\n\nPred: ", pred)
    except Exception as e:
        print("Error: Could not make a prediction.")

    print("Equation SOlving Begins")
    equation = format_equation(pred)
    equation = correct_equation(equation)

    if equation:
        print(f"Corrected Equation: {equation}")
        compute_equation(equation)
    else:
        print("Error: Could not identify a valid equation.")

    # Assuming `output.png` is created in the current working directory
    output_image_path = os.path.join(os.getcwd(), "output.png")

    cache_files = [
        os.path.join(os.getcwd(), 'cache/cca_output.png'),
        os.path.join(os.getcwd(), 'cache/cropped_image.png'),
        os.path.join(os.getcwd(), 'cache/deskewed_image.png'),
        os.path.join(os.getcwd(), 'cache/image.png'),
    ]

    for file in cache_files:
        if os.path.exists(file):
            os.remove(file)
            # print(f"Deleted: {file}")
        # else:
            # print(f"File not found: {file}")
    

    # Delete the final_letters folder
    cache_folder = os.path.join(os.getcwd(), 'cache/final_letters')

    if os.path.exists(cache_folder):
        shutil.rmtree(cache_folder)
        # print(f"Deleted folder: {cache_folder}")
    # else:
    #     print(f"Folder not found: {cache_folder}")

    print("\nCompleted Execution.")

    return jsonify({"equation": "success"}), 200
 
@app.route("/out", methods=["GET"])
@cross_origin()
def output():
    output_image_path = os.path.join(os.getcwd(), "output.png")
    return send_file(output_image_path, mimetype='image/png')


if __name__ == "__main__":
    app.run(debug=True)
