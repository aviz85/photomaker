from flask import Flask, render_template, request, jsonify
import replicate
import os
import base64
import io
import time
import random

app = Flask(__name__)

# Get the Replicate API token from environment variable
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise ValueError("No API token found. Please set the REPLICATE_API_TOKEN environment variable.")

# List of predefined prompts
PROMPTS = [
    "A whimsical cartoon character img with exaggerated features",
    "A heroic superhero img with a flowing cape",
    "A mysterious cyberpunk character img with glowing neon accents",
    "A serene nature spirit img surrounded by floating leaves and flowers",
    "A steampunk inventor img with intricate mechanical parts",
    "An elegant Victorian-era noble img in ornate clothing",
    "A futuristic astronaut img exploring an alien planet",
    "A mischievous fairy img with sparkling wings",
    "A wise old wizard img with a long beard and magical staff",
    "A fierce viking warrior img with intricate armor",
    "A graceful ballerina img mid-pirouette",
    "A rugged cowboy img in the Wild West",
    "An enchanted mermaid img swimming in a coral reef",
    "A stoic samurai img with traditional armor and katana",
    "A jolly Santa Claus img preparing gifts",
    "A mysterious film noir detective img in a shadowy alley",
    "A vibrant Bollywood dancer img in elaborate costume",
    "A determined Olympic athlete img crossing the finish line",
    "A charming 1950s pin-up model img",
    "A wise Native American chief img in traditional headdress",
    "A daring trapeze artist img mid-performance",
    "A regal Egyptian pharaoh img with golden accessories",
    "A groovy 1970s disco dancer img",
    "A menacing pirate captain img on a ship's deck",
    "A serene Buddhist monk img in meditation",
    "A fierce Amazonian warrior img in the jungle",
    "A dapper 1920s gangster img in a pinstripe suit",
    "A mystical fortune teller img with crystal ball",
    "A brave firefighter img rescuing a cat",
    "A cunning spy img in a sleek tuxedo",
    "A jolly medieval court jester img entertaining royalty",
    "A graceful geisha img in traditional kimono",
    "A determined mountain climber img reaching the summit",
    "A chic fashion designer img at a runway show",
    "A studious alchemist img in a cluttered laboratory",
    "A valiant knight img in shining armor",
    "A lively mariachi player img serenading",
    "A focused sushi chef img preparing delicate dishes",
    "A charismatic ringmaster img at a circus",
    "A peaceful Zen gardener img raking sand",
    "A daring stunt driver img mid-jump",
    "A passionate flamenco dancer img in mid-twirl",
    "A diligent watchmaker img with magnifying glass",
    "A spirited cheerleader img in mid-jump",
    "A focused archer img aiming at a distant target",
    "A carefree surfer img riding a giant wave",
    "A mysterious masked masquerade attendee img",
    "A dedicated paleontologist img excavating dinosaur bones",
    "A nimble parkour athlete img leaping between buildings",
    "A serene yoga instructor img in a complex pose",
    "A dazzling figure skater img performing a spin",
    "A rugged lumberjack img chopping wood",
    "A graceful tightrope walker img balancing high above",
    "A focused golfer img mid-swing",
    "A mystical druid img communing with nature"
]

STYLES = [
    "(No style)", "Cinematic", "Disney Charactor", "Digital Art",
    "Photographic (Default)", "Fantasy art", "Neonpunk", "Enhance",
    "Comic book", "Lowpoly", "Line art"
]


STYLES = [
    "(No style)", "Cinematic", "Disney Charactor", "Digital Art",
    "Photographic (Default)", "Fantasy art", "Neonpunk", "Enhance",
    "Comic book", "Lowpoly", "Line art"
]

@app.route('/')
def index():
    return render_template('index.html', prompts=PROMPTS, styles=STYLES)

@app.route('/generate', methods=['POST'])
def generate():
    image_data = request.json['image']
    prompt_index = int(request.json['promptIndex'])
    style_index = int(request.json.get('styleIndex', 4))  # Default to "Photographic (Default)"
    
    # Decode the base64 image
    image_data = base64.b64decode(image_data.split(',')[1])
    image = io.BytesIO(image_data)
    image.name = 'image.jpg'

    try:
        # Prepare input for the API
        input_data = {
            "input_image": image,
            "prompt": PROMPTS[prompt_index],
            "style_name": STYLES[style_index],
            "num_steps": 50,
            "num_outputs": 1,
            "guidance_scale": 5,
            "negative_prompt": "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
            "style_strength_ratio": 20,
            "seed": random.randint(0, 2147483647),
            "disable_safety_checker": False
        }

        # Create a prediction
        prediction = replicate.predictions.create(
            version="ddfc2b08d209f9fa8c1eca692712918bd449f695dabb4a958da31802a9570fe4",
            input=input_data
        )

        # Wait for the prediction to complete
        while prediction.status != "succeeded" and prediction.status != "failed":
            prediction.reload()
            time.sleep(1)

        if prediction.status == "succeeded":
            # The output is directly the list of image URLs
            return jsonify({"image_urls": prediction.output})
        else:
            return jsonify({"error": "Prediction failed"}), 400

    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)