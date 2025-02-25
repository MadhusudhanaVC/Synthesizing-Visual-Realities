import os
import io
import warnings
from PIL import Image
from flask import Flask, render_template, request, send_file
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

app = Flask(__name__)

# Initialize Stability API client
os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
os.environ['STABILITY_KEY'] = 'sk-FuMafgpS1jreBCOsvOSiqPhOep396Fd2KwH1s91PE8rIF0X4'
stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'],
    verbose=True,
    engine="stable-diffusion-xl-1024-v1-0",
)

@app.route('/')
def index():
    return render_template('indexx.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']

    answers = stability_api.generate(
        prompt=prompt,
        seed=1,
        steps=50,
        cfg_scale=8.0,
        width=1024,
        height=1024,
        samples=1,
        sampler=generation.SAMPLER_K_DPMPP_2M
    )

    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn("Your request activated the API's safety filters and could not be processed. Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                return send_file(img_bytes, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
