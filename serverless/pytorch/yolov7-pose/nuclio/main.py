import base64
import io
import json
import numpy
import torch
import torchvision
from PIL import Image

from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint


def init_context(context):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    context.logger.info(f"Preparing model for device {device}")

    weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
    model = weigths['model'].float()
    model.eval()
    if device.type == "cuda":
        model = model.half().to(device)
    context.user_data.model = model
    context.user_data.device = device

    context.user_data.sublabels = [
        "nose",
        "L shoulder",
        "R shoulder",
        "L elbow",
        "R elbow",
        "L wrist",
        "R wrist",
        "L hip",
        "R hip",
        "L knee",
        "R knee",
        "L ankle",
        "R ankle"
    ]

    context.logger.info("Function initialized")


def handler(context, event):
    model : torch.nn.Module = context.user_data.model
    device = context.user_data.device

    # load input image
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf).convert("RGB")
    image = torchvision.transforms.functional.pil_to_tensor(image).to(device).unsqueeze(0)

    # Preprocess the frame (you may need to customize this based on your model)
    image = image.to(torch.float16 if device.type == "cuda" else torch.float32) / 255
    context.logger.info(f"Got an image: {image.shape}")
    h, w = image.shape[-2:]
    image = torch.nn.functional.pad(image, (0, 0, (w - h) // 2, (w - h) // 2))
    image = torch.nn.functional.interpolate(image, scale_factor=1280.0 / w, mode="bilinear")

    # Make predictions using the neural network
    with torch.no_grad():
        prediction, _ = model(image)
        prediction = non_max_suppression_kpt(prediction, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
        prediction = output_to_keypoint(prediction)

    if len(prediction.shape) != 2:
        prediction = numpy.empty((0, 58))

    results = []
    for instance in prediction:
        pose = instance[7:].reshape(-1, 3)

        # drop unused points
        pose = numpy.concatenate((pose[0:1, :], pose[5:, :]))

        skeleton = {
            "type": "skeleton",
            "label": "Player",
            "elements": [{
                "type": "points",
                "label": label,
                "outside": bool(point[2] < 0.25),
                "occluded": bool(point[2] < 0.5),
                "points": [
                    point[0],
                    point[1] - (w - h) // 2 * (1280.0 / w)
                ],
                "confidence": str(point[2]),
            } for label, point in zip(context.user_data.sublabels, pose)],
        }

        if not all([element['outside'] for element in skeleton["elements"]]):
            results.append(skeleton)

    return context.Response(
        body=json.dumps(results),
        headers={},
        content_type="application/json",
        status_code=200
    )
