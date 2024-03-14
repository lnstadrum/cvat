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
    context.logger.info("Init detector...")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
    model = weigths['model'].float()
    model.eval()
    if torch.cuda.is_available():
        model = model.half().to(device)
    context.user_data.model = model
    context.user_data.device = device

    context.user_data.sublabels = [
        "neck",
        "head",
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
    context.logger.info("Run yolov7 pose estimation...")
    model : torch.nn.Module = context.user_data.model
    device = context.user_data.device

    # load input image
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf).convert("RGB")
    image = torchvision.transforms.functional.pil_to_tensor(image).to(device).unsqueeze(0)

    # Preprocess the frame (you may need to customize this based on your model)
    image = image.to(torch.float32) / 255
    context.logger.info(f"Got an image: {image.shape}")
    h, w = image.shape[-2:]
    image = torch.nn.functional.pad(image, (0, 0, (w - h) // 2, (w - h) // 2))
    image = torch.nn.functional.interpolate(image, scale_factor=1280.0 / w)

    # Make predictions using the neural network
    with torch.no_grad():
        prediction, _ = model(image)
        prediction = non_max_suppression_kpt(prediction, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
        prediction = output_to_keypoint(prediction)

    results = []
    for instance in prediction:
        pose = instance[7:].reshape(-1, 3)

        # remove/merge unused points
        pose = numpy.concatenate((
            pose[0:1, :],
            numpy.mean(pose[1:3, :], axis=0, keepdims=True),
            pose[5:, :],
        ))

        skeleton = {
            "type": "skeleton",
            "label": "Player",
            "elements": [{
                "type": "points",
                "label": label,
                "outside": 0 if point[2] > 0 else 1,
                "points": [
                    point[0],
                    point[1] - (w - h) // 2
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
