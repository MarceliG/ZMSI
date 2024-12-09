import os
import sys

from django.http import JsonResponse
from django.template.loader import render_to_string

# Main path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.append(BASE_DIR)

# Path to model
MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "model_2_classes")
sys.path.append(MODEL_PATH)

from src import predict_class


def add_comment(request) -> JsonResponse:
    """
    Add a comment.

    Args:
        request: Request from django inlude data like text etc.

    Returns:
        JsonResponse: Json object include status: ok/error, message: string, result: string
    """
    response = {"status": "error", "message": "Pole komentarz nie może być puste", "result": None}

    text = request.POST.get("text", "")

    if not text:
        return JsonResponse(response)
    rate_comment = predict_class(text=text, model_path=MODEL_PATH)

    response["status"] = "ok"
    response["message"] = None
    response["result"] = render_view(rate_comment, text)
    return JsonResponse(response)


def render_view(rate_comment: int, text: str) -> str:
    """
    Render string view.

    Args:
        rate_comment (int): Rate 0/1 from AI
        text (str): User text

    Returns:
        str: rendered comment box
    """
    return render_to_string("single_comment.html", {"rate_comment": rate_comment, "text": text})
