from django.http import HttpResponse
from django.shortcuts import render


def index(request) -> HttpResponse:
    """
    Show index page.

    Args:
        request: request: Request from django inlude data

    Returns:
        HttpResponse: rendered page
    """
    return render(request, "index.html")
