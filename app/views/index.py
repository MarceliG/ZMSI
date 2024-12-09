from django.http import HttpRequest, HttpResponse
from django.shortcuts import render


def index(request: HttpRequest) -> HttpResponse:
    """
    Show index page.

    Args:
        request: request: Request from django inlude data

    Returns:
        HttpResponse: rendered page
    """
    return render(request, "index.html")
