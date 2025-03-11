from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
import markdown2

def basic_chatbot_na_view(request):
    return render(request, "chatbot/basic_chatbot_na.html")
@login_required
def basic_chatbot_view(request):
    return render(request, "chatbot/basic_chatbot.html")
@login_required
def romance_chatbot_view(request):
    return render(request, "chatbot/romance_chatbot.html")
@login_required
def rofan_chatbot_view(request):
    return render(request, "chatbot/rofan_chatbot.html")
@login_required
def fantasy_chatbot_view(request):
    return render(request, "chatbot/fantasy_chatbot.html")
@login_required
def historical_chatbot_view(request):
    return render(request, "chatbot/historical_chatbot.html")
