# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: urls.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-05-21 (YYYY-MM-DD)
-----------------------------------------------
"""
from django.urls import path
from . import views


urlpatterns = [
    path('', views.index, name='index')
]