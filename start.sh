#!/bin/bash

# Ensure setuptools and wheel are installed
pip install setuptools wheel

# Start the application with gunicorn
gunicorn app:app --bind 0.0.0.0:10000