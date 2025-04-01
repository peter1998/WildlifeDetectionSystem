#!/bin/bash

# Setup script for Wildlife Detection System

# Exit on error
set -e

# Project root directory
ROOT_DIR="$HOME/Desktop/TU PHD/WildlifeDetectionSystem"

# Create Python virtual environment (if not already created)
if [ ! -d "$ROOT_DIR/api/venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$ROOT_DIR/api/venv"
fi

# Activate virtual environment
source "$ROOT_DIR/api/venv/bin/activate"

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r "$ROOT_DIR/api/requirements.txt"

# Create .env file if it doesn't exist
if [ ! -f "$ROOT_DIR/api/.env" ]; then
    echo "Creating .env file..."
    cat > "$ROOT_DIR/api/.env" << EOL
FLASK_APP=run.py
FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(16))')
EOL
fi

echo "Setup completed successfully!"
echo "To activate the virtual environment, run: source $ROOT_DIR/api/venv/bin/activate"
echo "To start the Flask server, run: cd $ROOT_DIR/api && flask run"
