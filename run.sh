# run slow_ai
cd code

# make the downloads folder if it doesn't already exist
mkdir -p images

# Check if the virtual environment directory exists
if [ ! -d "venv" ]; then
    # Start the venv only if it doesn't exist
    # pull in existing python packages
    # this is needed on the pi for picamera2 to work
    python3 -m venv --system-site-packages venv
fi

# activate the virtual environment
source venv/bin/activate

# install / check requirements
pip install -r requirements.txt

# Run your Python script
python trailcam.py

# Deactivate the virtual environment
deactivate