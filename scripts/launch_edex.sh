#!/bin/bash
# Launches the MOTHER AI inside eDEX-UI for a sci-fi terminal experience

# 1. Start eDEX-UI (from local clone)
(cd edex-ui && npm start) &

# 2. Wait a few seconds for eDEX to initialize
sleep 5

# 3. Run AI assistant in the same terminal
python main.py


