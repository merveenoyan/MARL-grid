To run the project, run:
- conda create --name env python=3.7
- pip install -r requirements.txt
- python run.py

Couple of arguments you can pass:
- "--runner": Location of runner as a list (str(List)) 
- "--chaser_2": Location of chaser 2 as a list (str(List))
- "--chaser_1": Location of chaser_1 as a list (str(List))
- "--blocks": List of location of blocks
- "--SIZE_X": Horizontal size (int)
- --SIZE_Y": Vertical size (int)
- "--exploitation_steps": Exploitation steps (int)
- "--exploration_steps: Exploration steps (int)
- "--episodes": Episodes (int)
- "--show_ep": Show every N episodes (int)
- "--learning_rate": Learning rate (float)
- "--gamma": "Discount factor for future rewards" (float)
 
To-do:
- Add direction to the state space (DONE)
- Take user parameters through GUI (DONE)
- Write environment as a separate class
- Take second best action when agents try to go over blocks or get out of the board


