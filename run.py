import pygame
import numpy as np
import pickle
import time
from PIL import Image
import cv2
import pickle
import math
import time
import argparse




""" Single Agent class, environment is defined inside it,
    it is used to define runner and the chasers. 
    Each agent has an x and y coordinates, used to calculate Q-values later on.
    Block configurations can either be given during instantiation,
    if not, blocks will have default values. They cannot take the initial positions of agents """


#size of the board, for this project, it's hard-coded
#SIZE_X = 8
#SIZE_Y = 8
class Agent:
  
  def __init__(self, SIZE_X, SIZE_Y, x, y, blocks=None):
    ## defining empty cells and blocks
    self.x = x
    self.y = y
    self.SIZE_X = SIZE_X
    self.SIZE_Y = SIZE_Y
    env = np.zeros(shape = [self.SIZE_X,self.SIZE_Y])
    if blocks == None:
      blocks = [[0,6],[3,5],[5,2]]
    
    

  def dist_x(self, other):
    return self.x-other.x

  def dist_y(self, other):
    return self.y-other.y

  #manhattan distance for penalties
  def dist(self, other):
    return(abs(self.x-other.x)+abs(self.y-other.y))

  #action  
  def action(self, choice):
      if choice == 0: #right
        self.move(x=1,y=0)
      elif choice == 1: #up
        self.move(x=0, y=1)
      elif choice == 2: #left
        self.move(x=-1, y=0)
      elif choice == 3: #down
        self.move(x=0, y=-1)
  
  def move(self, x=False, y=False):
    
    if not x:
      self.x += np.random.randint(-1, 2)
    else:
      self.x += x
    if not y:
      self.y += np.random.randint(-1,2)
    else:
      self.y += y
    
    #move
    
    new_x = self.x + x
    new_y = self.y + y

    #in case if they get out of table
    if self.x<0:
      self.x=0
    if self.x>=self.SIZE_X:
      self.x = self.SIZE_X-1
    if self.y<0:
      self.y=0
    if self.y>=self.SIZE_Y:
      self.y = self.SIZE_Y-1
    for i in blocks:
      if [self.x, self.y] == i:
        action = np.random.randint(0,4)
        self.action(action)
    #update positions



#Q-table
def Q_table(SIZE_X, SIZE_Y):

    q_table = {}
  
    for a in range(0, SIZE_X): #x coordinate of agent
        for b in range(0, SIZE_Y): #y coordinate of agent
            for c in range(-SIZE_X+1, SIZE_X): #distance between agents
              for d in range(-SIZE_Y+1, SIZE_Y): #distance between agents
                q_table[((a,b,c,d))]= [np.random.uniform(-4, 0) for i in range(5)] 
    
    print(f"q-table is {q_table.keys()}")
    return q_table



if __name__=="__main__":

  

  show = False
  parser = argparse.ArgumentParser()
  parser.add_argument("--runner", type=str, nargs = '?', default = "[0,0]", help = "Location of runner as a list")
  parser.add_argument("--chaser_2", type=str, nargs = '?', default = "[6,6]", help = "Location of chaser_2 as a list")
  parser.add_argument("--chaser_1", type=str, nargs = '?', default = "[6,5]", help = "Location of chaser_1 as a list")
  parser.add_argument("--blocks", type=str, nargs = '?', default = "[[0,6],[3,5],[5,2]]", help = "List of location of blocks")
  parser.add_argument("--SIZE_X", type=int, nargs = '?', default = 8, help = "Horizontal size")
  parser.add_argument("--SIZE_Y", type=int, nargs = '?', default = 8, help = "Vertical size")
  parser.add_argument("--exploitation_steps", type=int, nargs = '?', default = 150, help = "Exploitation steps")
  parser.add_argument("--exploration_steps", type=int, nargs = '?', default = 150, help = "Exploration steps")
  parser.add_argument("--episodes", type=int, nargs = '?', default = 100, help = "Episodes")
  parser.add_argument("--show_ep", type=int, nargs = '?', default = 10, help = "Show every N episodes")
  parser.add_argument("--learning_rate", type=float, nargs = '?', default = 0.1, help = "Learning rate")
  parser.add_argument("--gamma", type=float, nargs = '?', default = 0.1, help = "Discount factor for future rewards")
  args = parser.parse_args()

  exploration_steps = args.exploration_steps
  exploitation_steps = args.exploitation_steps
  show_ep = args.show_ep
  episodes = args.episodes
  learning_rate = args.learning_rate
  gamma = args.gamma
  SIZE_X = args.SIZE_X
  SIZE_Y = args.SIZE_Y
  runner_loc = eval(args.runner)
  chaser_1_loc = eval(args.chaser_1)
  chaser_2_loc = eval(args.chaser_2)
  blocks = eval(args.blocks)

  rounds = exploration_steps + exploitation_steps

  
  
  # RGB color coding
  d = {"runner_color":(0, 255, 0), "chaser1_color":(255,180, 20), "chaser2_color":(255,20,147), "block_color":(255, 255, 208)}


  
  chasers_win = 0
  for eps in range(episodes):
    
    if(eps%show_ep==0):
      show = True

    # initialize agents back to their original positions by the 
    # beginning of every game
    chaser1 = Agent(SIZE_X=SIZE_X, SIZE_Y=SIZE_Y, x = chaser_1_loc[0], y = chaser_1_loc[1])
    chaser2 = Agent(SIZE_X=SIZE_X, SIZE_Y=SIZE_Y, x = chaser_2_loc[0], y = chaser_2_loc[1])
    runner= Agent(SIZE_X=SIZE_X, SIZE_Y=SIZE_Y, x = runner_loc[0], y = runner_loc[1])
    
    # initialize Q_tables before training
    q_table_c1 = Q_table(SIZE_X=SIZE_X, SIZE_Y=SIZE_Y)
    q_table_c2 = Q_table(SIZE_X=SIZE_X, SIZE_Y=SIZE_Y)
    q_table_r = Q_table(SIZE_X=SIZE_X, SIZE_Y=SIZE_Y)
    
    for i in range(rounds):
      
      # states are (x, y, distance to the other agent)
      
      dstate_r = (runner.x, runner.y, min(runner.dist_x(chaser1),runner.dist_x(chaser2)), min(runner.dist_y(chaser1),runner.dist_y(chaser2)))

      dstate_c1 = (chaser1.x, chaser1.y, chaser1.dist_x(runner), chaser1.dist_y(runner))

      dstate_c2 = (chaser2.x, chaser2.y, chaser2.dist_x(runner), chaser2.dist_y(runner))
      
      #first action is a random one
      
      if i<exploration_steps: #for first ten turns explore

        action_c1 = np.random.randint(0,4)
        action_c2 = np.random.randint(0,4)
        action_r = np.random.randint(0,4)

      else: #for the rest of the episodes take the action that has
          #the highest q-value for that state

        action_c1 = np.argmax(dstate_c1)
        action_c2 = np.argmax(dstate_c2)
        action_r = np.argmax(dstate_r)


      #defining rewards
      chaser_reward_1 = 1
      chaser_reward_2 = 2
      catch_reward = 3
      runner_reward = 2
      

      #taking actions 

      runner.action(action_r)
      chaser1.action(action_c1)
      chaser2.action(action_c2)
      
      #rewards of each agent
      reward_1 = 0
      reward_2 = 0
      reward_r = 0

      #reward conditions
      #chasers get reward when they get close to runner

      if runner.dist_x(chaser1)==2 or runner.dist_x(chaser2)==2:
        reward_1 = chaser_reward_1

      elif runner.dist(chaser1)==1 or runner.dist(chaser2)==1:
        reward_2 = chaser_reward_2
      #runner gets reward when it is far from chasers
      elif runner.dist(chaser1)>4 or runner.dist(chaser2)>4:
        reward_r = runner_reward

      #both of the chasers get reward 
      elif (runner.x==chaser1.x and runner.y==chaser1.y):
        print("Game is over, runner is caught")
        reward_1 = catch_reward
        reward_r = -catch_reward
        chasers_win += 1
        break
      elif(runner.x==chaser2.x and runner.y==chaser2.y):
        print("Game is over, runner is caught")
        reward_2 = catch_reward
        reward_r = -catch_reward
        chasers_win += 1
        break
      
      #state updates  
      new_dstate_c2 = ( chaser2.x, chaser2.y, chaser2.dist_x(runner), chaser2.dist_y(runner) )
      new_dstate_c1 = ( chaser1.x, chaser1.y, chaser1.dist_x(runner), chaser1.dist_y(runner) )
      new_dstate_r = ( runner.x, runner.y, min(runner.dist_x(chaser1), runner.dist_x(chaser2)), min(runner.dist_y(chaser1), runner.dist_y(chaser2)))

      # calculating cumulated future reward
      future_qval_c1 = np.max(q_table_c1[new_dstate_c1])

      future_qval_c2 = np.max(q_table_c2[new_dstate_c2])

      future_qval_r = np.max(q_table_r[new_dstate_r])

      #retrieve q-values for each action
      current_qval_c1 = q_table_c1[dstate_c1][action_c1]
      current_qval_c2 = q_table_c2[dstate_c2][action_c2]
      current_qval_r = q_table_r[dstate_r][action_r]

      #calculate q-values
      new_qval_c1 = (1 - learning_rate) * current_qval_c1 + learning_rate * (reward_1 + gamma * future_qval_c1)
      new_qval_c2 = (1 - learning_rate) * current_qval_c2 + learning_rate * (reward_2 + gamma * future_qval_c2)
      new_qval_r = (1 - learning_rate) * current_qval_r + learning_rate * (reward_r + gamma * future_qval_r)


      #update q-table
      
      q_table_c1[dstate_c1][action_c1] = new_qval_c1
      q_table_c2[dstate_c2][action_c2] = new_qval_c2
      q_table_r[dstate_r][action_r] = new_qval_r
      
      #interface
    
      if(show):
        env = np.zeros((args.SIZE_X, args.SIZE_Y, 3), dtype=np.uint8)
        env[runner.x][runner.y] = d["runner_color"]
        env[chaser1.x][chaser1.y] = d["chaser1_color"]
        env[chaser2.x][chaser2.y] = d["chaser2_color"]

        for i in blocks:
          env[i[0]][i[1]] = d["block_color"]
          
        image = Image.fromarray(env, 'RGB')
        image = image.resize((1300, 800), resample=Image.NEAREST)

        
        cv2.imshow("ENV", np.array(image))


        # if the runner is caught
        if reward_1 == catch_reward or reward_2 == catch_reward:
          if cv2.waitKey(50000) and 0xFF == ord('q'):
            break
        else:
          if cv2.waitKey(1) & 0xFF == ord('q'):
            break

  print(f"Chasers win: {chasers_win}")
            