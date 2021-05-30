import pygame
import numpy as np
import pickle
import time
from PIL import Image
import cv2
import pickle
import math
import time


""" Single Agent class, environment is defined inside it,
    it is used to define runner and the chasers. 
    Each agent has an x and y coordinates, used to calculate Q-values later on.
    Block configurations can either be given during instantiation,
    if not, blocks will have default values. They cannot take the initial positions of agents """


#size of the board, for this project, it's hard-coded
SIZE=8
class Agent:
  
  def __init__(self, x, y, blocks=None):
    ## defining empty cells and blocks
    env = np.zeros(shape = [SIZE,SIZE])
    if blocks == None:
      blocks = [[0,6],[3,5],[5,2]]
    
    self.x = x
    self.y = y

  #manhattan distance
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
    if self.x>=SIZE:
      self.x = SIZE-1
    if self.y<0:
      self.y=0
    if self.y>=SIZE:
      self.y = SIZE-1
    for i in blocks:
      if [self.x, self.y] == i:
        action = np.random.randint(0,4)
        self.action(action)
    #update positions

  

# RGB color coding
d = {"runner_color":(0, 255, 0), "chaser1_color":(255,180, 20), "chaser2_color":(255,20,147), "block_color":(255, 255, 208)}

#params
episodes = 100
show_ep = 10
learning_rate = 0.1
gamma = 0.1


#Q-table
def Q_table():

    q_table = {}

    for a in range(0, SIZE): #x coordinate of agent
        for b in range(0, SIZE): #y coordinate of agent
            for c in range(0, SIZE+SIZE): #distance between agents
                q_table[((a,b,c))]= [np.random.uniform(-4, 0) for i in range(4)] 
    print(f"q-table is {q_table}")
    return q_table





if __name__=="__main__":
  
  blocks = [[0,6],[3,5],[5,2]]
  show = False


  # initialize Q_tables before training
  q_table_c1 = Q_table()
  q_table_c2 = Q_table()
  q_table_r = Q_table()

  
  chasers_win = 0
  for eps in range(episodes):
    
    if(eps%show_ep==0):
      show = True

    # initialize agents back to their original positions by the 
    # beginning of every game
    chaser1 = Agent(7,6)
    chaser2 = Agent(7,7)
    runner= Agent(0,0)
    
    
    for i in range(100):
      
      # states are (x, y, distance to the other agent)

      dstate_r = (runner.x, runner.y, min(runner.dist(chaser1),runner.dist(chaser2)))

      dstate_c1 = (chaser1.x, chaser1.y, runner.dist(chaser1))

      dstate_c2 = (chaser2.x, chaser2.y, runner.dist(chaser2))
      
      #first action is a random one
      
      if i<11: #for first ten turns explore
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

      if runner.dist(chaser1)==2 or runner.dist(chaser2)==2:
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
      new_dstate_c2 = ( chaser2.x, chaser2.y, runner.dist(chaser2) )
      new_dstate_c1 = ( chaser1.x, chaser1.y, runner.dist(chaser1) )
      new_dstate_r = ( runner.x, runner.y, min(chaser1.dist(runner), chaser2.dist(runner)))
      
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
        env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8) # 3 is the number of channels for RGB image
        env[runner.x][runner.y] = d["runner_color"]
        env[chaser1.x][chaser1.y] = d["chaser1_color"]
        env[chaser2.x][chaser2.y] = d["chaser2_color"]

        for i in blocks:
          env[i[0]][i[1]] = d["block_color"]
          
        image = Image.fromarray(env, 'RGB')
        image = image.resize((500, 500), resample=Image.NEAREST)

        
        cv2.imshow("ENV", np.array(image))


        # if the runner is caught
        if reward_1 == catch_reward or reward_2 == catch_reward:
          if cv2.waitKey(50000) and 0xFF == ord('q'):
            break
        else:
          if cv2.waitKey(1) & 0xFF == ord('q'):
            break

  print(f"Chasers win: {chasers_win}")
            