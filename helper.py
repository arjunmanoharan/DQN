import cv2
from collections import deque
import numpy as np
import random

class HelperClass:

	def __init__(self):
		self.height=85
		self.width=80
		self.batch_size=32


	#------------------------Generates the state to be processed----------------------#
	def generateState(self,stateQ):

		current_state=np.zeros((self.height,self.width,4),dtype='uint8')		
		reward=0
		for i in range(4):
			current_state[:,:,i]=stateQ[i]		

		return current_state


	#---------------------Reducing the size of the screen to 85X80 and onverting to grey scale--------------#
	def preProcessing(self,frame):
		
		x=(0.2126*frame[:,:,0]) + (0.7152*frame[:,:,1]) + (0.0722*frame[:,:,2])
		return x[::2, ::2][15:100,:]



	#--------creating a mini batch from the replay memory----------------#

	def createBatch(self,memory):
		count=0

		current_array=np.zeros((self.batch_size,self.height,self.width,4),dtype='float32')
		next_array=np.zeros((self.batch_size,self.height,self.width,4),dtype='float32')
		reward_array=np.zeros(self.batch_size)
		done_array=np.zeros(self.batch_size,dtype='bool')
		action_array=np.zeros((self.batch_size,4),dtype='uint8')

		mini_batch=random.sample(memory,self.batch_size)

		for current_state,action,reward,next_state,done in mini_batch:
			current_array[count]=current_state/255.0
			next_array[count]=next_state/255.0
			reward_array[count]=reward
			action_array[count]=action
			done_array[count]=done
			count+=1
		
		return current_array,action_array,reward_array,next_array,done_array