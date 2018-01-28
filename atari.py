import math
import tensorflow as tf
import numpy as np
import gym
import random
from helper import HelperClass
from collections import deque
from PIL import Image
class breakOut:

	def __init__(self):
				
		self.input_image=tf.placeholder('float32',shape=(None,85,80,4))
		self.action_mask=tf.placeholder('float32',shape=(None,4))
		self.label=tf.placeholder('float32',shape=(None,4))
		self.predictions=tf.placeholder('float32')		
		self.epsilon=1.0
		self.height=85
		self.width=80
		self.num_episodes=60000
		self.preFillLimit=32
		self.env=gym.make('BreakoutDeterministic-v4')
		self.memory=deque(maxlen=600000)
		self.stateQ=deque(maxlen=4)
		self.update_frequency=4
		self.learning_rate=0.00025
		self.gamma=0.99
		self.model,self.optimizer,self.loss=self.build_model()
		self.sess=tf.Session()
		self.sess.run(tf.global_variables_initializer())
		self.epsilon_decay_steps=1500000
		self.max_no_op=30
		
	
	#--------------Defining the main network------------------------#
	def build_model(self):	

		conv1 = tf.layers.conv2d(inputs=self.input_image,filters=16,kernel_size=[8,8],strides=(4,4),activation=tf.nn.relu,use_bias=True,
			kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),bias_initializer=tf.zeros_initializer())

		conv2 = tf.layers.conv2d(inputs=conv1,filters=32,kernel_size=[4,4],strides=(2,2),activation=tf.nn.relu,use_bias=True,
			kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),bias_initializer=tf.zeros_initializer())
		
		conv2_trans=tf.transpose(conv2,perm=[0,3,1,2])
		conv2_flat=tf.reshape(conv2_trans,[-1,2304])

		dense1=tf.layers.dense(inputs=conv2_flat, units=256, activation=tf.nn.relu,use_bias=True,
		kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),bias_initializer=tf.zeros_initializer())
		
		output=tf.layers.dense(inputs=dense1, units=4,activation=None,use_bias=True,
		kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),bias_initializer=tf.zeros_initializer())

		output_mask=tf.multiply(output,self.action_mask)
		
		loss=tf.losses.huber_loss(labels=self.label,predictions=output_mask)
	
		opt=tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,momentum=0.95,epsilon=0.01).minimize(loss)
	
		return output_mask,opt,loss
	
	#----- taking an epsilon greedy action -------#
	def chooseAction(self,state):

		prob=np.random.uniform(low=0,high=1)
		
		if prob < self.epsilon:
			return self.env.action_space.sample()
		else:
			
			q_value=self.sess.run(self.model,feed_dict={self.input_image:state/255.0,self.action_mask:np.ones((1,4))})
			return np.argmax(q_value[0,:])

	#---------------Taking a mini batch from memory and running the n/w on it-------------------#
	def replay(self):
		
		with tf.device('/gpu:0'):
			current_array,action_array,reward_array,next_array,done_array=helperObj.createBatch(breakObj.memory)
		
			next_qvalues=breakObj.sess.run(breakObj.model,feed_dict={breakObj.input_image:next_array,breakObj.action_mask:np.ones(action_array.shape)})
			
			next_qvalues[done_array]=0
							
			target=reward_array+breakObj.gamma*np.max(next_qvalues,axis=1)

			breakObj.sess.run([breakObj.optimizer],feed_dict={breakObj.label:action_array*target[:,None],breakObj.input_image:current_array,breakObj.action_mask:action_array})			
		
		
if __name__ == "__main__":
	
	i=0
	steps=0
	count=0

	breakObj=breakOut()
	helperObj=HelperClass()
	dummy=np.zeros((1,breakObj.height,breakObj.width,4),dtype='uint8')
	start=breakObj.env.reset()	
	epsilons=np.linspace(1.0,0.05,breakObj.epsilon_decay_steps)

	action_one=np.zeros(4,dtype='uint8')

	while i <breakObj.num_episodes:
		
		end_life = False
		total_reward=0
		total_loss=0
		previous_life=5
		done=False

		s=breakObj.env.reset()		
		for re in range(np.random.randint(low=4,high=breakObj.max_no_op)):#Do nothing at the start of the episode
		
			obs,reward,done,_=breakObj.env.step(0)
			breakObj.stateQ.append(helperObj.preProcessing(obs))		
		
		current_state=helperObj.generateState(breakObj.stateQ)		

		while not done:
			
			dummy[0]=current_state
			action=breakObj.chooseAction(dummy)
			next_frame,reward,done,lives=breakObj.env.step(action)
			total_reward+=reward

			lost_life=previous_life-lives['ale.lives']

			if lost_life==1 or done:#penalizing the agent if life is lost
				end_life=True
				reward=-1
						
			breakObj.stateQ.append(helperObj.preProcessing(next_frame))		
			
			next_state=helperObj.generateState(breakObj.stateQ)		
			action_one=np.eye(4)[action]#converting to one hot representation	
			
			breakObj.memory.append((current_state,action_one,reward,next_state,end_life))
			if steps>=breakObj.preFillLimit:
				if count==breakObj.update_frequency:#----------Pre filling the memory with some experiance by a random agent---------#
					count=0
					breakObj.replay()
				count+=1
			else:			
				count=0	
				i=0
		
			current_state=next_state
			previous_life=lives['ale.lives']
			
			steps+=1			
			end_life=False
		if steps>=breakObj.preFillLimit:
			print("Episode:=",i," reward=",total_reward," epsilon=",breakObj.epsilon)
			breakObj.epsilon=epsilons[min(steps,breakObj.epsilon_decay_steps-1)]#Linearly decaying epsilon
		i=i+1	
