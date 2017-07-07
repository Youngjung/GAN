from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *
from data_loader import create_dataloader

import pdb

def GRUcell( num_units, isTrain=False, dropout_keep_ratio=0.5 ):
	gru_cell = tf.contrib.rnn.GRUCell( num_units )
	if isTrain:
		gru_cell = tf.contrib.rnn.DropoutWrapper( gru_cell, 
													input_keep_prob=dropout_keep_ratio,
													output_keep_prob=dropout_keep_ratio )
	return gru_cell

# network for sequence generation
# ref: ptb_word_lm.py from tutorial rnn, line 104-200
def generator(z, vocab_size, opts, GTtext=None, reuse=False):
	with tf.variable_scope("generator") as scope:
		if reuse: scope.reuse_variables()

		# embedding map for GTtext embedding in teacher-forcing mode
		with tf.variable_scope("embedding") as scope_embedding:
			embedding_map = tf.get_variable( "map", [vocab_size,opts.gru_num_units] )

		# make 2-layer gru
		stacked_gru = tf.contrib.rnn.MultiRNNCell(
					[GRUcell(opts.gru_num_units,opts.phase=='train') for _ in range(opts.gru_num_layers)] )
		# feed z
		z_fc = linear( z, opts.gru_num_units, scope_name='z_fc', reuse=reuse )
		zero_state = stacked_gru.zero_state( batch_size=opts.batch_size, dtype=tf.float32 )
		cell_output, initial_state = stacked_gru( z_fc, zero_state )

		# manual unrolling is used because static_rnn and dynamic_rnn return the final state only
		scope.reuse_variables()
		state = initial_state
		gru_outputs = []
		gru_states = []
		if GTtext is not None: # for teacher-forcing mode
			GT_embedded = tf.nn.embedding_lookup( embedding_map, GTtext )
			GTs = tf.unstack( GT_embedded, opts.sequence_length, axis=1 )
			for time_step in range(opts.sequence_length):
				gru_output, state = stacked_gru( GTs[time_step], state )
				gru_outputs.append( cell_output )
				gru_states.append( tf.concat(state,axis=1) )
		else: # for free-running mode
			for time_step in range(opts.sequence_length):
				cell_output, state = stacked_gru( cell_output, state )
				gru_outputs.append( cell_output )
				gru_states.append( tf.concat(state,axis=1) )
			
		gru_outputs = tf.stack(gru_outputs, axis=1)
		gru_states = tf.stack(gru_states, axis=1)

		behavior = [gru_outputs, gru_states] 

		return behavior

def NLLlossForBehavior( behavior, GTtext, vocab_size ):
	gru_outputs = behavior[0]
	batch_size, sequence_length, gru_size = gru_outputs.get_shape().as_list()
	logits = linear( tf.reshape( gru_outputs,[-1,gru_size] ), vocab_size, scope_name='NLLlogits' )
	logits = tf.reshape( logits,[batch_size, sequence_length, vocab_size] )

	# average_across_timesteps might have to be set False
	return tf.contrib.seq2seq.sequence_loss( logits, GTtext, tf.ones([batch_size, sequence_length]) )
#	return tf.reduce_sum(
#			tf.nn.sparse_softmax_cross_entropy_with_logits(labels=GTtext, logits=logits) )
	

# network for sequence classification
# ref: https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/
def discriminator(behavior, opts, reuse=False):
	with tf.variable_scope("discriminator") as scope:
		if reuse:
			scope.reuse_variables()

		# make 2-layer gru
		stacked_gru = tf.contrib.rnn.MultiRNNCell(
						[GRUcell(opts.gru_num_units*3,True) for _ in range(opts.gru_num_layers)])

		# use dynamic_rnn
		behaviors = tf.concat(behavior, axis=2)
		zero_state = stacked_gru.zero_state( batch_size=opts.batch_size, dtype=tf.float32 )
		batch_size = behavior[0].get_shape().as_list()[0]
		lengths = tf.tile( [opts.sequence_length], [batch_size] )
		gru_outputs, _ = tf.nn.dynamic_rnn( cell=stacked_gru,
											inputs=behaviors,
											sequence_length=lengths,
											initial_state=zero_state,
											dtype=tf.float32,
											scope=scope )

		# get last output
		gru_outputs = tf.transpose( gru_outputs, [1,0,2] )
		gru_last_output = tf.gather( gru_outputs, int(gru_outputs.get_shape()[0])-1 )
		
		# fully connected
		logits = linear( gru_last_output, 2, scope_name='logits')

		return tf.nn.sigmoid(logits), logits
	
class TextGAN(object):
	def __init__(self, sess, opts):
		"""
		Args:
			sess: TensorFlow session
			batch_size: The size of batch. Should be specified before training.
			z_dim: (optional) Dimension of dim for Z. [100]
		"""
		self.sess = sess
		self.opts = opts

		self.data_loader = create_dataloader(opts)
		self.build_model()

	def build_model(self):
		opts = self.opts
		vocab_size = self.data_loader.vocab_size

		# t_z = [batch_size X z_dim], latent noise
		# t_realText = [batch_size X sequence_length], each row has word_ids
		t_z = tf.placeholder(tf.float32, [opts.batch_size, opts.z_dim], name='z')
		t_realText = tf.placeholder(tf.int32, [opts.batch_size, opts.sequence_length], name='real_text')
		
		t_freeBehavior = generator( t_z, vocab_size, opts )
		t_teacherBehavior = generator( t_z, vocab_size, opts, t_realText, reuse=True )

		t_D_free, t_D_free_logits = discriminator( t_freeBehavior, opts )
		t_D_teacher, t_D_teacher_logits = discriminator( t_teacherBehavior, opts, reuse=True)

		t_freeText = tf.argmax( tf.nn.softmax( t_freeBehavior[0] ), axis=2 )

		loss_NLL = NLLlossForBehavior( t_teacherBehavior, t_realText, vocab_size )
		loss_fool = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
											logits=t_D_free_logits, labels=tf.ones_like(t_D_free)))
		g_loss = loss_NLL + loss_fool

		d_loss_free = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
											logits=t_D_free_logits, labels=tf.zeros_like(t_D_free)))
		d_loss_teacher = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
											logits=t_D_teacher_logits, labels=tf.ones_like(t_D_teacher)))
		d_loss = d_loss_free + d_loss_teacher

		summary = {}
		summary['g_loss'] = tf.summary.scalar("g_loss", g_loss)
		summary['g_loss_NLL'] = tf.summary.scalar("g_loss_NLL", loss_NLL)
		summary['g_loss_fool'] = tf.summary.scalar("g_loss_fool", loss_fool)
		summary['d_loss'] = tf.summary.scalar("d_loss", d_loss)
		summary['d_loss_free'] = tf.summary.scalar("d_loss_free", d_loss_free)
		summary['d_loss_teacher'] = tf.summary.scalar("d_loss_teacher", d_loss_teacher)

		self.saver = tf.train.Saver()

		t_vars = tf.trainable_variables()
		g_vars = [var for var in t_vars if 'generator' in var.name]
		d_vars = [var for var in t_vars if 'discriminator' in var.name]

		self.input_tensors = { 't_z' : t_z,
							't_realText' : t_realText }

		self.variables = { 'g_vars' : g_vars,
						'd_vars' : d_vars }

		self.loss = { 'g_loss' : g_loss,
					'g_loss_NLL' : loss_NLL,
					'd_loss' : d_loss }

		self.outputs = { 'G' : t_freeBehavior,
						'freeText': t_freeText }

		self.summary = summary

		
	def train(self):
		opts = self.opts
		input_tensors = self.input_tensors
		loss = self.loss
		outputs = self.outputs
		variables = self.variables
		summary = self.summary

		d_optim = tf.train.AdamOptimizer(opts.lr, beta1=opts.beta1) \
							.minimize(loss['d_loss'], var_list=variables['d_vars'])
		g_optim = tf.train.AdamOptimizer(opts.lr, beta1=opts.beta1) \
							.minimize(loss['g_loss'], var_list=variables['g_vars'])
		self.summaryWriter = tf.summary.FileWriter("./logs", self.sess.graph)
		tf.global_variables_initializer().run()

		d_sum = tf.summary.merge([summary['d_loss'], summary['d_loss_free'], summary['d_loss_teacher']])
		g_sum = tf.summary.merge([summary['g_loss'], summary['g_loss_NLL'], summary['g_loss_fool']])

		sample_z = np.random.uniform(-1, 1, size=(opts.batch_size, opts.z_dim))
		
		counter = 1
		start_time = time.time()
		could_load, checkpoint_counter = self.load(opts.checkpoint_dir)
		if could_load:
			counter = checkpoint_counter

		f_valid = open('valid.txt','w')
		lossnames_to_print = ['g_loss', 'd_loss']
		for epoch in range(opts.nEpochs):
			nBatches = self.data_loader.nSamples['train'] // opts.batch_size

			for batch_idx in range(nBatches):
				real_text = self.data_loader.get_batch_text(batch_idx, opts.sequence_length)
				z_noise = np.random.uniform(-1,1, [opts.batch_size, opts.z_dim] )
	
				# Update D network
				_, d_loss, summary_str = self.sess.run([d_optim, loss['d_loss'], d_sum],
					feed_dict={ 
						input_tensors['t_z'] : z_noise,
						input_tensors['t_realText'] : real_text
					})
				self.summaryWriter.add_summary(summary_str, counter)
	
				# Update G network
				_, g_loss, summary_str = self.sess.run([g_optim, loss['g_loss'], g_sum],
					feed_dict={
						input_tensors['t_z'] : z_noise,
						input_tensors['t_realText'] : real_text
					})
				self.summaryWriter.add_summary(summary_str, counter)
	
				# Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
				_, g_loss, summary_str = self.sess.run([g_optim, loss['g_loss'], g_sum],
					feed_dict={
						input_tensors['t_z'] : z_noise,
						input_tensors['t_realText'] : real_text
					})
				self.summaryWriter.add_summary(summary_str, counter)
	
	
				counter += 1
				if counter % opts.print_every==0:
					elapsed = time.time() - start_time
					log( epoch, batch_idx, nBatches, lossnames_to_print, [g_loss,d_loss], elapsed )
	
#				if np.mod(counter, 100) == 1:
#					samples, d_loss, g_loss = self.sess.run(
#						[self.sampler, self.d_loss, self.g_loss],
#						feed_dict={
#								self.t_z: sample_z,
#								self.inputs: sample_inputs,
#								self.t_y:sample_labels,
#						}
#					)
#					manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
#					manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
#					save_images(samples, [manifold_h, manifold_w],
#								'./{}/train_{:02d}_{:04d}.png'.format(opts.sample_dir, epoch, batch_idx))
#					print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
				if np.mod(counter, opts.save_every_batch) == 1:
					self.save(opts.checkpoint_dir, counter)

			# run test for every epoch
			z_noise = np.random.uniform(-1,1, [opts.batch_size, opts.z_dim] )
			fake_text = self.sess.run(outputs['freeText'],
										feed_dict={ input_tensors['t_z'] : z_noise } )
			fake_text = " ".join([ self.data_loader.vocab.id_to_word(w) for w in fake_text[0].tolist() ])
			print( fake_text )
			f_valid.write( fake_text+'\n' )
			f_valid.flush()
		f_valid.close()

	@property
	def model_dir(self):
		return "{}_{}_{}_{}_{}".format( self.opts.model, self.opts.dataset, self.opts.batch_size,
										 self.opts.output_height, self.opts.output_width)
			
	def save(self, checkpoint_dir, step):
		model_name = "DCGAN.model"
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=step)

	def load(self, checkpoint_dir):
		import re
		print(" [*] Reading checkpoints...")
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
			print(" [*] Success to read {}".format(ckpt_name))
			return True, counter
		else:
			print(" [*] Failed to find a checkpoint")
			return False, 0


