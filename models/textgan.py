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
def generator(z, vocab_size, opts):
	with tf.variable_scope("generator") as scope:
		# make 2-layer gru
		stacked_gru = tf.contrib.rnn.MultiRNNCell(
					[GRUcell(opts.gru_num_units) for _ in range(opts.gru_num_layers)], state_is_tuple=True )

		# feed z
		z_fc = linear( z, opts.gru_num_units, scope )
		zero_state = stacked_gru.zero_state( batch_size=opts.batch_size, dtype=tf.float32 )
		cell_output, initial_state = stacked_gru( z_fc, zero_state )

		# manual unrolling 
		state = initial_state
		gru_outputs = []
		for time_step in range(opts.sequence_length):
			cell_output, state = stacked_gru( cell_output, state )
			gru_outputs.append( cell_output )
			
		gru_outputs_reshape = tf.reshape( gru_outputs, [-1, stacked_gru.output_size] )

		with tf.variable_scope('logits') as scope_logits:
			logits = linear( gru_outputs_reshape, vocab_size, scope_logits )

		# sample text from logits
		words = tf.nn.softmax( logits )
		words = tf.to_int32( words )
		#words = tf.multinomial( logits,1 )
		words = tf.reshape( words, [opts.batch_size, -1] )

		return words

# network for sequence classification
# ref: https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/
def discriminator(text, vocab_size, opts, reuse=False):
	with tf.variable_scope("discriminator") as scope:
		if reuse:
			scope.reuse_variables()

#		# represent text as one-hot
#		text = tf.one_hot( text, vocab_size, 1., 0. )
#
#		# reshape and embedding and back
#		text_reshaped = tf.reshape( text, [-1, vocab_size] )
#		with tf.variable_scope('embedding') as scope_embedding:
#			embedded_to_be_reshaped = linear( text_reshaped, opts.gru_num_units, scope_embedding )
#		embedded = tf.reshape( embedded_to_be_reshaped, [opts.batch_size, -1, opts.gru_num_units] )

		# embedding
		with tf.variable_scope("embedding") as scope_embedding:
			embedding_map = tf.get_variable( "map", [vocab_size,opts.gru_num_units] )
			text_embedded = tf.nn.embedding_lookup( embedding_map, text )

		# make 2-layer gru
		stacked_gru = tf.contrib.rnn.MultiRNNCell(
						[GRUcell(opts.gru_num_units,True) for _ in range(opts.gru_num_layers)])

		# unroll -> opts.sequence_length
		zero_state = stacked_gru.zero_state( batch_size=opts.batch_size, dtype=tf.float32 )
		gru_outputs, _ = tf.nn.dynamic_rnn( cell=stacked_gru,
											inputs=text_embedded,
											#sequence_length=opts.sequence_length,
											initial_state=zero_state,
											dtype=tf.float32,
											scope=scope )

		# get last output
		gru_outputs = tf.transpose( gru_outputs, [1,0,2] )
		gru_last_output = tf.gather( gru_outputs, int(gru_outputs.get_shape()[0])-1 )
		
		# fully connected
		with tf.variable_scope('logits') as scope_logits:
			logits = linear( gru_last_output, 2, scope_logits )

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
		self.model = opts.model
		self.batch_size = opts.batch_size

		self.data_loader = create_dataloader(opts)
		self.build_model()

	def build_model(self):
		opts = self.opts
		vocab_size = self.data_loader.vocab_size

		# t_z = [batch_size X z_dim], latent noise
		# t_realText = [batch_size X sequence_length], each row has word_ids
		t_z = tf.placeholder(tf.float32, [opts.batch_size, opts.z_dim], name='z')
		t_realText = tf.placeholder(tf.int32, [opts.batch_size, opts.sequence_length], name='real_text')
		
		t_fakeText = generator( t_z, vocab_size, opts )
		t_D_fake, t_D_fake_logits = discriminator( t_fakeText, vocab_size, opts )
		t_D_real, t_D_real_logits = discriminator( t_realText, vocab_size, opts, reuse=True)

		g_loss      = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
											logits=t_D_fake_logits, labels=tf.ones_like(t_D_fake)))

		d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
											logits=t_D_real_logits, labels=tf.ones_like(t_D_real)))
		d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
											logits=t_D_fake_logits, labels=tf.zeros_like(t_D_fake)))
		d_loss = d_loss_real + d_loss_fake

		summary = {}
		summary['g_loss'] = tf.summary.scalar("g_loss", g_loss)
		summary['d_loss_real'] = tf.summary.scalar("d_loss_real", d_loss_real)
		summary['d_loss_fake'] = tf.summary.scalar("d_loss_fake", d_loss_fake)
		summary['d_loss'] = tf.summary.scalar("d_loss", d_loss)

		self.saver = tf.train.Saver()

		t_vars = tf.trainable_variables()
		g_vars = [var for var in t_vars if 'generator' in var.name]
		d_vars = [var for var in t_vars if 'discriminator' in var.name]

		self.input_tensors = { 't_z' : t_z,
							't_realText' : t_realText }

		#self.variables = { 'g_vars' : g_vars,
		#				'd_vars' : d_vars }
		self.variables = { 'g_vars' : g_vars,
						'd_vars' : d_vars }

		self.loss = { 'g_loss' : g_loss,
					'd_loss' : d_loss }

		self.outputs = { 'G' : t_fakeText }

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
		self.summaryWriter = tf.summary.FileWriter("./logs", self.sess.graph)
		pdb.set_trace()
		g_optim = tf.train.AdamOptimizer(opts.lr, beta1=opts.beta1) \
							.minimize(loss['g_loss'], var_list=variables['g_vars'])
		tf.global_variables_initializer().run()

		d_sum = tf.summary.merge([summary['d_loss'], summary['d_loss_real'], summary['d_loss_fake']])

		sample_z = np.random.uniform(-1, 1, size=(opts.sample_num , opts.z_dim))
		
		counter = 1
		start_time = time.time()
		could_load, checkpoint_counter = self.load(opts.checkpoint_dir)
		if could_load:
			counter = checkpoint_counter

		for epoch in xrange(opts.nEpochs):
			batch_idxs = min(len(self.data), opts.train_size) // opts.batch_size
			real_text, z_noise = self.data_loader.get_batch()

			# Update D network
			_, summary_str = self.sess.run([d_optim, d_sum],
				feed_dict={ 
					input_tensors['t_z'] : z_noise,
					input_tensors['t_realText'] : real_text
				})
			self.writer.add_summary(summary_str, counter)

			# Update G network
			_, summary_str = self.sess.run([g_optim, summary['g_loss']],
				feed_dict={
					self.t_z: batch_z, 
				})
			self.writer.add_summary(summary_str, counter)

			# Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
			_, summary_str = self.sess.run([g_optim, summary['g_loss']],
				feed_dict={
					self.t_z: batch_z, 
				})
			self.writer.add_summary(summary_str, counter)


			errD_fake = self.d_loss_fake.eval({
					self.t_z: batch_z, 
					self.t_y:batch_labels
			})
			errD_real = self.d_loss_real.eval({
					self.inputs: batch_images,
					self.t_y:batch_labels
			})
			errG = self.g_loss.eval({
					self.t_z: batch_z,
					self.t_y: batch_labels
			})

			counter += 1
			if idx % self.print_every==0:
				print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
				% (epoch, idx, batch_idxs,
					time.time() - start_time, errD_fake+errD_real, errG))

			if np.mod(counter, 100) == 1:
				samples, d_loss, g_loss = self.sess.run(
					[self.sampler, self.d_loss, self.g_loss],
					feed_dict={
							self.t_z: sample_z,
							self.inputs: sample_inputs,
							self.t_y:sample_labels,
					}
				)
				manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
				manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
				save_images(samples, [manifold_h, manifold_w],
							'./{}/train_{:02d}_{:04d}.png'.format(opts.sample_dir, epoch, idx))
				print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
			if np.mod(counter, 500) == 2:
				self.save(opts.checkpoint_dir, counter)

	@property
	def model_dir(self):
		return "{}_{}_{}_{}_{}".format( self.model, self.dataset, self.batch_size, self.output_height, self.output_width)
			
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


