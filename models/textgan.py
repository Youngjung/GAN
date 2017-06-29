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

def generator(z, opts, initializer):
	lstm_cell = tf.contrib.rnn.BasicLSTMCell( num_units=opts.num_lstm_units )
	if opts.phase == 'train':
		lstm_cell = tf.contrib.rnn.DropoutWrapper( lstm_cell,
													input_keep_prob=opts.lstm_dropout_keep_prob,
													output_keep_prob=opts.lstm_dropout_keep_prob )
	with tf.variable_scope("generator") as scope:
		zero_state = lstm_cell.zero_state( batch_size=opts.batch_size, dtype=tf.float32 )
		_, initial_state = lstm_cell( z, zero_state )

		scope.reuse_variables()

		if opts.phase == 'test':
			tf.concat( axis=1, values=initial_state, name='initial_state' )
			state_feed = tf.placeholder( dtype=tf.float32,
										shape = [None, sum(lstsm_cell.state_size)],
										name='state_feed')
			state_tuple = tf.split( state_feed, 2, 1 )

			lstm_outputs, state_tuple = lstm_cell( tf.squeeze( z, axis=[1] ), state=state_tuple )

			tf.concat( axis=1, values=state_tuple, name='state' )
		else:
			lstm_outputs, _ = tf.nn.dynamic_rnn( cell=lstm_cell,
												inputs=z,
												sequence_length=opts.sequence_length,
												initial_state=initial_state,
												dtype=tf.float32,
												scope=scope )

		lstm_outputs = tf.reshape( lstm_outputs, [-1, lstm_cell.output_size] )

		with tf.variable_scope('logits') as scope_logits:
			logits = tf.contrib.layers.fully_connected( lstm_outputs,
														num_outputs = opts.vocab_size,
														activation_fn = None,
														weights_initializer = initializer,
														scope = scope_logits )
		if opts.phase == 'test':
			return tf.nn.softmax( logits, name='softmax' )
		return tf.nn.tanh(h4)

def discriminator(self, text, reuse=False):
	opts = self.opts
	with tf.variable_scope("discriminator") as scope:
		if reuse:
			scope.reuse_variables()

		GRU_cell1 = tf.contrib.rnn.GRUCell( num_units=self.opts.num_rnn_units )
		GRU_cell2 = tf.contrib.rnn.GRUCell( num_units=self.opts.num_rnn_units )

		if opts.phase == 'train':
			GRU_cell1 = tf.contrib.rnn.DropoutWrapper( GRU_cell1, input_keep_prob=0.5, output_keep_prob=0.5 )
			GRU_cell2 = tf.contrib.rnn.DropoutWrapper( GRU_cell2, input_keep_prob=0.5, output_keep_prob=0.5 )

		zero_state = GRU_cell1.zero_state( batch_size=opts.batch_size, dtype=tf.float32)
		_, initial_state = GRU_cell1( subject_text, zero_state )

		h0 = tf.nn.dynamic_rnn( cell=GRU_cell1,
									inputs=subject_Text, 
									sequence_length=?, 
									initial_state=initial_state,
									dtype=tf.float32,
									scope=lstm_scope)

		return tf.nn.sigmoid(h4), h4
	
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

		t_z = tf.placeholder(tf.float32, [None, opts.z_dim], name='z')
		t_realText = tf.placeholder(tf.float32, [opts.batch_size] + image_dims, name='real_images')
		
		t_fakeText = self.generator( t_z )
		t_D_fake, t_D_fake_logits = self.discriminator( t_fakeText )
		t_D_real, t_D_real_logits = self.discriminator( t_realText, reuse=True)

		g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(t_D_fake_logits, tf.ones_like(t_D_fake)))

		d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(t_D_real_logits, tf.oness_like(t_D_real)))
		d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(t_D_fake_logits, tf.zeros_like(t_D_fake)))
		d_loss = d_loss_real + d_loss_fake

		summary = []
		summary['g_loss'] = tf.summary.scalar("g_loss", g_loss)
		summary['d_loss_real'] = tf.summary.scalar("d_loss_real", d_loss_real)
		summary['d_loss_fake'] = tf.summary.scalar("d_loss_fake", d_loss_fake)
		summary['d_loss'] = tf.summary.scalar("d_loss", d_loss)

		self.saver = tf.train.Saver()

		t_vars = tf.trainable_variables()
		g_vars = [var for var in t_vars if 'g_' in var.name]
		d_vars = [var for var in t_vars if 'd_' in var.name]

		self.input_tensors = { 't_z' : t_z,
							't_realText' : t_realText }

		self.variables = { 'g_vars' : g_vars,
						'd_vars' : d_vars }

		self.loss = { 'g_loss' : g_loss,
					'd_loss' : d_loss }

		self.outputs = { 'G' : t_fakeText }

		self.summary = summary

		
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
		tf.global_variables_initializer().run()

		d_sum = tf.summary.merge([summary['d_loss'], summary['d_loss_real'], summary['d_loss_fake']])
		self.summaryWriter = tf.summary.FileWriter("./logs", self.sess.graph)

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
			_, summary_str = self.sess.run([d_optim, d_sum]],
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


