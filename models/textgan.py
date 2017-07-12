from __future__ import division
import os
import sys
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from misc import *
from data_loader.vocabulary import Vocabulary, load_vocab

import input_ops

import pdb

myInitializer = tf.random_uniform_initializer(minval=-0.08,maxval=0.08)

def float_type():
	return tf.float32

def GRUcell( num_units, isTraining=False, dropout_keep_prob=0.5 ):
	gru_cell = tf.contrib.rnn.GRUCell( num_units )
	if isTraining:
		gru_cell = tf.contrib.rnn.DropoutWrapper( gru_cell, 
													input_keep_prob=dropout_keep_prob,
													output_keep_prob=dropout_keep_prob )
	return gru_cell

def LSTMcell( num_units, isTraining=False, dropout_keep_prob=0.5 ):
	lstm_cell = tf.contrib.rnn.BasicLSTMCell( num_units=num_units )
	if isTraining:
		lstm_cell = tf.contrib.rnn.DropoutWrapper( lstm_cell,
													input_keep_prob=dropout_keep_prob,
													output_keep_prob=dropout_keep_prob )
	return lstm_cell

def build_seq_embeddings( input_seqs, vocab_size, opts ):

	with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
		embedding_map = tf.get_variable(
				name="map",
				shape=[vocab_size, opts.embedding_size],
				initializer=myInitializer)
		seq_embeddings = tf.nn.embedding_lookup(embedding_map, input_seqs)
	
	return seq_embeddings

# network for sequence generation
# ref: show_and_tell_model.py
def generator( t_inputs, opts, vocab_size, seq_length=None, isForInference=False, reuse=False ):
	seq_length = seq_length or opts.sequence_length
	isTraining = opts.phase=='train'

	# unpack inputs
	t_noise = t_inputs['noise']
	t_realText = t_inputs['text']
	if isForInference:
		t_realText = tf.expand_dims( t_realText, 1 )
	else:
		t_realText_shifted = t_inputs['text_shifted']
		t_input_mask = t_inputs['mask']

	# storage to be returned
	t_outputs = {}
	t_loss = {}
	t_misc = {}
	train_op = None # will be replaced below

	with tf.variable_scope("generator") as scope_generator:
		if reuse: scope_generator.reuse_variables()
		with tf.variable_scope("noise_fc") as scope_noise:
			t_noise_fc = tf.contrib.layers.fully_connected( inputs = t_noise,
															num_outputs = opts.gru_num_units,
															activation_fn = None,
															weights_initializer = myInitializer,
															scope = scope_noise )
		with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
			embedding_map = tf.get_variable(
					name="map",
					shape=[vocab_size, opts.embedding_size],
					initializer=myInitializer)
			t_realText_embedded = tf.nn.embedding_lookup(embedding_map, t_realText)
	
		with tf.variable_scope('lstm') as scope_lstm:
			# make cell
			if isForInference:
				cell = LSTMcell( opts.gru_num_units, isTraining=False ) 
			else:
				# 2-layer gru
				#gru_cells = [GRUcell( opts.gru_num_units, isTraining, opts.keep_prob ) for _ in range(opts.gru_num_layers)]
				#cell = tf.contrib.rnn.MultiRNNCell( gru_cells )
				cell = LSTMcell( opts.gru_num_units, isTraining, opts.keep_prob ) 

			# feed noise
			t_misc['initial_state'] = cell.zero_state( t_noise.get_shape()[0], dtype=float_type() )
			_, initial_state = cell( t_noise_fc, t_misc['initial_state'] )
		
			# allow the LSTM variables to be reused
			scope_lstm.reuse_variables()
			if isForInference:
				initial_state_for_inference = tf.concat(initial_state, axis=1, name='initial_state_for_inference')

				state_feed = tf.placeholder(dtype=float_type(), shape=[None,sum(cell.state_size)], name='state_feed')
				state_tuple = tf.split(state_feed, num_or_size_splits=2, axis=1)

				# run a single LSTM step
				outputs, state_tuple = cell( inputs=tf.squeeze(t_realText_embedded, axis=[1]), state=state_tuple)

				# concat the resulting state
				state_concat = tf.concat(axis=1, values=state_tuple, name='state')

			else:
				# run the batch of sequence embeddings through the LSTM
				sequence_length = tf.reduce_sum( t_input_mask, 1 )
				outputs, _ = tf.nn.dynamic_rnn( cell = 		cell,
												inputs =	t_realText_embedded,
												initial_state = 	initial_state,
												dtype = float_type(),
												scope = scope_generator )
		
				outputs = tf.reshape( outputs, [-1, cell.output_size] )

		with tf.variable_scope("logits") as scope_logits:
			logits = tf.contrib.layers.fully_connected( inputs = outputs,
														num_outputs = vocab_size,
														activation_fn = None,
														weights_initializer = myInitializer,
														scope = scope_logits )

		if isForInference:
			return tf.argmax( tf.nn.softmax(logits, name="softmax"), axis=1 ), initial_state_for_inference, state_feed, state_concat
		else:
			targets = tf.reshape( t_realText_shifted, [-1] )
			weights = tf.to_float( tf.reshape( t_input_mask, [-1] ) )
	
			losses = tf.nn.sparse_softmax_cross_entropy_with_logits( labels=targets, logits=logits )
			batch_loss = tf.div( tf.reduce_sum( tf.multiply(losses,weights) ), tf.reduce_sum(weights), name='batch_loss' )
			tf.losses.add_loss( batch_loss )
			total_loss = tf.losses.get_total_loss()
	
			return total_loss, losses, weights

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

		self.reader = tf.TFRecordReader()
		self.build_inputs()
		self.build_model()

	def build_inputs(self):
		opts = self.opts

		# load vocabulary
		filename_vocab = os.path.join( opts.data_dir, opts.dataset, 'word_counts.txt' )
		self.vocab = load_vocab( filename_vocab )

		# generate input filename queue
		input_queue = input_ops.prefetch_input_data(
				self.reader,
				opts.input_file_pattern,
				is_training=opts.phase=='train',
				batch_size=opts.batch_size,
				values_per_shard=opts.values_per_input_shard,
				input_queue_capacity_factor=opts.input_queue_capacity_factor,
				num_reader_threads=opts.num_input_reader_threads)
		
		# Image processing and random distortion. Split across multiple threads
		# with each thread applying a slightly different distortion.
		assert opts.num_preprocess_threads % 2 == 0
		images_and_captions = []
		for thread_id in range(opts.num_preprocess_threads):
			serialized_sequence_example = input_queue.dequeue()
			encoded_image, caption = input_ops.parse_sequence_example(
					serialized_sequence_example,
					image_feature=opts.image_feature_name,
					caption_feature=opts.caption_feature_name)
			image = process_image(encoded_image, opts, thread_id=thread_id)
			images_and_captions.append([image, caption])
		
		# Batch inputs.
		queue_capacity = (2 * opts.num_preprocess_threads *
											opts.batch_size)
		images, input_seqs, target_seqs, input_mask = (
				input_ops.batch_with_dynamic_pad(images_and_captions,
													 batch_size=opts.batch_size,
													 queue_capacity=queue_capacity))

		t_z = tf.placeholder(tf.float32, [opts.batch_size, opts.z_dim], name='z')
		t_realText = input_seqs
		t_realText_shifted = target_seqs
		t_mask = input_mask

		self.inputs = { 'noise': t_z,
						'text': t_realText,
						'text_shifted': t_realText_shifted,
						'mask': t_mask }
		
	def build_model(self):
		opts = self.opts
		vocab_size = self.vocab.vocab_size

		# self.inputs should be built beforehand
		total_loss, losses, weights = generator( self.inputs, opts, vocab_size ) 

		# for validation
		t_z_valid = tf.placeholder(tf.float32, [1, opts.z_dim], name='z_valid')
		t_text_valid = tf.placeholder( dtype=tf.int64, shape=[None], name='text_valid' )
		self.inputs_valid = { 'noise': t_z_valid,
							'text': t_text_valid }
		inferenced, initial_state, state_feed, state_concat = generator(
									 self.inputs_valid, opts, vocab_size, isForInference=True, reuse=True ) 

		self.total_loss = total_loss
		self.losses = losses
		self.weights = weights

		self.inferenced = inferenced
		self.initial_state = initial_state
		self.state_feed = state_feed
		self.state_concat = state_concat

		self.saver = tf.train.Saver()

	def train(self):
		sess = self.sess
		opts = self.opts
		global_step = tf.Variable( initial_value=0, name='global_step', trainable=False )

		# Set up the training ops
		nBatches = opts.num_examples_per_epoch // opts.batch_size
		decay_steps = int(nBatches*opts.decay_every_nEpochs)
		def _learning_rate_decay_fn( learning_rate, global_step):
			return tf.train.exponential_decay( learning_rate, global_step, decay_steps=decay_steps,
												decay_rate=opts.decay_factor, staircase=True )
		learning_rate_decay_fn=_learning_rate_decay_fn
		train_op = tf.contrib.layers.optimize_loss( loss=self.total_loss, global_step=global_step, learning_rate=opts.lr,
													optimizer='SGD', clip_gradients=5.0,
													learning_rate_decay_fn=learning_rate_decay_fn )
		summary_loss = tf.summary.scalar('loss', self.total_loss)

		self.summaryWriter = tf.summary.FileWriter(opts.log_dir, self.sess.graph)
		tf.global_variables_initializer().run()

		# start input enqueue threads
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

		counter = 0
		start_time = time.time()
		could_load, checkpoint_counter = self.load( opts.checkpoint_dir )
		if could_load:
			counter = checkpoint_counter
		lossnames_to_print = ['g_loss']
		try:
			f_valid = open('valid.txt','w')
			for epoch in range(opts.nEpochs):
				for batch_idx in range(nBatches):
					counter += 1
					z_noise = np.random.uniform(-1,1, [opts.batch_size, opts.z_dim] )
					_, g_loss, summary_str = self.sess.run([train_op, self.total_loss, summary_loss],
						feed_dict={
							self.inputs['noise'] : z_noise,
						})
					self.summaryWriter.add_summary(summary_str, counter)
	
					if counter % opts.print_every==0:
						elapsed = time.time() - start_time
						log( epoch, batch_idx, nBatches, lossnames_to_print, [g_loss], elapsed, counter )
	
					if np.mod(counter, opts.save_every_batch) == 1 or \
						(epoch==opts.nEpochs-1 and batch_idx==nBatches-1) :
						self.save(opts.checkpoint_dir, counter)

				# run test after every epoch
				z_noise = np.random.uniform(-1,1, [1, opts.z_dim] )
				initial_state = self.sess.run(self.initial_state, feed_dict={ self.inputs_valid['noise'] : z_noise } )
				state_output = initial_state
				word = np.array([vocab.start_id])
				fake_text = []
				for _ in range(opts.sequence_length):
					word, state_output = self.sess.run([self.inferenced, self.state_concat],
											feed_dict={ self.inputs_valid['text'] : word,
														self.state_feed: state_output } )
					fake_text.append( word )
				fake_text = " ".join([ vocab.id_to_word(w[0]) for w in fake_text ])
				print( fake_text )
				f_valid.write( fake_text+'\n' )
				f_valid.flush()

		except tf.errors.OutOfRangeError:
			print('Finished training: epoch limit reached')
		finally:
			coord.request_stop()
		coord.join(threads)

	def valid(self):
		sess = self.sess
		opts = self.opts
		vocab = self.vocab

		# Set up the training ops
		nBatches = opts.num_examples_per_epoch // opts.batch_size
		summary_loss = tf.summary.scalar('loss', self.total_loss)

		# start input enqueue threads
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

		counter = 0
		start_time = time.time()
		could_load, checkpoint_counter = self.load( opts.checkpoint_dir )
		if could_load:
			counter = checkpoint_counter
		else:
			print( 'cannot run validation: checkpoint load failed' )
			sys.exit()
		lossnames_to_print = ['g_loss']
		try:
			f_valid = open('valid.txt','w')
			for epoch in range(opts.nEpochs):
				# run test after every epoch
				z_noise = np.random.uniform(-1,1, [1, opts.z_dim] )
				initial_state = self.sess.run(self.initial_state, feed_dict={ self.inputs_valid['noise'] : z_noise } )
				state_output = initial_state
				word = np.array([vocab.start_id])
				fake_text = []
				pdb.set_trace()
				for _ in range(opts.sequence_length):
					word, state_output = self.sess.run([self.inferenced, self.state_concat],
											feed_dict={ self.inputs_valid['text'] : word,
														self.state_feed: state_output } )
					if word[0] == vocab.end_id:
						word = np.array([vocab.start_id])
					fake_text.append( word )
				fake_text = " ".join([ vocab.id_to_word(w[0]) for w in fake_text ])
				print( fake_text )
				f_valid.write( fake_text+'\n' )
				f_valid.flush()

		except tf.errors.OutOfRangeError:
			print('Finished : epoch limit reached')
		finally:
			coord.request_stop()
		coord.join(threads)
		
	def train_paused_for_following_ptb_first(self):
		# main training
		f_valid = open('valid.txt','w')
		lossnames_to_print = ['g_loss', 'd_loss']
		skip_learning_D = skip_learning_G = False
		for epoch in range(opts.nEpochs):
			nBatches = self.data_loader.nSamples['train'] // opts.batch_size
			valid_real_text = self.data_loader.get_batch_text(epoch, opts.sequence_length, 'valid')
			valid_z_noise = np.random.uniform(-1,1, [opts.batch_size, opts.z_dim] )

			for batch_idx in range(nBatches):
				real_text = self.data_loader.get_batch_text(batch_idx, opts.sequence_length)
				z_noise = np.random.uniform(-1,1, [opts.batch_size, opts.z_dim] )

				if skip_learning_D:
					d_loss, d_accFree, d_accTeacher, summary_str = self.sess.run(
						[loss['d_loss'], outputs['d_accFree'], outputs['d_accTeacher'], d_sum],
						feed_dict={ 
							input_tensors['t_z'] : z_noise,
							input_tensors['t_realText'] : real_text
						})
					self.summaryWriter.add_summary(summary_str, counter)
					d_acc = sum(d_accFree + d_accTeacher)*0.5/opts.batch_size
					skip_learning_D = d_loss < 0.1
				else:
					# Update D network
					_, d_loss, d_accFree, d_accTeacher, summary_str = self.sess.run(
						[d_optim, loss['d_loss'], outputs['d_accFree'], outputs['d_accTeacher'], d_sum],
						feed_dict={ 
							input_tensors['t_z'] : z_noise,
							input_tensors['t_realText'] : real_text
						})
					self.summaryWriter.add_summary(summary_str, counter)
					d_acc = sum(d_accFree + d_accTeacher)*0.5/opts.batch_size
					skip_learning_D = d_loss < 0.1
	
					if skip_learning_D:
						print( 'skip learning D : d_loss({:.4f}), acc({:.4f}={:.4f}+{:.4f})'.format(
								d_loss, d_acc,sum(d_accFree)/opts.batch_size,sum(d_accTeacher)/opts.batch_size) )

				if skip_learning_G:
					g_loss, summary_str = self.sess.run([loss['g_loss'], g_sum],
						feed_dict={
							input_tensors['t_z'] : z_noise,
							input_tensors['t_realText'] : real_text
						})
					skip_learning_G = g_loss < 0.5
				else:
					# Update G network
					_, g_loss, summary_str = self.sess.run([g_optim, loss['g_loss'], g_sum],
						feed_dict={
							input_tensors['t_z'] : z_noise,
							input_tensors['t_realText'] : real_text
						})
					self.summaryWriter.add_summary(summary_str, counter)
					skip_learning_G = g_loss < 0.5
					if skip_learning_G:
						print( 'skip learning G : g_loss({:.4f}, D\'s acc({:.4f}={:.4f}+{:.4f})'.format(
								g_loss, d_acc,sum(d_accFree)/opts.batch_size,sum(d_accTeacher)/opts.batch_size) )
	
				counter += 1
				if counter % opts.print_every==0:
					elapsed = time.time() - start_time
					log( epoch, batch_idx, nBatches, lossnames_to_print, [g_loss,d_loss], elapsed )
	
				if np.mod(counter, opts.save_every_batch) == 1 or \
					(epoch==opts.nEpochs-1 and batch_idx==nBatches-1) :
					self.save(opts.checkpoint_dir, counter)

			# run test after every epoch
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
		model_name = "TextGAN.model"
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


