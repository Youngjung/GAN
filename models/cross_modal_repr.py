import tensorflow as tf
import numpy as np
import scipy
import h5py
import ops
import os
import glob
from os.path import join
from utils import image_embedding, image_processing, vocabulary
import random
import shutil
import nltk
import time
import pdb

def resize_image_for_inception( image ):
	return tf.image.resize_images( image, size=[299,299], method=tf.image.ResizeMethod.BILINEAR )

class CrossModalRepr():
	def __init__(self, sess, config):
		self.sess = sess
		self.config = config
		self.model = config.model
		self.config.nCapsPerImage = 10
		self.config.caption_vector_length = 1024

		self.initializer = tf.random_uniform_initializer( minval = -0.08, maxval = 0.08 )

		self.build_model()


	def build_model(self):
		config = self.config

		lstm_cell = tf.contrib.rnn.BasicLSTMCell( config.num_lstm_units )
		d_lstm_cell = tf.contrib.rnn.BasicLSTMCell( config.num_lstm_units )

		# for train mode
		image = tf.placeholder( tf.float32, [config.batch_size, config.image_size, config.image_size, 3] )
		text_feature = tf.placeholder(tf.float32, [config.batch_size,config.caption_vector_length])
		input_text = tf.placeholder(tf.int64, [config.batch_size,None])
		input_mask = tf.placeholder(tf.int64, [config.batch_size,None])
		target_text = tf.placeholder(tf.int64, [config.batch_size,None])

		# for inference mode
		image_infer = tf.placeholder( tf.float32, [config.image_size, config.image_size, 3] )
		text_feature_infer = tf.placeholder( tf.float32, [config.caption_vector_length] )
		text_feature_infer_exp = tf.expand_dims( text_feature_infer, 0 )
		input_word = tf.placeholder( tf.int64, [None] )
		input_word_seq = tf.expand_dims( input_word, 1 )
		state_feed = tf.placeholder(tf.float32, [None, sum(lstm_cell.state_size)])
		state_tuple = tf.split( state_feed, 2, 1 )
		
		image_infer_exp = tf.expand_dims(image_infer,0)
		image_resized = resize_image_for_inception( image )
		image_resized_infer = resize_image_for_inception( image_infer_exp )
		inception_output = image_embedding.inception_v3( image_resized, trainable=False, is_training=True )
		with tf.variable_scope('image_embedding') as image_emb_scope:
			image_embeddings = tf.contrib.layers.fully_connected( inputs = inception_output,
																	num_outputs = config.embedding_size,
																	activation_fn = None,
																	weights_initializer = self.initializer,
																	biases_initializer = None,
																	scope = image_emb_scope )
		inception_output_infer = image_embedding.inception_v3( image_resized_infer, trainable=False, is_training=False, reuse=True )
		with tf.variable_scope('image_embedding') as image_emb_scope:
			image_emb_scope.reuse_variables()
			image_embeddings_infer = tf.contrib.layers.fully_connected( inputs = inception_output_infer,
																	num_outputs = config.embedding_size,
																	activation_fn = None,
																	weights_initializer = self.initializer,
																	biases_initializer = None,
																	scope = image_emb_scope )
			inception_vars = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3" )

		with tf.variable_scope('text_embedding'), tf.device('/cpu:0'):
			embedding_map = tf.get_variable( name="embedding_map",
											shape=[config.vocab_size, config.embedding_size],
											initializer=self.initializer)
			input_embeddings = tf.nn.embedding_lookup( embedding_map, input_text )
			word_embeddings = tf.nn.embedding_lookup( embedding_map, input_word_seq )

		lstm_cell_withDropout = tf.contrib.rnn.DropoutWrapper( lstm_cell,
													input_keep_prob=config.dropout_keep_prob,
													output_keep_prob=config.dropout_keep_prob)
		zero_state = lstm_cell.zero_state( config.batch_size, dtype=tf.float32 )
		zero_state_infer = lstm_cell.zero_state( 1, dtype=tf.float32 )

		with tf.variable_scope('lstm', initializer=self.initializer) as lstm_scope:
			_, initial_state_text = lstm_cell_withDropout( text_feature, zero_state )
			lstm_scope.reuse_variables()
			_, initial_state_image = lstm_cell_withDropout( image_embeddings, zero_state )
			_, initial_state_infer = lstm_cell( text_feature_infer_exp, zero_state_infer )
			initial_state_infer = tf.concat( initial_state_infer, 1 )

			text_lengths = tf.reduce_sum( input_mask, 1 )
			lstm_outputs_text, _ = tf.nn.dynamic_rnn( cell=lstm_cell_withDropout,
														inputs = input_embeddings,
														sequence_length = text_lengths,
														initial_state = initial_state_text,
														dtype = tf.float32,
														scope = lstm_scope )
			lstm_outputs_image, _ = tf.nn.dynamic_rnn( cell=lstm_cell_withDropout,
														inputs = input_embeddings,
														sequence_length = text_lengths,
														initial_state = initial_state_image,
														dtype = tf.float32,
														scope = lstm_scope )

			lstm_outputs_infer, state_tuple = lstm_cell( inputs = tf.squeeze( word_embeddings, axis=[1] ), state=state_tuple )
			state_tuple = tf.concat( state_tuple, 1 )

		lstm_outputs_reshaped_text = tf.reshape( lstm_outputs_text, [-1, lstm_cell.output_size] )
		lstm_outputs_reshaped_image = tf.reshape( lstm_outputs_image, [-1, lstm_cell.output_size] )
		with tf.variable_scope('logits') as logits_scope:
			logits_text = tf.contrib.layers.fully_connected( inputs = lstm_outputs_reshaped_text,
														num_outputs = config.vocab_size,
														activation_fn = None,
														weights_initializer = self.initializer,
														scope = logits_scope )
			logits_scope.reuse_variables()
			logits_image = tf.contrib.layers.fully_connected( inputs = lstm_outputs_reshaped_image,
														num_outputs = config.vocab_size,
														activation_fn = None,
														weights_initializer = self.initializer,
														scope = logits_scope )
			logits_infer = tf.contrib.layers.fully_connected( inputs = lstm_outputs_infer,
															num_outputs = config.vocab_size,
															activation_fn = None,
															weights_initializer = self.initializer,
															scope = logits_scope )

		targets_reshaped = tf.reshape( target_text, [-1] )
		mask_reshaped = tf.to_float( tf.reshape( input_mask, [-1] ) )
		
		losses_text = tf.nn.sparse_softmax_cross_entropy_with_logits( logits = logits_text, labels = targets_reshaped )
		losses_image = tf.nn.sparse_softmax_cross_entropy_with_logits( logits = logits_image, labels = targets_reshaped )
		batch_loss_text = tf.div( tf.reduce_sum( tf.multiply( losses_text, mask_reshaped ) ),
								tf.reduce_sum( mask_reshaped ),
								name = 'batch_loss_text' )
		tf.losses.add_loss( batch_loss_text )
		batch_loss_image = tf.div( tf.reduce_sum( tf.multiply( losses_image, mask_reshaped ) ),
								tf.reduce_sum( mask_reshaped ),
								name = 'batch_loss_image' )
		tf.losses.add_loss( batch_loss_image )
		NLL_loss = tf.losses.get_total_loss()
		self.NLL_loss = NLL_loss

		softmax_infer = tf.nn.softmax( logits_infer )
		next_word = tf.argmax( softmax_infer, 1 )
		
		d_lstm_cell = tf.contrib.rnn.DropoutWrapper( d_lstm_cell,
													input_keep_prob=config.dropout_keep_prob,
													output_keep_prob=config.dropout_keep_prob)
		with tf.variable_scope('d_lstm', initializer=self.initializer) as lstm_scope:
			_, initial_state_text = d_lstm_cell( text_feature, zero_state )
			lstm_scope.reuse_variables()
			_, initial_state_image = d_lstm_cell( image_embeddings, zero_state )
			d_outputs_text, _ = tf.nn.dynamic_rnn( cell = d_lstm_cell,
													inputs = lstm_outputs_text,
													sequence_length = text_lengths,
													initial_state = initial_state_image,
													dtype = tf.float32,
													scope = lstm_scope )
			d_outputs_image, _ = tf.nn.dynamic_rnn( cell = d_lstm_cell,
													inputs = lstm_outputs_image,
													sequence_length = text_lengths,
													initial_state = initial_state_image,
													dtype = tf.float32,
													scope = lstm_scope )
		
		last_idx = tf.expand_dims( text_lengths, 1 ) - 1
		batch_range = tf.expand_dims(tf.constant( np.array(range(config.batch_size)),dtype=tf.int64 ),1)
		gather_idx = tf.concat( [batch_range,last_idx], axis=1 )
		last_output_text = tf.gather_nd( d_outputs_text, gather_idx )
		last_output_image = tf.gather_nd( d_outputs_image, gather_idx )

		with tf.variable_scope('d_logits') as logits_scope:
			d_logits_text = tf.contrib.layers.fully_connected( inputs = last_output_text,
														num_outputs = 1,
														activation_fn = None,
														weights_initializer = self.initializer,
														scope = logits_scope )
			logits_scope.reuse_variables()
			d_logits_image = tf.contrib.layers.fully_connected( inputs = last_output_image,
														num_outputs = 1,
														activation_fn = None,
														weights_initializer = self.initializer,
														scope = logits_scope )

		d_loss_text = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( name = 'd_loss_text',
										logits = d_logits_text, labels = tf.ones_like(d_logits_text) ) )
		d_loss_image = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( name = 'd_loss_image',
										logits = d_logits_image, labels = tf.zeros_like(d_logits_image) ) )
		d_loss = d_loss_text + d_loss_image

		g_loss_text = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( name = 'g_loss_text',
										logits = d_logits_text, labels = tf.zeros_like(d_logits_text) ) )
		g_loss_image = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( name = 'g_loss_image',
										logits = d_logits_image, labels = tf.ones_like(d_logits_image) ) )
		g_loss = g_loss_text + g_loss_image

		t_vars = tf.trainable_variables()
		d_vars = [var for var in t_vars if 'd_' in var.name]
		g_vars = [var for var in t_vars if 'd_' not in var.name]

		self.input_tensors = {
			'image' : image,
			'image_infer' : image_infer,
			'text_feature' : text_feature,
			'input_text' : input_text,
			'input_mask' : input_mask,
			'target_text' : target_text,
			'text_feature_infer' : text_feature_infer,
			'input_word' : input_word,
			'state_feed' : state_feed
		}

		self.variables = {
			'all' : t_vars,
			'D' : d_vars,
			'G' : g_vars,
		}

		self.loss = {
			'NLL' : NLL_loss,
			'D' : d_loss,
			'G' : g_loss,
		}

		self.outputs = {
			'image_embeddings_infer' : image_embeddings_infer,
			'initial_state_infer' : initial_state_infer,
			'state_tuple_infer' : state_tuple,
			'softmax_infer' : softmax_infer,
			'next_word' : next_word 
		}
		
		self.saver = tf.train.Saver()
		self.inception_loader = tf.train.Saver( inception_vars )

	def train(self):
		config = self.config
		input_tensors = self.input_tensors
		loss = self.loss
		outputs = self.outputs
		variables = self.variables

		optim_NLL = tf.train.AdamOptimizer(config.lr, beta1=config.beta1)\
									.minimize(loss['NLL'], var_list=variables['all'])
		optim_D = tf.train.AdamOptimizer(config.lr, beta1=config.beta1)\
									.minimize(loss['D'], var_list=variables['D'])
		optim_G = tf.train.AdamOptimizer(config.lr, beta1=config.beta1)\
									.minimize(loss['G'], var_list=variables['G'])
	
		tf.global_variables_initializer().run()
		print( 'loading inception checkpoint... from {}'.format( config.inception_checkpoint_file ) )
		self.inception_loader.restore( self.sess, config.inception_checkpoint_file )
	
		self.load_training_data()

		could_load, checkpoint_counter = self.load(config.checkpoint_dir)
		if could_load:
			counter = checkpoint_counter
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")

		start_time = time.time()
		nBatch = self.loaded_data['data_length']/config.batch_size
		counter = 1
		for i in range(config.nEpochs):
			batch_no = 0
			while batch_no*config.batch_size < self.loaded_data['data_length']:
				caption_vectors, captions, captions_shifted, mask, images = self.get_training_batch(batch_no)

				# run training optimizer
				_, NLL_loss = self.sess.run( [optim_NLL, loss['NLL']],
									feed_dict={ input_tensors['text_feature']:caption_vectors,
												input_tensors['input_text']:captions,
												input_tensors['target_text']:captions_shifted,
												input_tensors['input_mask']:mask,
												input_tensors['image']:images,
												} )
				log( i, batch_no, nBatch, ['NLL'], [NLL_loss], time.time()-start_time, counter )
				batch_no += 1
				counter += 1
				if (batch_no % config.save_every_batch) == 0:
					print( "Saving Images, Model" )
					self.save(config.checkpoint_dir, counter)

					# validation : text (for now, use training sample)
					state = self.sess.run( self.outputs['initial_state_infer'],
											feed_dict={ input_tensors['text_feature_infer']:caption_vectors[0]} )
					next_word = [self.vocab.start_id]
					sentence = []
					while next_word != self.vocab.end_id:
						next_word, state = self.sess.run( [ self.outputs['next_word'], self.outputs['state_tuple_infer'] ],
															feed_dict={ input_tensors['input_word']:next_word,
																		input_tensors['state_feed']:state } )
						sentence.append( self.vocab.id_to_word(int(next_word[0])) )
					print( 'validation' )
					print( 'text) ' + ' '.join(sentence) )
					# validation : image (for now, use training sample)
					image_embedding = self.sess.run( self.outputs['image_embeddings_infer'],
											feed_dict={ input_tensors['image_infer']:images[0]} )
					state = self.sess.run( self.outputs['initial_state_infer'],
											feed_dict={ input_tensors['text_feature_infer']:image_embedding[0]} )
					next_word = [self.vocab.start_id]
					sentence = []
					while next_word != self.vocab.end_id:
						next_word, state = self.sess.run( [ self.outputs['next_word'], self.outputs['state_tuple_infer'] ],
															feed_dict={ input_tensors['input_word']:next_word,
																		input_tensors['state_feed']:state } )
						sentence.append( self.vocab.id_to_word(int(next_word[0])) )
					print( 'image) ' + ' '.join(sentence) )
					print( 'GT) ' + ' '.join([ self.vocab.id_to_word(int(w)) for w in captions[0] ] ) )

			if i%5 == 0:
				self.save(config.checkpoint_dir, counter, note="epoch{}".format(i))
	
	def load_training_data(self):
		data_dir = self.config.data_dir
		dataset = self.config.dataset
		if dataset == 'flowers':
			start_time = time.time()
			print( 'loading skipthought...' )
			data = h5py.File(join(data_dir, 'flower_tv.hdf5'))
			flower_captions = {}
			for ds in data.iteritems():
				flower_captions[ds[0]] = np.array(ds[1])
			image_list = [key.encode('ascii','ignore') for key in flower_captions]
			image_list.sort()
	
			img_75 = int(len(image_list)*0.75)
			training_image_list = image_list[0:img_75]
			random.shuffle(training_image_list)

			# load reedscot_cvpr2016 encoded captions if specified
			print( 'loading DSSJE...' )
			data_icml_dir = join(data_dir,'flowers_caption_icml')
			data_cvpr_dir = join(data_dir,'cvpr2016_flowers/text_c10')
			full_filenames_feature = {}
			full_filenames_text = {}
			classnames = os.listdir( data_icml_dir )
			for classname in classnames:
				if os.path.isdir( join(data_icml_dir,classname) ) :
					filenames = glob.glob( join(data_icml_dir,classname,'*.npy') )
					for filename in filenames:
						full_filenames_feature[os.path.basename(filename[0:-4])+'.jpg'] = filename
				if os.path.isdir( join(data_cvpr_dir,classname) ) :
					filenames = glob.glob( join(data_cvpr_dir,classname,'*.txt') )
					for filename in filenames:
						full_filenames_text[os.path.basename(filename[0:-4])+'.jpg'] = filename
			flower_captions_reedscot = {}
			flower_text_features = {}
			all_captions = []
			for filename, npy in full_filenames_feature.iteritems():
				flower_text_features[filename] = np.load( npy )
			print( 'loading captions...' )
			for filename, txt in full_filenames_text.iteritems():
				with open(txt,'r') as f_txt:
					captions = [process_caption(line) for line in f_txt]
				flower_captions_reedscot[filename] = captions
				all_captions.extend( captions )
			flower_captions = flower_captions_reedscot
	
			fname_vocab = os.path.join( data_cvpr_dir, self.config.vocab_file )
			if os.path.isfile( fname_vocab ):
				print( 'loading vocab from {}...'.format(fname_vocab) )
				vocab = vocabulary.Vocabulary( fname_vocab )
			else:
				print( 'creating vocab...' )
				vocab = vocabulary.create_vocab(all_captions, fname_vocab )

			self.vocab = vocab
			self.loaded_data = {
				'image_list' : training_image_list,
				'captions' : flower_captions,
				'text_features' : flower_text_features,
				'data_length' : len(training_image_list)
			}
			print( 'load_training_data() done: {:.0f}s'.format(time.time()-start_time) )
	
		else:
			with open(join(data_dir, 'meta_train.pkl')) as f:
				meta_data = pickle.load(f)
			# No preloading for MS-COCO
			return meta_data


	def get_training_batch(self, batch_no):
		image_dir = join(self.config.data_dir,'flowers/jpg')
		loaded_data = self.loaded_data
		vocab = self.vocab
		batch_size = self.config.batch_size
		image_size = self.config.image_size
		caption_vector_length = self.config.caption_vector_length

		text_features = np.zeros((batch_size, caption_vector_length))
		images = np.zeros( (batch_size, image_size, image_size, 3) )

		cnt = 0
		captions = []
		for i in range(batch_no * batch_size, (batch_no+1) * batch_size ):
			idx = i % len(loaded_data['image_list'])
			image_id = loaded_data['image_list'][idx]

			random_caption = random.randint(0,self.config.nCapsPerImage-1) # randint arguments are inclusive
			captions.append( loaded_data['captions'][ image_id ][ random_caption ] )
			text_features[cnt,:] = loaded_data['text_features'][ image_id ][ random_caption ][0:caption_vector_length]

			fname_image = join(image_dir,image_id)
			image = image_processing.load_image_array( fname_image, image_size )
			images[cnt,:,:,:] = image

			cnt += 1

		max_caption_length = len(max(captions, key=len))
		mat_captions = np.zeros( (batch_size, max_caption_length), dtype=np.int  )
		for i, cap in enumerate(captions):
			for j, word in enumerate(cap):
				mat_captions[i,j] = vocab.word_to_id( word )

		mat_captions_shifted = np.copy( mat_captions[:,1:] )
		mat_captions = mat_captions[:,:-1]
		mat_captions_mask = (mat_captions>0).astype(int)

		return text_features, mat_captions, mat_captions_shifted, mat_captions_mask, images
	
	@property
	def model_dir(self):
		return "{}_{}".format( self.model, self.config.dataset )

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

	def save(self, checkpoint_dir, step, note=""):
		if len(note) > 0:
			note = "_"+note
		model_name = self.model+".model"+note
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=step)


	def save_for_vis(self, real_images, generated_images, image_files):
		sample_dir = join(self.sample_dir, 'text2image_'+self.textEncoder)
#		shutil.rmtree( sample_dir )
		if not os.path.exists( sample_dir ):
			os.makedirs( sample_dir )
	
		for i in range(0, real_images.shape[0]):
			real_image_255 = np.zeros( (64,64,3), dtype=np.uint8)
			real_images_255 = (real_images[i,:,:,:])
			scipy.misc.imsave( join(sample_dir, '{}_{}.jpg'.format(i, image_files[i].split('/')[-1] )) , real_images_255)
	
			fake_image_255 = np.zeros( (64,64,3), dtype=np.uint8)
			fake_images_255 = (generated_images[i,:,:,:])
			scipy.misc.imsave(join(sample_dir, 'fake_image_{}.jpg'.format(i)), fake_images_255)


def process_caption( caption, start_word='<S>', end_word='</S>' ):
	tokenized_caption = [start_word]
	tokenized_caption.extend( nltk.tokenize.word_tokenize( caption.lower() ) )
	tokenized_caption.append( end_word )
	return tokenized_caption

def log( epoch, batch, nBatches, lossnames, losses, elapsed, counter=None, filelogger=None ):
	nDigits = len(str(nBatches))
	str_buffer = ''
	assert( len(lossnames) == len(losses) )
	isFirst = True
	for lossname, loss in zip(lossnames,losses):
		if not isFirst:
			str_buffer += ', '
		str_buffer += '({}={:.4f})'.format( lossname, loss )
		isFirst = False

	m,s = divmod( elapsed, 60 )
	h,m = divmod( m,60 )
	timestamp = "{:2}:{:02}:{:02}".format( int(h),int(m),int(s) )
	log = "t{} e:b={}:{}/{} {})".format( timestamp, epoch, batch, nBatches, str_buffer )
	if counter is not None:
		log = "c{} ".format(counter) + log
	print( log )
	if filelogger:
		filelogger.write( log )
	return log

def save_flags( path ):
    flags_dict = tf.flags.FLAGS.__flags
    with open(path, 'w') as f:
        for key,val in flags_dict.iteritems():
            f.write( '{} = {}\n'.format(key,val) )

if __name__ == "__main__":
	tf.app.run()


