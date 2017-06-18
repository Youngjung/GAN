import tensorflow as tf
import numpy as np
import scipy
import h5py
import ops
import os
import glob
from os.path import join
import image_processing
import random
import shutil
import pdb

class Text2Image:
	'''
	OPTIONS
	z_dim : Noise dimension 100
	t_dim : Text feature dimension 256
	image_size : Image Dimension 64
	gf_dim : Number of conv in the first layer generator 64
	df_dim : Number of conv in the first layer discriminator 64
	gfc_dim : Dimension of gen untis for for fully connected layer 1024
	caption_vector_length : Caption Vector Length 2400
	batch_size : Batch Size 64
	'''
	def __init__(self, sess, options):
		self.textEncoder = options.textEncoder
		self.sess = sess
		self.model = options.model
		self.image_size = options.input_height
		self.batch_size = options.batch_size
		self.z_dim = options.z_dim
		if options.caption_vector_length is not None:
			self.caption_vector_length = options.caption_vector_length
		else:
			if options.textEncoder == 'skipthought':
				self.caption_vector_length = 2400
			elif options.textEncoder == 'DSSJE':
				self.caption_vector_length = 1024
			else:
				raise ValueError('unknown text encoder')
		if options.textEncoder == 'skipthought':
			self.nCapsPerImage = 5
		elif options.textEncoder == 'DSSJE':
			self.nCapsPerImage = 10
		else:
			raise ValueError('unknown text encoder')
		self.t_dim = options.t_dim
		self.gf_dim = options.gf_dim
		self.df_dim = options.df_dim
		self.sample_dir = options.sample_dir
		self.checkpoint_dir = options.checkpoint_dir
		self.nEpochs = options.nEpochs
		self.dataset = options.dataset
		self.lr = options.lr
		self.beta1 = options.beta1
		self.resume_model = options.resume_model
		self.data_dir = options.data_dir
		self.save_every_batch = options.save_every_batch
		self.sample_dir = options.sample_dir

		self.g_bn0 = ops.batch_norm(name='g_bn0')
		self.g_bn1 = ops.batch_norm(name='g_bn1')
		self.g_bn2 = ops.batch_norm(name='g_bn2')
		self.g_bn3 = ops.batch_norm(name='g_bn3')

		self.d_bn1 = ops.batch_norm(name='d_bn1')
		self.d_bn2 = ops.batch_norm(name='d_bn2')
		self.d_bn3 = ops.batch_norm(name='d_bn3')
		self.d_bn4 = ops.batch_norm(name='d_bn4')

		self.input_tensors, self.variables, self.loss, self.outputs, self.checks = self.build_model()

	def build_model(self):
		img_size = self.image_size
		self.t_z = tf.placeholder('float32', [self.batch_size,self.z_dim])
		self.t_real_image = tf.placeholder('float32', [self.batch_size,img_size, img_size, 3 ], name = 'real_image')
		self.t_wrong_image = tf.placeholder('float32', [self.batch_size,img_size, img_size, 3 ], name = 'wrong_image')
		self.t_real_caption = tf.placeholder('float32', [self.batch_size, self.caption_vector_length], name = 'real_caption_input')

		fake_image = self.generator(self.t_z, self.t_real_caption)
		
		disc_real_image, disc_real_image_logits   = self.discriminator(self.t_real_image, self.t_real_caption)
		disc_wrong_image, disc_wrong_image_logits   = self.discriminator(self.t_wrong_image, self.t_real_caption, reuse = True)
		disc_fake_image, disc_fake_image_logits   = self.discriminator(fake_image, self.t_real_caption, reuse = True)
		
		g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake_image_logits, tf.ones_like(disc_fake_image)))
		
		d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real_image_logits, tf.ones_like(disc_real_image)))
		d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_wrong_image_logits, tf.zeros_like(disc_wrong_image)))
		d_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake_image_logits, tf.zeros_like(disc_fake_image)))

		d_loss = d_loss1 + d_loss2 + d_loss3

		t_vars = tf.trainable_variables()
		d_vars = [var for var in t_vars if 'd_' in var.name]
		g_vars = [var for var in t_vars if 'g_' in var.name]

		input_tensors = {
			't_real_image' : self.t_real_image,
			't_wrong_image' : self.t_wrong_image,
			't_real_caption' : self.t_real_caption,
			't_z' : self.t_z
		}

		variables = {
			'd_vars' : d_vars,
			'g_vars' : g_vars
		}

		loss = {
			'g_loss' : g_loss,
			'd_loss' : d_loss
		}

		outputs = {
			'generator' : fake_image
		}

		checks = {
			'd_loss1': d_loss1,
			'd_loss2': d_loss2,
			'd_loss3' : d_loss3,
			'disc_real_image_logits' : disc_real_image_logits,
			'disc_wrong_image_logits' : disc_wrong_image,
			'disc_fake_image_logits' : disc_fake_image_logits
		}
		
		return input_tensors, variables, loss, outputs, checks

	def build_generator(self):
		img_size = self.image_size
		t_real_caption = tf.placeholder('float32', [self.batch_size, self.caption_vector_length], name = 'real_caption_input')
		t_z = tf.placeholder('float32', [self.batch_size, self.z_dim])
		fake_image = self.sampler(t_z, t_real_caption)
		
		input_tensors = {
			't_real_caption' : t_real_caption,
			't_z' : t_z
		}
		
		outputs = {
			'generator' : fake_image
		}

		return input_tensors, outputs

	# Sample Images for a text embedding
	def sampler(self, t_z, t_text_embedding):
		tf.get_variable_scope().reuse_variables()
		
		s = self.image_size
		s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
		
		reduced_text_embedding = ops.lrelu( ops.linear(t_text_embedding, self.t_dim, 'g_embedding') )
		z_concat = tf.concat(1, [t_z, reduced_text_embedding])
		z_ = ops.linear(z_concat, self.gf_dim*8*s16*s16, 'g_h0_lin')
		h0 = tf.reshape(z_, [-1, s16, s16, self.gf_dim * 8])
		h0 = tf.nn.relu(self.g_bn0(h0, train = False))
		
		h1 = ops.deconv2d(h0, [self.batch_size, s8, s8, self.gf_dim*4], name='g_h1')
		h1 = tf.nn.relu(self.g_bn1(h1, train = False))
		
		h2 = ops.deconv2d(h1, [self.batch_size, s4, s4, self.gf_dim*2], name='g_h2')
		h2 = tf.nn.relu(self.g_bn2(h2, train = False))
		
		h3 = ops.deconv2d(h2, [self.batch_size, s2, s2, self.gf_dim*1], name='g_h3')
		h3 = tf.nn.relu(self.g_bn3(h3, train = False))
		
		h4 = ops.deconv2d(h3, [self.batch_size, s, s, 3], name='g_h4')
		
		return (tf.tanh(h4)/2. + 0.5)

	# GENERATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def generator(self, t_z, t_text_embedding):
		
		s = self.image_size
		s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
		
		reduced_text_embedding = ops.lrelu( ops.linear(t_text_embedding, self.t_dim, 'g_embedding') )
		z_concat = tf.concat(1, [t_z, reduced_text_embedding])
		z_ = ops.linear(z_concat, self.gf_dim*8*s16*s16, 'g_h0_lin')
		h0 = tf.reshape(z_, [-1, s16, s16, self.gf_dim * 8])
		h0 = tf.nn.relu(self.g_bn0(h0))
		
		h1 = ops.deconv2d(h0, [self.batch_size, s8, s8, self.gf_dim*4], name='g_h1')
		h1 = tf.nn.relu(self.g_bn1(h1))
		
		h2 = ops.deconv2d(h1, [self.batch_size, s4, s4, self.gf_dim*2], name='g_h2')
		h2 = tf.nn.relu(self.g_bn2(h2))
		
		h3 = ops.deconv2d(h2, [self.batch_size, s2, s2, self.gf_dim*1], name='g_h3')
		h3 = tf.nn.relu(self.g_bn3(h3))
		
		h4 = ops.deconv2d(h3, [self.batch_size, s, s, 3], name='g_h4')
		
		return (tf.tanh(h4)/2. + 0.5)

	# DISCRIMINATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def discriminator(self, image, t_text_embedding, reuse=False):
		if reuse:
			tf.get_variable_scope().reuse_variables()

		h0 = ops.lrelu(ops.conv2d(image, self.df_dim, name = 'd_h0_conv')) #32
		h1 = ops.lrelu( self.d_bn1(ops.conv2d(h0, self.df_dim*2, name = 'd_h1_conv'))) #16
		h2 = ops.lrelu( self.d_bn2(ops.conv2d(h1, self.df_dim*4, name = 'd_h2_conv'))) #8
		h3 = ops.lrelu( self.d_bn3(ops.conv2d(h2, self.df_dim*8, name = 'd_h3_conv'))) #4
		
		# ADD TEXT EMBEDDING TO THE NETWORK
		reduced_text_embeddings = ops.lrelu(ops.linear(t_text_embedding, self.t_dim, 'd_embedding'))
		reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,1)
		reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,2)
		tiled_embeddings = tf.tile(reduced_text_embeddings, [1,4,4,1], name='tiled_embeddings')
		
		h3_concat = tf.concat( 3, [h3, tiled_embeddings], name='h3_concat')
		h3_new = ops.lrelu( self.d_bn4(ops.conv2d(h3_concat, self.df_dim*8, 1,1,1,1, name = 'd_h3_conv_new'))) #4
		
		h4 = ops.linear(tf.reshape(h3_new, [self.batch_size, -1]), 1, 'd_h3_lin')
		
		return tf.nn.sigmoid(h4), h4

	def train(self):
		input_tensors = self.input_tensors
		loss = self.loss
		outputs = self.outputs
		variables = self.variables
		checks = self.checks

		d_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(loss['d_loss'], var_list=variables['d_vars'])
		g_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(loss['g_loss'], var_list=variables['g_vars'])
	
		tf.global_variables_initializer().run()
	
		self.saver = tf.train.Saver()
		if self.resume_model:
			self.saver.restore(self.sess, self.resume_model)
	
		loaded_data = self.load_training_data(self.data_dir, self.dataset)
	
		nBatch = loaded_data['data_length']/self.batch_size
		for i in range(self.nEpochs):
			batch_no = 0
			while batch_no*self.batch_size < loaded_data['data_length']:
				real_images, wrong_images, caption_vectors, z_noise, image_files = self.get_training_batch(batch_no, self.batch_size,
					self.image_size, self.z_dim, self.caption_vector_length, 'train', self.data_dir, self.dataset, loaded_data)
	
				# DISCR UPDATE
				check_ts = [ checks['d_loss1'] , checks['d_loss2'], checks['d_loss3']]
				_, d_loss, gen, d1, d2, d3 = self.sess.run([d_optim, loss['d_loss'], outputs['generator']] + check_ts,
					feed_dict = {
						input_tensors['t_real_image'] : real_images,
						input_tensors['t_wrong_image'] : wrong_images,
						input_tensors['t_real_caption'] : caption_vectors,
						input_tensors['t_z'] : z_noise,
					})
	
	
				# GEN UPDATE
				_, g_loss, gen = self.sess.run([g_optim, loss['g_loss'], outputs['generator']],
					feed_dict = {
						input_tensors['t_real_image'] : real_images,
						input_tensors['t_wrong_image'] : wrong_images,
						input_tensors['t_real_caption'] : caption_vectors,
						input_tensors['t_z'] : z_noise,
					})
	
				# GEN UPDATE TWICE, to make sure d_loss does not go to 0
				_, g_loss, gen = self.sess.run([g_optim, loss['g_loss'], outputs['generator']],
					feed_dict = {
						input_tensors['t_real_image'] : real_images,
						input_tensors['t_wrong_image'] : wrong_images,
						input_tensors['t_real_caption'] : caption_vectors,
						input_tensors['t_z'] : z_noise,
					})
				print( "epoch {} batch {}/{} (d1,d2,d3)=({:.4f}, {:.4f}, {:.4f}) (D,G)=({:.4f}, {:.4f})".format( \
								 i, batch_no, nBatch, d1, d2, d3, d_loss, g_loss ) )
				batch_no += 1
				if (batch_no % self.save_every_batch) == 0:
					print( "Saving Images, Model" )
					self.save_for_vis(real_images, gen, image_files)
					self.save(self.checkpoint_dir, batch_no)
			if i%5 == 0:
				self.save(self.checkpoint_dir, batch_no, note="epoch{}".format(i))
	
	def load_training_data(self, data_dir, dataset):
		if dataset == 'flowers':
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
			if self.textEncoder == 'DSSJE':
				print( 'loading DSSJE...' )
				data_icml_dir = join(data_dir,'flowers_caption_icml')
				full_filenames = {}
				classnames = os.listdir( data_icml_dir )
				for classname in classnames:
					if os.path.isdir( join(data_icml_dir,classname) ) :
						filenames = glob.glob( join(data_icml_dir,classname,'*.npy') )
						for filename in filenames:
							full_filenames[os.path.basename(filename[0:-4])+'.jpg'] = filename
				flower_captions_reedscot = {}
				for filename, npy in full_filenames.iteritems():
					flower_captions_reedscot[filename] = np.load( npy )
				flower_captions = flower_captions_reedscot
	

			return {
				'image_list' : training_image_list,
				'captions' : flower_captions,
				'data_length' : len(training_image_list)
			}
	
		else:
			with open(join(data_dir, 'meta_train.pkl')) as f:
				meta_data = pickle.load(f)
			# No preloading for MS-COCO
			return meta_data

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
	
	
	def get_training_batch(self, batch_no, batch_size, image_size, z_dim,
		caption_vector_length, split, data_dir, dataset, loaded_data = None):
		if dataset == 'mscoco':
			with h5py.File( join(data_dir, 'tvs/'+split + '_tvs_' + str(batch_no))) as hf:
				caption_vectors = np.array(hf.get('tv'))
				caption_vectors = caption_vectors[:,0:caption_vector_length]
			with h5py.File( join(data_dir, 'tvs/'+split + '_tv_image_id_' + str(batch_no))) as hf:
				image_ids = np.array(hf.get('tv'))
	
			real_images = np.zeros((batch_size, 64, 64, 3))
			wrong_images = np.zeros((batch_size, 64, 64, 3))
	
			image_files = []
			for idx, image_id in enumerate(image_ids):
				image_file = join(data_dir, '%s2014/COCO_%s2014_%.12d.jpg'%(split, split, image_id) )
				image_array = image_processing.load_image_array(image_file, image_size)
				real_images[idx,:,:,:] = image_array
				image_files.append(image_file)
	
			# TODO>> As of Now, wrong images are just shuffled real images.
			first_image = real_images[0,:,:,:]
			for i in range(0, batch_size):
				if i < batch_size - 1:
					wrong_images[i,:,:,:] = real_images[i+1,:,:,:]
				else:
					wrong_images[i,:,:,:] = first_image
	
			z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])
	
	
			return real_images, wrong_images, caption_vectors, z_noise, image_files
	
		if dataset == 'flowers':
			real_images = np.zeros((batch_size, 64, 64, 3))
			wrong_images = np.zeros((batch_size, 64, 64, 3))
			captions = np.zeros((batch_size, caption_vector_length))
	
			cnt = 0
			image_files = []
			for i in range(batch_no * batch_size, batch_no * batch_size + batch_size):
				idx = i % len(loaded_data['image_list'])
				image_file =  join(data_dir, 'flowers/jpg/'+loaded_data['image_list'][idx])
				image_array = image_processing.load_image_array(image_file, image_size)
				real_images[cnt,:,:,:] = image_array
	
				# Improve this selection of wrong image
				wrong_image_id = random.randint(0,len(loaded_data['image_list'])-1)
				wrong_image_file =  join(data_dir, 'flowers/jpg/'+loaded_data['image_list'][wrong_image_id])
				wrong_image_array = image_processing.load_image_array(wrong_image_file, image_size)
				wrong_images[cnt, :,:,:] = wrong_image_array
	
				random_caption = random.randint(0,self.nCapsPerImage-1) # randint arguments are inclusive
				captions[cnt,:] = loaded_data['captions'][ loaded_data['image_list'][idx] ][ random_caption ][0:caption_vector_length]
				image_files.append( image_file )
				cnt += 1
	
			z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])
			return real_images, wrong_images, captions, z_noise, image_files
	
	@property
	def model_dir(self):
		return "{}_{}_{}_{}".format( self.model, self.dataset, self.batch_size, self.image_size, self.image_size)
