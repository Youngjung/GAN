import argparse
import os
import scipy.misc
import numpy as np
import pdb

from models.dcgan import DCGAN
from models.text2image import Text2Image
from models.cyclegan import CycleGAN
from models.utils import pp, visualize, to_json, show_all_variables

import tensorflow as tf

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', default='DCGAN', help='model to use [DCGAN, Text2Image, CycleGAN]')
parser.add_argument('--nEpochs', type=int, default=25, help='# of epoch')
parser.add_argument('--save_every_batch', type=int, default=30, 
						help='Save Model/Samples every x iterations over batches(does not overwrite previously saved models)')
parser.add_argument('--print_every', type=int, default=1, 
						help='print the debug information every print_freq iterations')
parser.add_argument('--batch_size', type=int, default=64, help='# images in batch')
parser.add_argument('--train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', type=str2bool, default=True, help='if flip the images for data argumentation')
parser.add_argument('--which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', default='train', help='train, test')
parser.add_argument('--save_latest_freq', type=int, default=5000, 
						help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--continue_train', type=str2bool, default=False, 
						help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--serial_batches', type=str2bool, default=False, 
						help='f 1, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--serial_batch_iter', type=str2bool, default=True, help='iter into serial image list')
parser.add_argument('--checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', default='./samples', help='sample are saved here')
parser.add_argument('--test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--use_resnet', type=str2bool, default=True, help='generation network using reidule block')
parser.add_argument('--use_lsgan', type=str2bool, default=True, help='gan loss defined in lsgan')
parser.add_argument('--max_size', type=int, default=50, 
						help='max size of image pool, 0 means do not use image pool')

parser.add_argument('--z_dim', type=int, default=100, help="Noise Dimension")
parser.add_argument('--t_dim', type=int, default=256, help='Text feature dimension')
parser.add_argument('--gf_dim', type=int, default=64, help='Number of conv in the first layer gen.')
parser.add_argument('--df_dim', type=int, default=64, help='Number of conv in the first layer discr.')
parser.add_argument('--gfc_dim', type=int, default=1024, help='Dimension of gen untis for fully connected layer [1024]')
parser.add_argument('--dfc_dim', type=int, default=1024, help='Dimension of discrim untis for for fully connected layer [1024]')
parser.add_argument('--caption_vector_length', type=int, default=2400, help='Caption Vector Length')
parser.add_argument('--resume_model', type=str, default=None, help='Pre-Trained Model Path, to resume from')
parser.add_argument('--input_height', type=int, default=108, help="The size of image to use (will be center cropped)")
parser.add_argument('--input_width', type=int, default=None,
						help="The size of image to use (will be center cropped), None=input_height")
parser.add_argument('--output_height', type=int, default=64, help="The size of output image to produce" )
parser.add_argument('--output_width', type=int, default=None, help="The size of output image to produce , None=output_height")
parser.add_argument('--dataset', default='celebA', help="The name of dataset [celebA, mnist, lsun, horse2zebra, mscoco, flowers]")
parser.add_argument('--data_dir', type=str, default="data", help='Data Directory')
parser.add_argument('--input_fname_pattern', default='*.jpg', help="Glob pattern of filename of input images [*]")
parser.add_argument('--crop', type=str2bool, default=True, help="True for training, False for testing [False]")
parser.add_argument('--visualize', type=str2bool,default=False, help="True for visualizing, False for nothing [False]")
#parser.add_argument('--', type=int, default=None, help="")

args = parser.parse_args()

def main(_):

	if args.input_width is None:
		args.input_width = args.input_height
	if args.output_width is None:
		args.output_width = args.output_height

	if not os.path.exists(args.checkpoint_dir):
		os.makedirs(args.checkpoint_dir)
	if not os.path.exists(args.sample_dir):
		os.makedirs(args.sample_dir)
	

	#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
	run_config = tf.ConfigProto()
	run_config.gpu_options.allow_growth=True

	with tf.Session(config=run_config) as sess:
		if args.model == 'DCGAN':
			args.y_dim=None
			if args.dataset == 'mnist':
				args.y_dim=10
			model = DCGAN( sess, args )
		elif args.model == 'Text2Image':
			model = Text2Image( sess, args )
		elif args.model == 'CycleGAN':
			model = CycleGAN( sess, args )
		else:
			raise Exception("undefined model")

		show_all_variables()

		if args.phase=='train':
			model.train()
		else:
			if not model.load(args.checkpoint_dir)[0]:
				raise Exception("[!] Train a model first, then run test mode")

		# Below is codes for visualization
		if args.model == 'DCGAN':
			OPTION = 1
			visualize(sess, model, args, OPTION)
	
if __name__ == '__main__':
	tf.app.run()
