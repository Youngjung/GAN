from torch.utils.serialization import load_lua
import os
import numpy as np
import pdb

icml_root = '/home/cvpr-gb/youngjung/GAN/data/flowers_caption_icml'

classnames = os.listdir( icml_root )
for classname in classnames:
	print( 'running in {}...'.format(classname) )
	classpath = os.path.join(icml_root,classname)
	if os.path.isdir( classpath ):
		filenames = os.listdir( classpath )
		for filename in filenames:
			if filename[-3:] == 'npy':
				continue
			t7 = os.path.join( icml_root, classname, filename )
			data = load_lua( t7 )
			txt = data['txt'].numpy()
			np.save( t7[:-3], txt )
	
