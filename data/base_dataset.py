## TODO SEP 08 VERSION

"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from scipy.ndimage import rotate
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import skimage.transform as transform
import torch
import math
import cv2



class BaseDataset(data.Dataset, ABC):
	"""This class is an abstract base class (ABC) for datasets.

	To create a subclass, you need to implement the following four functions:
	-- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
	-- <__len__>:                       return the size of dataset.
	-- <__getitem__>:                   get a data point.
	-- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
	"""

	def __init__(self, opt):
		"""Initialize the class; save the options in the class

		Parameters:
			opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
		"""
		self.opt = opt
		self.root = opt.dataroot

	@staticmethod
	def modify_commandline_options(parser, is_train):
		"""Add new dataset-specific options, and rewrite default values for existing options.

		Parameters:
			parser          -- original option parser
			is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

		Returns:
			the modified parser.
		"""
		return parser

	@abstractmethod
	def __len__(self):
		"""Return the total number of images in the dataset."""
		return 0

	@abstractmethod
	def __getitem__(self, index):
		"""Return a data point and its metadata information.

		Parameters:
			index - - a random integer for data indexing

		Returns:
			a dictionary of data with their names. It ususally contains the data itself and its metadata information.
		"""
		pass

# TODO change this to pick just one pair of 2D slices.
def get_params(opt, vol_shape):
	crop_z, crop_y, crop_x = opt.crop_size

	assert (vol_shape[0] - crop_z >= 0)
	assert (vol_shape[1] - crop_y >= 0)
	assert (vol_shape[2] - crop_x >= 0)

	z = random.randint(0, np.maximum(0, vol_shape[0] - crop_z))
	y = random.randint(0, np.maximum(0, vol_shape[1] - crop_y))
	x = random.randint(0, np.maximum(0, vol_shape[2] - crop_x))

	flip_axis = np.random.randint(0,3)

	angle_3D = random.randint(0, 359)

	return {'crop_pos': (z, y, x), 'flip_axis': flip_axis, 'angle_3D':angle_3D}

def get_transform(opt, params = None):
	transform_list = []
	image_dimension = int(opt.image_dimension)

	if 'random3Drotate' in opt.preprocess:
		if params is None:
			transform_list += [transforms.Lambda(lambda img_np: __randomrotate_clean_3D_xy(img_np))]
		else:
			transform_list += [transforms.Lambda(lambda img_np: __rotate_clean_3D_xy(img_np, angle=params['angle_3D']))]

	if 'randomrotate' in opt.preprocess:
		if params is None:
			transform_list += [transforms.Lambda(lambda img_np: __randomrotate(img_np))]
		else:
			transform_list += [transforms.Lambda(lambda img_np: __rotate(img_np, params['rotate_params']))]

	if 'randomcrop' in opt.preprocess:
		if params is None:
			transform_list += [transforms.Lambda(lambda img_np: __randomcrop(img_np, opt.crop_size))]
		else:
			transform_list += [transforms.Lambda(lambda img_np: __crop(img_np, params['crop_pos'], opt.crop_size))]

	if 'centercrop' in opt.preprocess:
		transform_list += [transforms.Lambda(lambda img_np: __centercrop(img_np, opt.crop_portion))]

	transform_list += [transforms.Lambda(lambda img_np: __normalize(img_np, opt.img_params))]

	if 'randomflip' in opt.preprocess:
		if params is None:
			transform_list+= [transforms.Lambda(lambda  img_np: __randomflip(img_np))]
		else:
			transform_list+= [transforms.Lambda(lambda  img_np: __flip(img_np, params['flip_axis']))]

	if 'addColorChannel' in opt.preprocess:
		transform_list += [transforms.Lambda(lambda  img_np:__addColorChannel(img_np))]

	if 'reorderColorChannel' in opt.preprocess:
		transform_list += [transforms.Lambda(lambda  img_np:__reorderColorChannel(img_np))]

	if 'addBatchChannel' in opt.preprocess:
		transform_list += [transforms.Lambda(lambda  img_np:__addColorChannel(img_np))]

	transform_list += [transforms.Lambda(lambda  img_np:__toTensor(img_np))]

	return transforms.Compose(transform_list)

# normalize to 0-1 range. Note that mean and std. are calculated as scaled on 0-1 scale.
def __normalize(img_np, img_params = None):
	if img_np.dtype == 'uint8':
		img_normd = (img_np / (2**8*1.0 - 1)).astype(float)
		mean, std = img_params
		# mean = (mean / (2 ** 8 * 1.0 - 1)).astype(float)
		# std = (std / (2 ** 8 * 1.0 - 1)).astype(float)
		img_normd = (img_normd - mean) / std

	elif img_np.dtype == 'uint16':
		img_normd = (img_np / (2**16*1.0 - 1)).astype(float)
		mean, std = img_params
		# mean = (mean / (2**16*1.0 - 1)).astype(float)
		# std = (std / (2**16*1.0 - 1)).astype(float)
		img_normd = (img_normd - mean) / std

	else:
		assert "Image type is not recognized."
	return img_normd

def __randomrotate(img_np):
	# random_angle = np.random.randint(0, 90)
	random_angle = np.random.choice((-90,90,-180,180,-270,270))
	if len(img_np.shape) > 2: # 3D data
		random_axis = tuple(np.random.choice(3, 2, replace = False))
	elif len(img_np.shape) == 2: #2D data
		random_axis = (0,1)
	img_np_rotated = rotate(img_np, random_angle, axes = random_axis, reshape = False, mode = 'reflect')
	return img_np_rotated

def __rotate(img_np, rotate_params):
	angle, axis = rotate_params
	img_np_rotated = rotate(img_np, angle, axes = axis, reshape = False, mode = 'reflect')
	return img_np_rotated

def __permutate(img_np):
	axis_len = len(img_np.shape)
	axes = np.arange(axis_len)
	np_list = []
	np_list.append(img_np)
	for axis in axes:
		img_np_flipped = np.flip(img_np, axis)
		np_list.append(img_np_flipped)

	img_np = np.stack(np_list, axis=0)

	return img_np

def __randomcontrast(img_np, randomcontrast_val): # randomly change the contrast of the image (by constast-stretch)
	random_contrast = random.randint(randomcontrast_val, 99) # by default, randomcontrast_val = 50
	img_min = np.min(img_np)
	img_max = np.max(img_np)

	top_val = np.percentile(img_np, random_contrast)
	img_processed = np.clip(img_np, top_val, None)

	if img_max == top_val:
		img_normed = img_np
	else:
		img_normed = (img_processed - top_val) * ((img_max - img_min) / (img_max - top_val)) + img_min
	# img_normed = img_normed.astype('uint16')

	return img_normed

def __randomcrop(img_np, crop_size):

	if len(img_np.shape) > 2: # 3D data
		crop_z, crop_y, crop_x = crop_size
		assert (img_np.shape[0] - crop_z >= 0)
		assert (img_np.shape[1] - crop_y >= 0)
		assert (img_np.shape[2] - crop_x >= 0)

		z = random.randint(0, img_np.shape[0] - crop_z)
		y = random.randint(0, img_np.shape[1] - crop_y)
		x = random.randint(0, img_np.shape[2] - crop_x)

		if crop_x == 0:
			x_reach = None
			x = 0
		else:
			x_reach = x + crop_x
		if crop_y == 0:
			y_reach = None
			y = 0
		else:
			y_reach = y + crop_y

		if crop_z == 0:
			z_reach = None
			z = 0
		else:
			z_reach = z + crop_z

		img_cropped = img_np[z:z_reach, y:y_reach, x:x_reach]

	elif len(img_np.shape) == 2: # 2D data
		crop_y, crop_x = crop_size  # For 2D, crop_z will be ignored.

		assert (img_np.shape[0] - crop_y >= 0)
		assert (img_np.shape[1] - crop_x >= 0)

		y = random.randint(0, img_np.shape[0] - crop_y)
		x = random.randint(0, img_np.shape[1] - crop_x)

		if crop_y == 0:
			y_reach = None
			y = 0
		else:
			y_reach = y + crop_y
		if crop_x == 0:
			x_reach = None
			x = 0
		else:
			x_reach = x + crop_x

		img_cropped = img_np[y:y_reach, x:x_reach]

	return img_cropped

def __reorderColorChannel (img_np):
	# re-order the order so that y, x, c -> c, y, x
	img_np = np.swapaxes(img_np, 0, 2) #y, x, c -> c, x, y
	img_np = np.swapaxes(img_np, 1, 2) #c, x, y -> c, y, x
	return img_np


def __centercrop(img_np, crop_portion):
	crop_portion = (100 - crop_portion*1.0) / 100 # crop_portion of 90% means cropping out 10%.

	if len(img_np.shape) > 2: # 3D data
		z, y, x = img_np.shape
		crop_z, crop_y, crop_x = int(z * crop_portion / 2), int(y * crop_portion / 2), int(x * crop_portion / 2)  # For 2D, crop_z will be ignored.
		img_cropped = img_np[crop_z:-crop_z, crop_y:-crop_y, crop_x:-crop_x]

	else: # 2D data
		y, x = img_np.shape
		crop_y, crop_x = int(y * crop_portion / 2), int(x * crop_portion / 2)
		img_cropped = img_np[crop_y:-crop_y, crop_x:-crop_x]

	return img_cropped

def __crop(img_np, pos, crop_size):
	z, y, x = pos
	crop_z, crop_y, crop_x = crop_size
	img_cube = img_np[z:z + crop_z, y:y + crop_y, x:x + crop_x]
	return img_cube

def __flip(img_np, axis):
	img_np_flipped = np.flip(img_np, axis)
	return img_np_flipped

def __randomgamma(img_np):
	gamma_val = np.random.uniform(1.0, 1.5)
	img_np_gammad = (img_np) ** (1/gamma_val)
	return img_np_gammad

def __randomflip(img_np):
	axis_len = len(img_np.shape)
	axis_list = list(range(axis_len))
	random.shuffle(axis_list)
	img_np_flipped = img_np
	for i in range(axis_len):
		chance = np.random.uniform(0, 1)
		if chance < 0.5:
			axis = axis_list.pop()
			img_np_flipped = np.flip(img_np_flipped, axis)
	return img_np_flipped

def __toTensor(img_np):
	img_np = img_np.astype(float)
	assert img_np.dtype == np.float
	img_tensor = torch.from_numpy(img_np).float()
	return img_tensor

# add a color channel to the grayscale numpy image.
def __addColorChannel(img_np):
	img_np = np.expand_dims(img_np, axis=0)
	# print (img_np.shape)
	return img_np

### Clean rotation
# Ref: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders

def rotate_image(image, angle):
	"""
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

	# Get the image size
	# No that's not an error - NumPy stores image matricies backwards
	image_size = (image.shape[1], image.shape[0])
	image_center = tuple(np.array(image_size) / 2)

	# Convert the OpenCV 3x2 rotation matrix to 3x3
	rot_mat = np.vstack(
		[cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
	)

	rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

	# Shorthand for below calcs
	image_w2 = image_size[0] * 0.5
	image_h2 = image_size[1] * 0.5

	# Obtain the rotated coordinates of the image corners
	rotated_coords = [
		(np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
		(np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
		(np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
		(np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
	]

	# Find the size of the new image
	x_coords = [pt[0] for pt in rotated_coords]
	x_pos = [x for x in x_coords if x > 0]
	x_neg = [x for x in x_coords if x < 0]

	y_coords = [pt[1] for pt in rotated_coords]
	y_pos = [y for y in y_coords if y > 0]
	y_neg = [y for y in y_coords if y < 0]

	right_bound = max(x_pos)
	left_bound = min(x_neg)
	top_bound = max(y_pos)
	bot_bound = min(y_neg)

	new_w = int(abs(right_bound - left_bound))
	new_h = int(abs(top_bound - bot_bound))

	# We require a translation matrix to keep the image centred
	trans_mat = np.matrix([
		[1, 0, int(new_w * 0.5 - image_w2)],
		[0, 1, int(new_h * 0.5 - image_h2)],
		[0, 0, 1]
	])

	# Compute the tranform for the combined rotation and translation
	affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

	# Apply the transform
	result = cv2.warpAffine(
		image,
		affine_mat,
		(new_w, new_h),
		flags=cv2.INTER_LINEAR
	)

	return result


def largest_rotated_rect(w, h, angle):
	"""
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

	quadrant = int(math.floor(angle / (math.pi / 2))) & 3
	sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
	alpha = (sign_alpha % math.pi + math.pi) % math.pi

	bb_w = w * math.cos(alpha) + h * math.sin(alpha)
	bb_h = w * math.sin(alpha) + h * math.cos(alpha)

	gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

	delta = math.pi - alpha - gamma

	length = h if (w < h) else w

	d = length * math.cos(alpha)
	a = d * math.sin(alpha) / math.sin(delta)

	y = a * math.cos(gamma)
	x = y * math.tan(gamma)

	return (
		bb_w - 2 * x,
		bb_h - 2 * y
	)


def crop_around_center(image, width, height):
	"""
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

	image_size = (image.shape[1], image.shape[0])
	image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

	if (width > image_size[0]):
		width = image_size[0]

	if (height > image_size[1]):
		height = image_size[1]

	x1 = int(image_center[0] - width * 0.5)
	x2 = int(image_center[0] + width * 0.5)
	y1 = int(image_center[1] - height * 0.5)
	y2 = int(image_center[1] + height * 0.5)

	return image[y1:y2, x1:x2]

def __rotate_clean(image, angle):
	image_height, image_width = image.shape

	image_rotated = rotate_image(image, angle)
	image_rotated_cropped = crop_around_center(image_rotated, *largest_rotated_rect(
		image_width,
		image_height,
		math.radians(angle)
	))

	return image_rotated_cropped

def __rotate_clean_3D_xy(image_vol, angle):
	slice_list = []
	for slice in image_vol:
		slice_rotated = __rotate_clean(slice, angle)
		slice_list.append(slice_rotated)
	img_vol_rotated = np.array(slice_list)
	return img_vol_rotated

def __randomrotate_clean_3D_xy(image_vol):
	angle = random.randint(0, 359)
	slice_list = []
	for slice in image_vol:
		slice_rotated = __rotate_clean(slice, angle)
		slice_list.append(slice_rotated)
	img_vol_rotated = np.array(slice_list)
	return img_vol_rotated
