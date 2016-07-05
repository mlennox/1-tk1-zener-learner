# looping a bunch of times...
# pick/load one of symbols
# create image 2 or 3 times the size of the symbol
# add the symbol to the centre
# randomly distort, translate and rotate the image
# crop to the image
# resize to size suitable for neural net input
# save image and metadata in whatever way is required
#  - probably just save to a named folder and have a second program
# to generate the final training data from that

from PIL import Image
from PIL import ImageDraw
from PIL import ImageOps
import getopt
import math
import numpy
import os
import random
import string


def fetch_symbol_images():
	symbols_images = {}
	source_image_path = "../zener-images"

	for root, dirs, files in os.walk(source_image_path):
		for f in files:
			if f.endswith(".png"):
				image_name = string.split(f, ".")
				image = Image.open(source_image_path + "/" + f)
				symbols_images[image_name[0]] = image

	return symbols_images


# https://github.com/nathancahill/snippets/blob/master/image_perspective.py
# pa - starting points
# pb - ending points
# func will find the relevant coeffs that will result in the transformation of pa to pb
# and this will be used to transform the entire image
def find_coeffs(pa, pb):
	matrix = []
	for p1, p2 in zip(pa, pb):
		matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
		matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

	A = numpy.matrix(matrix, dtype=numpy.float)
	B = numpy.array(pb).reshape(8)

	res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
	return numpy.array(res).reshape(8)


def generate_random_shifts(img_size, factor):
	w = img_size[0] / factor
	h = img_size[1] / factor
	shifts = []
	for s in range(0, 4):
		w_shift = (random.random() - 0.5) * w
		h_shift = (random.random() - 0.5) * h
		shifts.append((w_shift, h_shift))
	return shifts


# create random perspective
def create_perspective(img, factor):
	img_size = img.size
	w = img_size[0]
	h = img_size[1]
	shifts = generate_random_shifts(img_size, factor)
	coeffs = find_coeffs(
		[(shifts[0][0], shifts[0][1]),
			(w + shifts[1][0], shifts[1][1]),
			(w + shifts[2][0], h + shifts[2][1]),
			(shifts[3][0], h + shifts[3][1])], [(0, 0), (w, 0), (w, h), (0, h)])
	return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)


# due to rotation and/or perspective we will need to fill in the background
def mask_image(img):
	mask = Image.new("RGBA", img.size, (255, 255, 255, 255))
	return Image.composite(img, mask, img)


# will adjust the canvas so that perspective transforms will not result in the image being cropped
# also adjusts the image to be square along largest dimension - makes things easier later on
# assumes the image background is white...
def adjust_canvas(img, factor):
	padding_factor = 4  # allows more space for image distortion
	width, height = img.size
	# choose largest dimension
	img_largest_dim = (width, height)[width < height]
	canvas_dim = int(math.floor(img_largest_dim + (padding_factor * (img_largest_dim / factor))))
	canvas_size = (canvas_dim, canvas_dim)
	img_pos = (int(math.floor((canvas_size[0] - width) / 2)), int(math.floor((canvas_size[1] - height) / 2)))
	new_canvas = Image.new("RGBA", canvas_size, (255, 255, 255, 255))
	new_canvas.paste(img, (img_pos[0], img_pos[1], img_pos[0] + width, img_pos[1] + height))
	return new_canvas


# will randomly rotate the image

def rotate_image(img, rotation):
	# we want to have random rotations but my feeling is 
	# we should have more smaller rotations than larger
	# this skews the random numbers toward zero
	rotation_factor = math.pow(random.uniform(0.0, 1.0), 4)
	# we want to rotate either way
	rotation_direction = (1, -1)[random.random() > 0.5]
	rotation_angle = int(math.floor(rotation * rotation_factor * rotation_direction))
	return img.rotate(rotation_angle)


# crop the image to a square that bounds the image using largest bounding-box dimension
# and then resize the image to the size desired for the neural net training
def crop_resize(img, dimension):
	inv_img = ImageOps.invert(img.convert("RGB"))
	# returns left, upper, right, lower
	left, upper, right, lower = inv_img.getbbox()
	width = right - left
	height = lower - upper
	if width > height:
		# we want to add half the difference between width and height
		# to the upper and lower dimension
		padding = int(math.floor((width - height) / 2))
		upper -= padding
		lower += padding
	else:
		padding = int(math.floor((height - width) / 2))
		left -= padding
		right += padding

	img = img.crop((left, upper, right, lower))

	# Image.LANCZOS
	# Image.BICUBIC
	return img.resize((dimension, dimension), Image.LANCZOS)


# pulls together all the methods to distort and finalise the image
def distort_image(img, factor, rotation, dimension):
	img = create_perspective(img, factor)
	img = rotate_image(img, rotation)
	img = mask_image(img)
	img = crop_resize(img, dimension)
	return img


# TODO : put these on the command line
# determines how much perspective distortion to use
# factor <= 1 - no distortion
# 1 >= factor <= 5 - slight distortion
# 5 >= factor <= 30 - large but usable distortion
# factor > 30 = very large distortion
perspective_factor = 60
size_factor = 100.0 * (1.0 / perspective_factor)

# specify maximum rotation in degrees
rotation_range = 45

# set the required size of the image sent to the neural net
# image will be cropped/resized to training_dimensionx x training_dimension
training_dimension = 32  # requires 1024 inputs in neural net

# number of images fro Zener symbol
generated_symbols = 10

# load the images
images = fetch_symbol_images()

for symbol_name in images:
	symbol_img = images[symbol_name]
	adjusted_img = adjust_canvas(symbol_img, size_factor)
	for variant in range(1, generated_symbols + 1):
		deformed_image = distort_image(adjusted_img, size_factor, rotation_range, training_dimension)
		generated_folder = '../generated/' + symbol_name + "/"
		if not os.path.exists(generated_folder):
			os.makedirs(generated_folder)
		deformed_image.save(generated_folder + symbol_name + str(variant) + ".png")
