#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import scipy.stats as sci
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
from matplotlib.widgets import RectangleSelector

class Pic():
	def __init__(self, jpg):
		self.orig = mpimg.imread(jpg)
	
	def plot(self, jpg, title):
		"""Plot original picture
		
		Args:
			jpg: (numpy array) original 3D picture
			title: (string) image title
		"""
		self.jpg = jpg
		self.title = title
		
		plt.figure()
		plt.imshow(self.jpg)
		plt.title(self.title)
		plt.show()
		
	def reshape_2D(self, jpg_3D):
		"""Reshape a 3D image into a 2D one
		
		Args:
			jpg_3D: (numpy array) original 3D picture
			
		Returns:
			_2D: (numpy array) original 3D picture reshaped into the 2D one
		"""
		self._3D = jpg_3D
		global r
		global g
		global b
		
		r,g,b = self._3D.shape
		self._2D = self._3D.reshape((r*g, b))
		
		return self._2D
	
	def quantize(self, jpg, ncluster):
		"""Quantize a picture by applying kmeans
		
		Args:
			jpg: (numpy array) 2D picture
			ncluster: (int) number of clusters to recognize
		
		Returns: 
			quant: (numpy array) quantized picture
		"""
		self.jpg = jpg
		self.clusters = ncluster
		global kmeans
		
		kmeans = KMeans(n_clusters = self.clusters, random_state = 0)
		print ("\n>> performing kmeans...")
		kmeans.fit(self.jpg)
		self.kmeans_centroids = kmeans.cluster_centers_.astype('uint8')
		self.quant_2D = self.jpg.copy()
		
		for kc in range(self.clusters):
			quant_color=self.kmeans_centroids[kc,:]
			index=(kmeans.labels_==kc)
			self.quant_2D[index,:]=quant_color
		self.quant=self.quant_2D.reshape((r,g,b))
		
		return self.quant
	
	def isolate(self, centroids):
		"""Allow the manual selection of the mole through the mouse click
		
		Args:
			centroids: (numpy array) array containing the picture colors centroids
		
		Returns:
			subset: (numpy array) isolated mole picture
		"""
		self.x1 = int(x1)
		self.x2 = int(x2)
		self.y1 = int(y1)
		self.y2 = int(y2)
		self.centroids = centroids
		global dark_index

		sc = np.sum(self.centroids, axis=1)
		dark_index = sc.argmin()
		
		
		# im_sel is a boolean NDarray with N1 rows and N2 columns
		im_sel=(kmeans.labels_.reshape(r,g)==dark_index)
		# im_sel is now an integer NDarray with N1 rows and N2 columns
		im_sel=im_sel*1
				
		subset = im_sel[self.y1:self.y2, self.x1:self.x2]
		
		return subset
	
	def save_data(self, data, title):
		"""Save the subset into a dat file
		
		Args:
			data: (numpy array) picture subset
			title: (string)
		"""
		df = pd.DataFrame.from_records(data)
		df.to_csv(title+".dat")
	
	def save_table(self, entry, ratio, title):
		"""Save the perimeter ratio into a latex table file
		
		Args:
			entry: (string) name of the file
			ratio: (float) perimeter ratio
			title: (string) filename
		"""
		entry = entry.split(".jpg")
		if not os.path.isfile(title):
			with open(title, "w") as file:
				file.write("\\begin{tabular}{lr}\hline\n{} &  Perimeter Ratio\\\\\hline\n"+ 
				           entry[0]+"\t&\t"+str(ratio)+"\\\\\n")
		else:
			with open(title, "a") as file:
				file.write(entry[0]+"\t&\t"+str(ratio)+"\\\\\n")

"""Cleaning and contour finding algorithm"""
		
def clean(data, ray):
	"""Clean the bicolor picture of the mole by filling empty pixels
	or by deleting extra pixels
	
	Args:
		data: (numpy array) the matrix contains only 1s and 0s according 
		      to the presence/absence of the color
		ray: (int) ray of the pixel centered matrix which compares the surrounding
		     pixels
	
	Returns:
		clean_data: (numpy array) same matrix as data, but the picture is cleaned
	"""
	
	n_row = np.size(data, 0)
	n_col = np.size(data, 1)

	clean_data = np.zeros((n_row, n_col), dtype=int)
	classes = []

	for i_1 in range(n_row):
		for j_1 in range(n_col):
			x_1 = [i_1, j_1]
			classes = []
			for i_2 in range(i_1-ray, i_1+ray+1):
				for j_2 in range(j_1-ray, j_1+ray+1):
					if(j_2 >= 0 and i_2<n_row and j_2<n_col):
						x_2 = [i_2, j_2]
						if x_2!=x_1:
							classes.append(data[i_2, j_2])
			mode = sci.mode(classes)
			clean_data[i_1, j_1] = mode[0]
	
	return clean_data

def contour_manager(data):
	"""Randomly enter the picture from the left edge, detect the first
	contour pixel and call the direction manager functions to search and 
	follow the mole contour.
	
	Args:
		data: (numpy array) bicolor cleaned mole picture
	
	Returns:
		data: (numpy array) cleaned mole picture with the contour marked
			  with a third color
	"""
	global START_0
	global START_1
	n_col = np.size(data, 1)
	n_row = np.size(data, 0)
	
	# enter the mole
	for i in range(n_col):
		if data[n_row/2, i] == 1:
			data[n_row/2, i] == 3
			START_0 = n_row/2
			START_1 = i
			break

	row = n_row/2
	col = i
	direction = determine_direction(data, row, col)
	row, col = apply_direction(direction, row, col)
	
	# find the contour until the starting pixel is not encountered
	while(row!=START_0 or col!=START_1):
		direction = determine_direction(data, row, col)
		row, col = apply_direction(direction, row, col)
	
	return data

def determine_direction(data, start_row, start_col):
	"""Create a shifting matrix centered in a contour pixel.
	If the matrix has been learned in the past by the software
	(the matrix has been written in the moves array), the direction
	is automatically applied. Otherwise the user must specify it and
	the new information is written in the moves array.
	
	Args:
		data: (numpy array) cleaned mole picture
		start_row: (int) row index of the matrix center
		start_col: (int) column index of the matrix center
	
	Returns:
		code: (int) encoded direction to move the matrix
	"""
	data[start_row, start_col] = 3
	# create the shifting matrix
	matrix = data[np.ix_([start_row-1, start_row,start_row+1],
						[start_col-1, start_col,start_col+1])]
	# "open" the matrix horizontally
	open_matrix = np.array([ matrix[0, 0],
							 matrix[0, 1],
							 matrix[0, 2],
							 matrix[1, 0],
							 matrix[1, 2],
							 matrix[2, 0],
							 matrix[2, 1],
							 matrix[2, 2]])
	
	# training files creation
	if not os.path.isdir("train"):
		print matrix
		os.mkdir("train")
		
		direction = raw_input("direction: ")
		code = encode_direction(direction)
		to_save = np.hstack((open_matrix,code))
		np.save("train/train", to_save)
	
	else:
		learnt = False
		# check if the matrix is in the training file
		from_file = np.load("train/train.npy")
		try:
			np.size(from_file, 1)
		except IndexError:
			open_train = np.delete(from_file, 8)
			if np.array_equal(open_train, open_matrix):
				learnt = True
				code = from_file[8]
		else:
			for index in range(np.size(from_file, 0)):
				open_train = np.delete(from_file[index, :], 8)
				if np.array_equal(open_train, open_matrix):
					learnt = True
					code = from_file[index, 8]
					break
		
		# the matrix is not present
		if not learnt:
			print matrix
			plt.imshow(data)
			plt.show()
			# manually train the software
			direction = raw_input("direction: ")
			code = encode_direction(direction)
			to_save = np.hstack((open_matrix,code))
			to_save = np.vstack((from_file, to_save))
			np.save("train/train", to_save)

	return code

def encode_direction(cmd):
	"""Encode the direction of the shifting matrix to save it as
	the last element of the moves array
	
	Args:
		cmd: (string) manually inserted direction
	
	Returns:
		code: (int) encoded direction
	"""
	if cmd == "down":
		code = 0
	elif cmd == "downright":
		code = 1
	elif cmd == "right":
		code = 2
	elif cmd == "upright":
		code = 3
	elif cmd == "up":
		code = 4
	elif cmd == "upleft":
		code = 5
	elif cmd == "left":
		code = 6
	else:
		code = 7
	
	return code

def apply_direction(direction, start_row, start_col):
	"""Move the shifting matrix along the contour on the basis of
	the learned moves.
	
	Args:
		direction: (int) encoded direction found in the moves array
		start_row: (int) row index of the matrix center
		start_col: (int) column index of the matrix center 
	
	Returns:
		start_row: (int) shifted row index of the new matrix center
		start_col: (int) shifted column index of the new matrix center
	"""
	if direction == 4: # up
		start_row -= 1
	elif direction == 3: # upright
		start_row -= 1
		start_col += 1
	elif direction == 2: # right
		start_col += 1
	elif direction == 1: # downright
		start_row += 1
		start_col += 1
	elif direction == 0: # down
		start_row += 1
	elif direction == 7: # downleft
		start_row += 1
		start_col -= 1
	elif direction == 6: # left
		start_col -= 1
	elif direction == 5: # upleft
		start_row -= 1
		start_col -= 1
	
	return (start_row, start_col)
				
def fill_inside(data):
	"""Scan the contour-marked mole picture and fill all the holes
	inside the mole
	
	Args:
		data: (numpy array) picture to fill
	
	Returns:
		data: (numpy array) filled picture
	"""
	n_row = np.size(data, 0)
	n_col = np.size(data, 1)
	
	# from left to right
	for i in range(n_row):
		inside = False
		for j in range(n_col):
			if data[i, j] == 3:
				if data[i, j-1] != 3:
					buff = data[i, j-1]
				if data[i, j+1] !=3:
					if data[i, j+1] == 0 and buff == 1:
						inside = False
					elif data[i, j+1] == 1 and buff == 0:
						inside = True
					elif data[i, j+1] == 1 and buff == 1:
						inside = True
					elif data[i, j+1] == 0 and buff == 0:
						temp_check = np.array([data[i, j+2],
											   data[i, j+3],
											   data[i, j+4],
											   data[i, j+5],
											   data[i, j+6]])
											   
						m = sci.mode(temp_check)
						if m[0] == 1:
							inside = True
						elif m[0] == 0:
							inside = False
					
			if data[i, j] == 0 and inside:
				data [i, j] = 1
			if data[i, j] == 1 and inside == False:
				data [i, j] = 0
	
	# from right to left
	for i in range(n_row):
		inside = False
		for j in range(n_col):
			if n_col-j+1<n_col:
				if data[i, n_col-j] == 3:
					if data[i, n_col-j+1]!=3:
						dubb = data[i, n_col-j+1]
					if data[i, n_col-j-1]!=3:
						if data[i, n_col-j-1] == 0 and buff == 1:
							inside = False
						elif data[i, n_col-j-1] == 1 and buff == 0:
							inside = True
						elif data[i, n_col-j-1] == 1 and buff == 1:
							inside = True
						elif data[i, n_col-j-1] == 0 and buff == 0:
							inside = False					
				if data[i, n_col-j] == 0 and inside:
					data [i, n_col-j] = 1
				if data[i, n_col-j] == 1 and inside == False:
					data [i, n_col-j] = 0
	
			
	plt.imshow(data)
	plt.show()
	
	return data

def get_ratio(data):
	"""Determine the moles perimeter and area, then compute the 
	ratio between the mole perimeter and the one of a circle having
	the same area of the mole
	
	Args:
		data: (numpy array) final mole picture
	
	Returns:
		size_ratio: (float) perimeters ratio
	"""
	n_row = np.size(data, 0)
	n_col = np.size(data, 1)
	area = np.count_nonzero(data) # mole area
	
	# contour only
	for i in range(n_row):
		for j in range(n_col):
			if data[i, j]!=3:
				data[i, j]=0
			else:
				data[i, j]=1
	
	perimeter = np.count_nonzero(data) # mole perimeter
	
	# perimeter of a circle with the same area of the mole
	circle_perimeter = np.sqrt(area/np.pi)*2*np.pi
	
	# ratio between the two perimeters
	size_ratio = perimeter/circle_perimeter
	
	return size_ratio
	
	
def line_select_callback(eclick, erelease):
	"""Callback function used to determine the manually selected
	mole area of the picture
	
	Args:
		eclick: (matplotlib object) coordinates of the mouse click point
		erelease: (matplotlib object) coordinates of the mouse release point
	"""
	global x1, y1, x2, y2

	x1, y1 = eclick.xdata, eclick.ydata
	x2, y2 = erelease.xdata, erelease.ydata
		
	
				
# main
RAY = 4
		
#np.set_printoptions(precision=2,threshold=np.nan)
plt.close('all')

for item in os.listdir("data/"):
	original = "data/"+item
	print ("\n>> processing "+item)
	
	# get the original picture
	fig = Pic(original)
	fig.plot(fig.orig, "original_figure")

	# quantize picture
	fig_2D = fig.reshape_2D(fig.orig)
	quant_2D = fig.quantize(fig_2D, 3)
	
	# isolate mole
	fig_temp, current_ax = plt.subplots() 
	plt.imshow(quant_2D)
	RS = RectangleSelector( current_ax,
							line_select_callback,
							drawtype='box', useblit=True,
							button=[1, 3],  # don't use middle button
							minspanx=5, minspany=5,
							spancoords='pixels',
							interactive=True)
	plt.title("select the mole and close the figure.")
	plt.show()
	
	subset = fig.isolate(fig.kmeans_centroids)
	plt.title("subset")
	plt.imshow(subset)
	plt.show()

	
	# applying cleaning algorithm
	print (">> cleaning image")
	clean_subset = clean(subset, RAY)
	plt.imshow(clean_subset)
	plt.title("cleaned subset")
	plt.show()
	
	print (">> finding contour")
	contour = contour_manager(clean_subset)
	final = fill_inside(contour)
	# get perimeter ratio
	ratio = get_ratio(contour)
	print ratio

	fig.save_table(item, ratio, "table.dat")

with open("table.dat", "a") as file:
	file.write("\hline\n\\end{tabular}")
