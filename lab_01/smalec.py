import numpy as np
import matplotlib.pyplot as plt
# from scipy.spatial import KDTree
from sklearn.neighbors import KDTree
from mpl_toolkits.mplot3d import Axes3D

def plot(xx, yy, zz):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(xx, yy, zz)

def plot_surf(zz):
	plt.figure()
	p = plt.imshow(zz)
	plt.colorbar(p)

data_to_load = './data/wraki_utm.txt'
input_tmp = input(f'Read file (default: {data_to_load}): ')
data_to_load = input_tmp if input_tmp != '' else data_to_load

grid_spacing = 0.2
input_tmp = input(f'Enter grid spacing (default: {grid_spacing}): ')
grid_spacing = input_tmp if input_tmp != '' else grid_spacing
# rect_size = input('Enter rectangle size: ')
# rect_size = 0.4
radius_length = 0.4
input_tmp = input(f'Enter radius length (default: {radius_length}): ')
radius_length = input_tmp if input_tmp != '' else radius_length

min_points = 1
input_tmp = input(f'Enter minimum points required to mean (default: {min_points}): ')
min_points = input_tmp if input_tmp != '' else min_points

# Read File
data = np.loadtxt(data_to_load)
# print(data)
points_mins = data.min(axis=0)
points_maxs = data.max(axis=0)
# print(f'Mins: {points_mins}')
# print(f'Maxs: {points_maxs}')
print(f'Between X: {points_maxs[0] - points_mins[0]}, Y: {points_maxs[1] - points_mins[1]}')

x = np.arange(points_mins[0], points_maxs[0], grid_spacing)
y = np.arange(points_mins[1], points_maxs[1], grid_spacing)
xx, yy = np.meshgrid(x, y)
print('Grid size:', len(x), len(y))

xy_data = data[:, :2]
# print(xy_data)
tree = KDTree(xy_data)

# Example query
# ids = tree.query_radius([[452615.6, 5967815.8]], r=radius_length)
# print(ids)

zz = np.empty((xx.shape[0], xx.shape[1]))

for i in range(xx.shape[0]):
	for j in range(xx.shape[1]):
		ids = tree.query_radius([[xx[i,j], yy[i,j]]], r=radius_length)

		if len(ids[0]) < min_points:
			zz[i, j] = np.nan
			continue

		zz[i, j] = np.mean(data[ids[0], 2])


plot(xx, yy, zz)
plot_surf(zz)

plt.show()
