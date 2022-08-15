from cmath import sqrt
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib as mpl
from matplotlib import pyplot as plt
import math
from scipy.stats import multivariate_normal

def gaussDeriv2D(sigma):
  
  ksize = 2*math.ceil(3*sigma) + 1
  
  # Create ranges for x and y
  range_x = np.linspace(-1*math.ceil(3*sigma), math.ceil(3*sigma), ksize)
  range_y = np.linspace(-1*math.ceil(3*sigma), math.ceil(3*sigma), ksize)
      
  # Create Deriv matrices
  g_x = np.ndarray((ksize, ksize))
  g_y = np.ndarray((ksize, ksize))
  
  # Guassian deriv over x and y 
  gauss_x = lambda x,y: x / (2 * math.pi * sigma ** 4) * math.exp(-1*(x ** 2 + y ** 2)/(2 * sigma ** 2))
  gauss_y = lambda x,y: y / (2 * math.pi * sigma ** 4) * math.exp(-1*(x ** 2 + y ** 2)/(2 * sigma ** 2))
  
  # Calculate Gaussian deriv's over x and y range
  for y in range(ksize):
      for x in range(ksize):
          g_x[y][x] = gauss_x(range_x[x], range_y[y])
          g_y[y][x] = gauss_y(range_x[x], range_y[y])
          
  return g_x, g_y

def getHOG(im_dx, im_dy, num_cells = 4, num_bins=4, use_360 = False):
  
  im_dx = np.array(im_dx)
  im_dy = np.array(im_dy)
  
  im_height = im_dy.shape[0]
  im_width = im_dy.shape[1]
    
  im_grad = ((im_dx**2) + (im_dy**2))**0.5
  im_dir = np.arctan2(im_dy, (im_dx+1e-6))*180/math.pi if use_360 else np.arctan(im_dy/(im_dx+1e-6))*180/math.pi
  
  # fig, ax = plt.subplots(2, 2, figsize=(10, 10))
  # ax = ax.ravel()
  # ax[0].imshow(mpl.colors.Normalize()(im_dx))
  # ax[1].imshow(mpl.colors.Normalize()(im_dy))
  # ax[2].imshow(mpl.colors.Normalize()(im_grad))
  # ax[3].imshow(mpl.colors.Normalize()(im_dir))
  # plt.show()
  
  bin_size = 360/num_bins if use_360 else 180/num_bins
    
  # print(bin_size)

  # Histogram dims = num_cells x num_bins x 3
  histogram_of_counts = np.zeros((num_cells, num_bins, 3))
  histogram_of_magnitudes = np.zeros((num_cells, num_bins, 3))
  
  num_cells_x = int(num_cells**.5)
  num_cells_y = int(num_cells**.5)
  
  for i in range(num_cells_y):
    for j in range(num_cells_x):
      for y in range(int(im_height/num_cells_y)*i, int(im_height/num_cells_y)*(i+1)):
        for x in range(int(im_width/num_cells_x)*j, int(im_width/num_cells_x)*(j+1)):
          mag = im_grad[y,x]

          dir = im_dir[y,x]
          # print(dir)
          
          cell_idx = i*num_cells_y + j
          bin_idx = np.floor((dir + 180)/bin_size).astype(int) if use_360 else np.floor((dir + 90)/bin_size).astype(int)
          bin_idx[0] = min(bin_idx[0], 7)
          bin_idx[1] = min(bin_idx[1], 7)
          bin_idx[2] = min(bin_idx[2], 7)

          histogram_of_counts[cell_idx, bin_idx] += 1
          histogram_of_magnitudes[cell_idx, bin_idx] += mag
  
  histogram_of_counts = histogram_of_counts.flatten()
  histogram_of_counts /= np.linalg.norm(histogram_of_counts, ord=1)
  
  histogram_of_magnitudes = histogram_of_magnitudes.flatten()  
  histogram_of_magnitudes /= np.linalg.norm(histogram_of_magnitudes, ord=1)

  # print(histogram_of_counts)
  # print(histogram_of_magnitudes)

  return histogram_of_counts, histogram_of_magnitudes

train_csv = pd.read_csv('D:/GermanTrafficSignDataSet/Train.csv')
test_csv = pd.read_csv('D:/GermanTrafficSignDataSet/Test.csv')

image_root_dir = 'D:/GermanTrafficSignDataSet'

####################################################
# PARAMETERS
####################################################
min_class = 0
max_class = 42

max_train_per_class = 500
max_val_per_class = 10
max_test_per_class = 3

gauss_sigma = .75

hog_num_cells = 9
hog_num_directions = 8
hog_use_counts = False

use_360 = True

testing = False

####################################################
# LOAD TRAIN AND VAL SET
####################################################
train_list = []
val_list = []

####################################################
# Shuffle Training Set
####################################################
train_csv = train_csv.sample(frac=1).reset_index(drop=True)

print('Loading training images...')
# Loop through classes
for i in range(min_class, max_class + 1):
  train_array = []
  val_array = []
  # Get each row
  for idx, row in train_csv.loc[train_csv['ClassId'] == i].head(max_train_per_class+max_val_per_class).iterrows():
    # load image
    im = cv.imread(f'{image_root_dir}/{row["Path"]}')
    # convert from BGR to RGB
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    
    # Get region of interest from image
    roi = im[row['Roi.Y1']+1:row['Roi.Y2']+1, row['Roi.X1']+1:row['Roi.X2']+1]

    if len(val_array) < max_val_per_class:
      val_array.append(roi)

    else:
      train_array.append(roi)

  print(f'Class {i} training image count: {len(train_array)}')
  print(f'Class {i} validation image count: {len(val_array)}')  
      
  train_list.append(train_array)
  val_list.append(val_array)
  
####################################################
# LOAD TEST SET
####################################################

####################################################
# Shuffle Test Set
####################################################
test_csv = test_csv.sample(frac=1).reset_index(drop=True)

test_list = []

print('Loading test images...')
# Loop through classes
for i in range(min_class, max_class + 1):
  test_array = []
  # Get each row
  for idx, row in test_csv.loc[test_csv['ClassId'] == i].head(max_test_per_class).iterrows():
    # load image
    im = cv.imread(f'{image_root_dir}/{row["Path"]}')
    # convert from BGR to RGB
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    
    # Get region of interest from image
    roi = im[row['Roi.Y1']+1:row['Roi.Y2']+1, row['Roi.X1']+1:row['Roi.X2']+1]

    test_array.append(roi)
      
  test_list.append(test_array)
  
print('Finished loading images')

####################################################
# GAUSSIAN DERIVATIVES
####################################################
g_x, g_y = gaussDeriv2D(gauss_sigma)

####################################################
# TRAINING
####################################################
print('Training...')

class_hogs = []
for cls_i, cls in enumerate(train_list):
  print(f'Training class {cls_i}...')
  class_array = None
  for im in cls:
    im_dx = cv.filter2D(im.astype(float), -1, g_x, borderType=cv.BORDER_REPLICATE)
    im_dy = cv.filter2D(im.astype(float), -1, g_y, borderType=cv.BORDER_REPLICATE)
    hist_counts, hist_mags = getHOG(im_dx, im_dy, hog_num_cells, hog_num_directions, use_360=use_360)

    if class_array is None:
      class_array = hist_counts
    else:
      class_array = np.vstack((class_array, hist_counts))

  # print(class_array.shape)
  class_hogs.append(class_array)
  print(f'Finished training class {cls_i}.')
  
# Shape = [# class, # im, len(hog)]
class_hogs = np.array(class_hogs)
  
class_means = []
class_covs = []

for cls in class_hogs:
  class_means.append(np.mean(cls, axis=0))
  class_covs.append(np.cov(cls, rowvar=False))
  
for i, (mean, cov) in enumerate(zip(class_means, class_covs)):
  print(f'Class {i} mean: ', mean)
  print(f'Class {i} covariance mat: ', cov.shape)

if not testing:
  total_correct = 0
  total_guess = 0

  accuracy_list = []

  print('Validating...')
  for cls_i, cls in enumerate(val_list):
    print(f'Validating class {cls_i}...')
    cls_correct = 0
    cls_guess = 0
    for im in cls:
      im_dx = cv.filter2D(im.astype(float), -1, g_x, borderType=cv.BORDER_REPLICATE)
      im_dy = cv.filter2D(im.astype(float), -1, g_y, borderType=cv.BORDER_REPLICATE)
      hog_counts, hog_mags = getHOG(im_dx, im_dy, hog_num_cells, hog_num_directions, use_360=use_360)

      likelihoods = []

      for i, (cls_mean, cls_cov) in enumerate(zip(class_means, class_covs)):
        if hog_use_counts:
          likelihood = multivariate_normal.logpdf(hog_counts, cls_mean, cls_cov, allow_singular=True)
        else: 
          likelihood = multivariate_normal.logpdf(hog_mags, cls_mean, cls_cov, allow_singular=True)
        likelihoods.append(likelihood)

      predicted_class = np.argmax(np.array(likelihoods))

      print(f'Predicted class: {predicted_class}; Actual class = {cls_i}')

      if predicted_class == cls_i: 
        cls_correct += 1
        total_correct += 1

      cls_guess += 1
      total_guess += 1

    accuracy_list.append((float(cls_correct)/float(cls_guess), cls_correct, cls_guess))
    print(f'Finished validating class {cls_i}.')
    print(f'Class {cls_i} Accuracy = {float(cls_correct)/float(cls_guess)}, {cls_correct}/{cls_guess} Correct.\n\n')

  for cls_i, accuracy in enumerate(accuracy_list):
    print(f'Class {cls_i} Accuracy = {accuracy[0]}, {accuracy[1]}/{accuracy[2]} Correct.\n\n')

  print(f'Total Accuracy = {float(total_correct)/float(total_guess)}, {float(total_correct)}/{float(total_guess)} Correct.')

else:
  total_correct = 0
  total_guess = 0

  accuracy_list = []

  print('Testing...')
  for cls_i, cls in enumerate(test_list):
    print(f'Testing class {cls_i}...')
    cls_correct = 0
    cls_guess = 0
    for im in cls:
      im_dx = cv.filter2D(im.astype(float), -1, g_x, borderType=cv.BORDER_REPLICATE)
      im_dy = cv.filter2D(im.astype(float), -1, g_y, borderType=cv.BORDER_REPLICATE)
      hog_counts, hog_mags = getHOG(im_dx, im_dy, hog_num_cells, hog_num_directions, use_360=use_360)

      likelihoods = []

      for i, (cls_mean, cls_cov) in enumerate(zip(class_means, class_covs)):
        likelihood = multivariate_normal.logpdf(hog_counts, cls_mean, cls_cov, allow_singular=True)
        likelihoods.append(likelihood)

      predicted_class = np.argmax(np.array(likelihoods))

      print(f'Predicted class: {predicted_class}; Actual class = {cls_i}')

      if predicted_class == cls_i: 
        cls_correct += 1
        total_correct += 1

      cls_guess += 1
      total_guess += 1

    accuracy_list.append((float(cls_correct)/float(cls_guess), cls_correct, cls_guess))
    print(f'Finished testing class {cls_i}.')

  for cls_i, accuracy in enumerate(accuracy_list):
    print(f'Class {cls_i} Accuracy = {accuracy[0]}, {accuracy[1]}/{accuracy[2]} Correct.\n\n')

  print(f'Total Accuracy = {float(total_correct)/float(total_guess)}, {float(total_correct)}/{float(total_guess)} Correct.')