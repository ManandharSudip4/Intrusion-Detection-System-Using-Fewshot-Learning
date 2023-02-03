import pandas as pd
import os
from IGTD_Functions import min_max_transform, table_to_image, min_max_transform_prediction
import numpy as np
import time
import argparse

def argument_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("-e", "--euclidean", help="use euclidean distance", action="store_true")
  parser.add_argument("-m", "--manhattan", help="use manhattan distance", action="store_true")
  args = parser.parse_args()
  return args.euclidean, args.manhattan


num_row = 9    # Number of pixel rows in image representation
num_col = 9    # Number of pixel columns in image representation
num = num_row * num_col # Number of features to be included for analysis, which is also the total number of pixels in image representation
save_image_size = 3 # Size of pictures (in inches) saved during the execution of IGTD algorithm.
max_step = 1000 #10000    # The maximum number of iterations to run the IGTD algorithm, if it does not converge.
val_step = 300  # The number of iterations for determining algorithm convergence. If the error reduction rate
                # is smaller than a pre-set threshold for val_step itertions, the algorithm converges.

# dataset_path = "/content/CICIDS2017/20percent_of_combined.csv"
# result_path = "/content/drive/MyDrive/datasets/IGTD-results/"

dataset_path = "/home/user/manualpartition/teamIDS/Datasets/70percent_train_of_all_combined.csv"

result_path = "IGTD-results30Com/"

# Import the example data and linearly scale each feature so that its minimum and maximum values are 0 and 1, respectively.
# data = pd.read_csv('../Data/Data.txt', low_memory=False, sep='\t', engine='c', na_values=['na', '-', ''], 
#                 header=0, index_col=0)

if __name__ == "__main__":
  euclidean, manhattan = argument_parser()
  t1 = time.time()
  data = pd.read_csv(dataset_path, low_memory=False, sep=',', engine='c', na_values=['na', '-', ''], 
                  header=0)
  label = data['Label']
  data.drop('Label', axis=1, inplace=True)
  data['01'] = 0
  data['02'] = 0
  data['03'] = 0
  data['04'] = 0

  # Replacing infinite with nan
  data.replace([np.inf, -np.inf], np.nan, inplace=True)
  # Dropping all the rows with nan values
  data.dropna(inplace=True)
  t2 = time.time()

  print(f'Time required to read data: {t2-t1:.2f}s')

  print(f'before reshaping: {data.shape}')
  data = data.iloc[:, :num]
  print(f'after reshaping: {data.shape}')
  # norm_data = min_max_transform(data.values)
  norm_data = min_max_transform_prediction(data.values)
  norm_data = pd.DataFrame(norm_data, columns=data.columns, index=data.index)

  # Run the IGTD algorithm using (1) the Euclidean distance for calculating pairwise feature distances and pariwise pixel
  # distances and (2) the absolute function for evaluating the difference between the feature distance ranking matrix and
  # the pixel distance ranking matrix. Save the result in Test_1 folder.
  if euclidean:
    print('Converting [Euclidean] ...')
    t3 = time.time()
    fea_dist_method = 'Euclidean'
    image_dist_method = 'Euclidean'
    error = 'squared'
    result_dir = os.path.join(result_path, "euclidean")
    os.makedirs(name=result_dir, exist_ok=True)
    table_to_image(label, norm_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
                  max_step, val_step, result_dir, error)
    t4 = time.time()
    print(f'Time required to convert 1d to 2d [Euclidean]: {t4-t3:.2f}s | {(t4-t3)/60:.2f}m')

  # Run the IGTD algorithm using (1) the Pearson correlation coefficient for calculating pairwise feature distances,
  # (2) the Manhattan distance for calculating pariwise pixel distances, and (3) the square function for evaluating
  # the difference between the feature distance ranking matrix and the pixel distance ranking matrix.
  # Save the result in Test_2 folder.
  if manhattan:
    print('Converting [Pearson Manhattan] ...')
    t5 = time.time()
    fea_dist_method = 'Pearson'
    image_dist_method = 'Manhattan'
    error = 'abs'
    result_dir = os.path.join(result_path, "manhattan")
    os.makedirs(name=result_dir, exist_ok=True)
    table_to_image(label, norm_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
                  max_step, val_step, result_dir, error)
    t6 = time.time()
    print(f'Time required to convert 1d to 2d [Pearson Manhattan]: {t6-t5:.2f}s | {(t6-t5)/60:.2f}m')
