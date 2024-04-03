


"""
Don't forget:

	- relire tout le code pour voir si la structure générale est logique


"""


########################################################

## MODULES WITH EPOCHS TUPLE AND ADD_TARGET IN FOR LOOP

########################################################

## DATA SPLITER METHODS
import platform
import os
uname = platform.uname()
env_ = os.environ
if 'COLAB_GPU' in env_ or 'COLAB_JUPYTER_IP' in env_:

	### Install Pyrebase4
	for _ in range(2):
		try:
			import pyrebase
			break
		except:
			# !pip install Pyrebase4
			print("Installing Pyrebase4 ...\n")
			os.system("pip install Pyrebase4")

	### Import modules
	from tensorflow import keras
	from tensorflow.keras.models import load_model as krs_load_model
	import tensorflow as tf

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import warnings
import time
import datetime
import pickle

warnings.simplefilter(action = 'ignore', category = UserWarning)
pd.options.mode.chained_assignment = None
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)

def save_pickle(filepath, data):
	with open(filepath, 'wb') as file_pi:
		pickle.dump(data, file_pi)

def load_pickle(filepath):
	try:
		with open(filepath, "rb") as file_pi:
			loaded_ = pickle.load(file_pi)
		return loaded_
	except FileNotFoundError:
		return False


def check_file_exists(filepath):
	return os.path.isfile(path = filepath)


if 'COLAB_GPU' in env_ or 'COLAB_JUPYTER_IP' in env_:
	class EachEpochCallback(keras.callbacks.Callback):
		def __init__(self, epochs, test_nbr, 
			general_path, modulos_save_landmark, 
			saved_last_epoch_filepath, good_color, alert_color, bold):
			
			self.durations = []
			self.epochs = epochs
			self.saved_last_epoch_filepath = saved_last_epoch_filepath
			self.test_nbr = test_nbr
			self.general_path = general_path
			self.modulos_save_landmark = modulos_save_landmark
			self.good_color = good_color
			self.alert_color = alert_color
			self.bold = bold
		
		def on_epoch_begin(self, epoch, logs=None):
			self.now = time.time()

		def on_epoch_end(self, epoch, logs):
			later = time.time()
			duration = later - self.now
			self.durations.append(duration)

			# loss_n_val_loss['loss'].append(logs["loss"])
			# loss_n_val_loss['val_loss'].append(logs["val_loss"])
			# save_pickle(filepath = history_filepath, data = loss_n_val_loss)

			with open(self.saved_last_epoch_filepath, "a", encoding = "utf-8") as f:
				f.write(f'{epoch}\n')

			# if (epoch+1)%self.modulos_save_landmark['long'] == 0:
			# 	firebase_storage = FirebaseStorage(firebase_config = firebase_config)

			# 	# ### landmarks_filenames = [self.saved_last_epoch_filepath,
			# 	# 						self.model_filepath,]

			# 	landmarks_filenames = [f'model_test_nbr_{self.test_nbr}.h5',
			# 						f'saved_last_epoch_filepath_test_nbr_{self.test_nbr}.txt',]

			# 	for f in landmarks_filenames:
			# 		try:
			# 			result_file_url = firebase_storage.upload_file(
			# 						local_file_path_name = self.general_path + f, 
			# 						cloud_file_path_name = "instantanely/landmarks/" + f,
			# 						)
			# 			print_style(f"{f} uploaded. Its url is: \n\t {result_file_url}",
			# 											color = self.good_color, bold = self.bold)
			# 		except FileNotFoundError:
			# 			print_style(f"No file named: {f}, in Google Drive !!!",
			# 						color = self.alert_color, bold = self.bold)

			if (epoch+1)%3 == 0:
				staying_epochs = self.epochs - (epoch+1)
				average_duration = sum(self.durations)/len(self.durations)
				staying_time = average_duration * staying_epochs
				finish_at = time.time() + staying_time
				now_time_ = time.time()
				if 'COLAB_GPU' in env_ or 'COLAB_JUPYTER_IP' in env_:
					finish_at += (3600*2)
					now_time_ += (3600*2)
				finish_at = datetime.datetime.fromtimestamp(finish_at)
				now_time_ = datetime.datetime.fromtimestamp(now_time_)

				print("*\n"*3)
				print("Time now is            :", now_time_)
				print("Running will finish at :", finish_at)
				print("Staying time           :", round(staying_time/60, 3), "Minutes or", round(staying_time/3600, 3), "Hours")
				print("*\n"*3)


class FirebaseStorage:
	def __init__(self, firebase_config):
		import pyrebase
		firebase = pyrebase.initialize_app(firebase_config)
		self.storage = firebase.storage()

	def upload_file(self, local_file_path_name, cloud_file_path_name):
		uploading = self.storage.child(cloud_file_path_name).put(local_file_path_name)
		url_file_on_cloud = self.storage.child(cloud_file_path_name).get_url(None)
		return url_file_on_cloud

	def download_file(self, cloud_file_path_name, local_file_path_name, verbose = False):
		try:
			self.storage.child(cloud_file_path_name).download('', filename = local_file_path_name)
			if verbose:
				print("File successfully downloaded !")
		except Exception as e:
			print("Error :\n", e)


firebase_config = {
	"apiKey": "AIzaSyANPOUHxWG48LpdFATg10gJwY42ouX5p04",
	"authDomain": "saving-data-2ee4b.firebaseapp.com",
	"databaseURL": "",
	"projectId": "saving-data-2ee4b",
	"storageBucket": "saving-data-2ee4b.appspot.com",
	"messagingSenderId": "403260823146",
	"appId": "1:403260823146:web:597c2ac9422b7aa37e1c13",
	"measurementId": "G-3P1DE9GSD5"
}


def manage_wargins():
	pd.options.mode.chained_assignment = None
	warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def load_the_model(model_filepath):
	return krs_load_model(model_filepath)


def print_style(text, color = None, bold = False, underline = False):
	colors = {"purple":'\033[95m',
			"cyan":'\033[96m',
			"darkcyan":'\033[36m',
			"blue":'\033[94m',
			"green":'\033[92m',
			"yellow":'\033[93m',
			"red":'\033[91m'}
	other_style = {"bold":'\033[1m',
				"underline":'\033[4m'}
	end = '\033[0m'
	if color is None and bold and not underline:
		print(other_style['bold'] + text + end)
	if color is None and not bold and underline:
		print(other_style['underline'] + text + end)
	if color is None and bold and underline:
		print(other_style['bold'] + other_style['underline'] + text + end)
	if bold and not underline and color is not None:
		print(colors[color.lower()] + other_style['bold'] + text + end)
	if underline and not bold and color is not None:
		print(colors[color.lower()] + other_style['underline'] + text + end)
	if underline and bold and color is not None:
		print(colors[color.lower()] + other_style['bold'] + other_style['underline'] + text + end)
	if not bold and not underline and color is not None:
		print(colors[color.lower()] + text + end)
	if not bold and not underline and color is None:
		print(text)



## DATA SCALER METHODS

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import random

def data_scaler(df):
  dataset = df.copy()
  columns = dataset.columns.tolist()
  scaler = MinMaxScaler(feature_range=(0, 1))
  dataset = scaler.fit_transform(dataset)
  dataset = pd.DataFrame(dataset, columns = columns)
  return dataset, scaler

def data_unscaler(df, scaler):
  df = df.copy()
  columns = df.columns.tolist()
  unscaled_df = scaler.inverse_transform(df)
  unscaled_df = pd.DataFrame(unscaled_df, columns = columns)
  return unscaled_df


## DATA ENGENEERING METHODS

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from pprint import pprint
import pandas as pd
import numpy as np
import random
import pywt
pd.options.mode.chained_assignment = None


def denoise_with_fft(data):
	### Compute the Fast Fourier Transform (FFT)
	n = len(data)
	fhat = np.fft.fft(data, n) 						# Compute the FFT
	PSD = fhat * np.conj(fhat) / n 					# Power spectrum (power per f.
	# freq = (1/(dt*n)) * np.arange(n) 				# Create x-axis of frequencie.
	# L = np.arange(1, np.floor(n/2), dtype='int') 	# Only plot the first half of

	# plt.plot(PSD)
	# plt.show()


	## Use the PSD to filter out noise
	indices = PSD > 100 			# Find all freqs with large power
	PSDclean = PSD * indices 		# Zero out all others
	fhat = indices * fhat 			# Zero out small Fourier coeffs. in Y
	ffilt = np.fft.ifft(fhat) 		# Inverse FFT for filtered time signal

	print(ffilt)

	# Plots
	fig,axs = plt.subplots(2, 1)

	plt.sca(axs[0])
	plt.plot(data, color='c',)
	plt.legend()

	plt.sca(axs[1])
	plt.plot(ffilt, color='k', label = 'ffilt')
	plt.legend()

	plt.show()

	return ffilt



class ManageWithWaveletes:
	def __init__(self, signal):
		self.signal = signal

	### 1. Signal Analysis using DWT
	def dwt_signal_analysis(self, plot_result = False):
		# Apply DWT:
		coeffs = pywt.dwt(self.signal, 'db1')
		cA, cD = coeffs

		if plot_result:
			# Plotting
			plt.figure(figsize=(12, 4))

			plt.subplot(1, 3, 1)
			plt.plot(self.signal)
			plt.title("Original Signal")

			plt.subplot(1, 3, 2)
			plt.plot(cA)
			plt.title("Approximation Coefficients")

			plt.subplot(1, 3, 3)
			plt.plot(cD)
			plt.title("Detail Coefficients")

			plt.tight_layout()
			plt.show()

		return {"cA":cA, "cD":cD}



	# 2. Denoising Signal Using Wavelet Transform
	def denoise_with_wavelet_trans(self, threshold, plot_result = False, mode = 'soft'):

		# mode possibilities : {'soft', 'hard', 'garrote', 'greater', 'less'}

		# Perform a multi-level wavelet decomposition
		coeffs = pywt.wavedec(self.signal, 'db1', level=4)

		# Set a threshold to nullify smaller coefficients (assumed to be noise)
		coeffs_thresholded = [pywt.threshold(c, threshold, mode=mode) for c in coeffs]

		# Reconstruct the signal from the thresholded coefficients
		denoised_signal = pywt.waverec(coeffs_thresholded, 'db1')

		if plot_result:

			# Plotting the noisy and denoised signals
			plt.figure(figsize=(12, 4))
			plt.subplot(1, 2, 1)
			plt.plot(self.signal)
			plt.title("Noisy Signal")

			plt.subplot(1, 2, 2)
			plt.plot(denoised_signal)
			plt.title("Denoised Signal")

			plt.tight_layout()
			plt.show()

		return denoised_signal

### ETS DECOMPOSITION
def seasonal_decompose_(series, freq, model = "additive"):
	try:
		decomposition = seasonal_decompose(series,
			freq = freq, ## pour une saisonnalité de 60 unités.
			model = model, ## values "additive" or "multiplicative"
			)
	except TypeError:
		decomposition = seasonal_decompose(series,
			period = freq, ## pour une saisonnalité de 60 unités.
			model = model, ## values "additive" or "multiplicative"
			)
	return decomposition



### STATIONNARISATION DES DONNEES:
### ______________________________
def stationnarizer(df, column_name, limit = None):
	def stationnarity_checker(time_series):
		result = adfuller(time_series)
		labels = ["ADF Test Statistic", "p-value", "Number of Observations Used"]
		if result[1] <= 0.05:
			return "Données stationnaires."
		else:
			return "Données NON stationnaires."

	_df = df.copy()
	_df['Diff 0'] = _df[column_name]

	for i in range(1, 1_000_000):
		_df[f'Diff {str(i)}'] = _df[f'Diff {str(i-1)}'] - _df[f'Diff {str(i-1)}'].shift(1)
		_df[f'Diff {str(i)}'].fillna(method = "bfill", inplace = True)
		result_check_stationnarity = stationnarity_checker(time_series = _df[f'Diff {str(i)}'])

		if result_check_stationnarity == "Données stationnaires.":
			data = _df[f"Diff {str(i)}"].copy()
			return data

		elif limit is not None and i >= limit:
			data = _df[f"Diff {str(i)}"].copy()
			return data


def stationnarizer_v2(df, column_name, limit):
	_df = df.copy()
	_df['Diff 0'] = _df[column_name]
	for i in range(1, limit+1):
		_df[f'Diff {str(i)}'] = _df[f'Diff {str(i-1)}'] - _df[f'Diff {str(i-1)}'].shift(1)
		_df[f'Diff {str(i)}'].fillna(method = "bfill", inplace = True)

	return _df[f'Diff {limit}']




# HANDLE COLUMNS METHODS

def handle_columns_v2(df,
	close_column_name, 
	freqs_seasonal, 
	target_col_name, 
	target_type = None,
	target_shift = None,
	limit_stationnarization = 3,
	wavelets_threshold = 0.5,
	ratio_true_trend = None,
	diviseur_ecart_entre_list_values = 1):

	"""
	diviseur_ecart_entre_list_values: Si jamais on cherher à classifier la tendance et qu'en filtrant
	l'etiquettage on obtient une valeur tassez superieur au pourcentage voulu (==> ratio_true_trend)
	alors il faut augmenter la valeur de 'diviseur_ecart_entre_list_values' qui est égale à 1 par
	defaut.
	On peut essayer avec 2 ou 5 ou 10 etc.
	"""

	def drop_first_row_if_odd(df):
		if len(df)%2 != 0:
			df = df.iloc[1:, :]
			df.reset_index(inplace = True, drop = True)
			return df
		return df

	df = df.copy()
	df = drop_first_row_if_odd(df = df)

	def duplicate_rows(series):
		new_col = [(item, item) for item in series]
		new_col_ = []
		for item in new_col:
			new_col_.append(item[0])
			new_col_.append(item[1])
		return new_col_

	def get_range_neutral_trend(list_, ratio_true_trend, diviseur_ecart_entre_list_values = 1):
		assert 0.0 < ratio_true_trend < 1.0, "The value of 'ratio_true_trend' must be in the interval ]0,1[."
		ratio_trend_neutral = 1.0 - ratio_true_trend
		ecart = 1e-06/diviseur_ecart_entre_list_values
		step = 0
		while True:
			step += ecart
			interval = (-step, step)
			low_data = [d for d in list_ if d>= interval[0] and d<interval[1]]
			ratio = len(low_data)/len(list_)
			if ratio >= ratio_trend_neutral:
				return {"interval":interval, "ratio_get":1-ratio}
			if ratio >= 1.0:
				break


	#### 0. Price brut:
	if target_col_name is not None:
		df_price_n_target = df[[close_column_name, target_col_name]]
	else:
		df_price_n_target = df[[close_column_name]]

	#### 1. STATIONNARIZED PRICE COLUMN:
	# ## data = stationnarizer(df = df, column_name = close_column_name, limit = limit_stationnarization)
	stationnarized_col = stationnarizer_v2(df = df,
										column_name = close_column_name,
										limit = limit_stationnarization)
	df_stationnarized_col = pd.DataFrame({f'stationnarized_{close_column_name}':stationnarized_col})

	#### 2. GET SEASONALITIES:
	df_seasonal = pd.DataFrame()
	for idx, freq_seasonal in enumerate(freqs_seasonal):
		dec = seasonal_decompose_(series = df[close_column_name], freq = freq_seasonal, model = "additive")
		df_seasonal["seasonality_"+str(freq_seasonal)] = dec.seasonal


	#### 3. DENOISING WITH WAVELETS:
	manage_with_waveletes = ManageWithWaveletes(signal = df[close_column_name])
	df_denoising_with_wvlts = pd.DataFrame()
	# wavelets_modes = ['soft', 'hard', 'garrote', 'greater', 'less']
	# wavelets_modes = ['soft'] # idem on thresholds
	# wavelets_modes = ['hard'] # idem on thresholds and its idem to soft
	# wavelets_modes = ['garrote'] # idem on thresholds and its idem to soft
	# wavelets_modes = ['greater'] # idem on thresholds and its idem to soft
	# wavelets_modes = ['less'] # idem on thresholds but different to soft

	wavelets_modes = ['soft', 'less']
	# plot_rank = 1
	# rows = 2
	# cols = 2
	# plt.subplot(rows,cols,1)
	# plt.plot(df[close_column_name])
	# plt.title("Price")

	for w_mode in wavelets_modes:
		denoised_data = manage_with_waveletes.denoise_with_wavelet_trans(threshold = wavelets_threshold,
																		mode = w_mode)
		df_denoising_with_wvlts[f"{w_mode}_{wavelets_threshold}"] = denoised_data

		# plot_rank += 1
		# plt.subplot(rows,cols,plot_rank)
		# plt.plot(denoised_data)
		# plt.title(f"{w_mode}_{wavelets_threshold}")

	## Add soft stationnarized column:
	df_denoising_with_wvlts[f'soft_{wavelets_threshold}_stationnarized'] = stationnarizer_v2(df = df_denoising_with_wvlts,
																column_name = f'soft_{wavelets_threshold}',
																limit = limit_stationnarization)


	#### 4. GET DWT:
	df_dwt = pd.DataFrame()
	dwt_ = manage_with_waveletes.dwt_signal_analysis()
	cA = dwt_["cA"]
	cD = dwt_["cD"]
	cA = duplicate_rows(series = cA)
	cD = duplicate_rows(series = cD)
	df_dwt['dwt_cA'] = cA
	df_dwt['dwt_cD'] = cD
	df_dwt['dwt_cA_stationnarized'] = stationnarizer_v2(df = df_dwt,
												column_name = "dwt_cA",
												limit = limit_stationnarization)

	df_treated = pd.concat([df_price_n_target, df_stationnarized_col,
							df_seasonal, df_denoising_with_wvlts,
							df_dwt], axis = 1)

	### 5. ADD TARGET COLUMN:

	if (target_shift is not None) and (target_type is not None):

		assert target_type.lower() == 'classification' or target_type.lower() == 'regression', \
						"The value of 'target_type' must be 'classification' or 'regression'."

		target = df[close_column_name].shift(-abs(target_shift))

		if target_type.lower() == "regression":
			df_treated['target'] = target

		else: ### if target type is classification:
			df_target = pd.DataFrame({close_column_name:df[close_column_name], 'target':target})
			df_target['sens'] = df_target['target'] - df_target[close_column_name]
			list_ = df_target[['sens']]
			list_.dropna(inplace = True)
			list_ = list_['sens'].tolist()

			range_neutral_trend_ratio_get = get_range_neutral_trend(list_ = list_,
														ratio_true_trend = ratio_true_trend,
														diviseur_ecart_entre_list_values = diviseur_ecart_entre_list_values)
			range_neutral_trend = range_neutral_trend_ratio_get["interval"]
			ratio_get = range_neutral_trend_ratio_get["ratio_get"]

			if (ratio_true_trend - ratio_get)*100 >= 5:
				good_ratio = False
				text_ = f"Veuillez augmenter la valeur de 'diviseur_ecart_entre_list_values' car le ratio obtenu ratio_get: {ratio_get} est assez écarté de celui voulu, ratio_true_trend: {ratio_true_trend}"
				assert good_ratio == True, text_
			# elif ratio_get - ratio_true_trend >= 5:
			# 	good_ratio = False
			# 	text_ = "Veuillez diminuer la valeur de 'diviseur_ecart_entre_list_values' car le ratio obtenu est assez écarté de celui voulu."
			# 	assert good_ratio == True, text_

			target = np.where(df_target['sens'] < range_neutral_trend[0], -1,
									np.where(df_target['sens'] > range_neutral_trend[1], 1, 0.0))
			df_treated['target'] = target
		stop_at = len(df_treated) - target_shift
		# df_treated.reset_index(inplace = True, drop = True)
		df_treated = df_treated.head(stop_at)

	return df_treated


### ADD TARGET COLUMN SEPARATELY:
def add_target(df,
	close_column_name, 
	target_type,
	target_shift,
	drop_added_nan,
	ratio_true_trend = None,
	diviseur_ecart_entre_list_values = 1):

	"""
	diviseur_ecart_entre_list_values: Si jamais on cherher à classifier la tendance et qu'en filtrant
	l'etiquettage on obtient une valeur tassez superieur au pourcentage voulu (==> ratio_true_trend)
	alors il faut augmenter la valeur de 'diviseur_ecart_entre_list_values' qui est égale à 1 par
	defaut.
	On peut essayer avec 2 ou 5 ou 10 etc.
	"""

	# def drop_first_row_if_odd(df):
	# 	if len(df)%2 != 0:
	# 		df = df.iloc[1:, :]
	# 		df.reset_index(inplace = True, drop = True)
	# 		return df
	# 	return df

	df = df.copy()
	# df = drop_first_row_if_odd(df = df)

	def get_range_neutral_trend(list_, ratio_true_trend, diviseur_ecart_entre_list_values = 1):
		assert 0.0 < ratio_true_trend < 1.0, "The value of 'ratio_true_trend' must be in the interval ]0,1[."
		ratio_trend_neutral = 1.0 - ratio_true_trend
		ecart = 1e-06/diviseur_ecart_entre_list_values
		step = 0
		while True:
			step += ecart
			interval = (-step, step)
			low_data = [d for d in list_ if d>= interval[0] and d<interval[1]]
			ratio = len(low_data)/len(list_)
			if ratio >= ratio_trend_neutral:
				return {"interval":interval, "ratio_get":1-ratio}
			if ratio >= 1.0:
				break

	df_to_treat = df.copy()

	### ADD TARGET COLUMN:
	assert target_type.lower() == 'classification' or target_type.lower() == 'regression', \
					"The value of 'target_type' must be 'classification' or 'regression'."

	target = df[close_column_name].shift(-abs(target_shift))

	if target_type.lower() == "regression":
		df_to_treat['target'] = target

	else: ### if target type is classification:
		df_target = pd.DataFrame({close_column_name:df[close_column_name], 'target':target})
		df_target['sens'] = df_target['target'] - df_target[close_column_name]
		list_ = df_target[['sens']]
		list_.dropna(inplace = True)
		list_ = list_['sens'].tolist()

		range_neutral_trend_ratio_get = get_range_neutral_trend(list_ = list_,
													ratio_true_trend = ratio_true_trend,
													diviseur_ecart_entre_list_values = diviseur_ecart_entre_list_values)
		range_neutral_trend = range_neutral_trend_ratio_get["interval"]
		ratio_get = range_neutral_trend_ratio_get["ratio_get"]

		if (ratio_true_trend - ratio_get)*100 >= 5:
			good_ratio = False
			text_ = f"Veuillez augmenter la valeur de 'diviseur_ecart_entre_list_values' car le ratio obtenu ratio_get: {ratio_get} est assez écarté de celui voulu, ratio_true_trend: {ratio_true_trend}"
			assert good_ratio == True, text_
		# elif ratio_get - ratio_true_trend >= 5:
		# 	good_ratio = False
		# 	text_ = "Veuillez diminuer la valeur de 'diviseur_ecart_entre_list_values' car le ratio obtenu est assez écarté de celui voulu."
		# 	assert good_ratio == True, text_

		target = np.where(df_target['sens'] < range_neutral_trend[0], -1,
								np.where(df_target['sens'] > range_neutral_trend[1], 1, 0.0))
		df_to_treat['target'] = target
	if drop_added_nan:
		stop_at = len(df_to_treat) - target_shift
		# df_to_treat.reset_index(inplace = True, drop = True)
		df_to_treat = df_to_treat.head(stop_at)

	return df_to_treat



def split_train_test(df, train_ratio):
	df = df.copy()
	train_size = int(train_ratio * len(df))
	test_size = int((1-train_ratio)*len(df))
	df_train = df.head(train_size)
	df_test = df.tail(test_size)
	df_train.reset_index(inplace = True, drop = True)
	df_test.reset_index(inplace = True, drop = True)
	result = {'df_train':df_train, 'df_test':df_test}
	return result


def split_x_y(df, data_cols_names, target_col_name, look_back):
	df = df.copy()
	assert look_back >= 1, "The 'look_back' value must be >= 1"
	df.reset_index(inplace = True, drop = True)
	dataX = []
	dataY = []
	for idx in df.index:
		fragment = df.iloc[idx:idx+look_back, :]
		fragment_X = np.array(fragment[data_cols_names])
		fragment_Y = np.array(fragment[target_col_name].tolist()[-1])
		if len(fragment) < look_back:
			break
		dataX.append(fragment_X)
		dataY.append(fragment_Y)

	dataX = np.array(dataX)
	dataY = np.array(dataY)
	return {"dataX":dataX, "dataY":dataY}


def split_x_y_of_frgts(df_fragments, look_back, target_col_name, data_cols_names):
	assert look_back >= 1, "The 'look_back' value must be >= 1"
	dataX = []
	dataY = []
	for df_frgt_ in df_fragments:
		df_frgt_selected = df_frgt_.tail(look_back)
		fragment_selected_X = np.array(df_frgt_selected[data_cols_names])
		fragment_selected_Y = np.array(df_frgt_selected[target_col_name].tolist()[-1])
		dataX.append(fragment_selected_X)
		dataY.append(fragment_selected_Y)

	dataX = np.array(dataX)
	dataY = np.array(dataY)
	return {"dataX":dataX, "dataY":dataY}


def handle_columns_each_fragment(df, freqs_seasonal, look_back, target_col_name, close_column_name):
	df = df.copy()
	df.reset_index(inplace = True, drop = True)

	message_not_enough_len = "la longueur du dataframe est insuffisante !"
	preventor_ecart = 3
	assert df.shape[0] >= max(freqs_seasonal) * preventor_ecart, message_not_enough_len
	assert df.shape[0] >= look_back * preventor_ecart, message_not_enough_len

	len_fragments = max(max(freqs_seasonal) * preventor_ecart, look_back * preventor_ecart)
	nbr_fragments = df.shape[0] - len_fragments + 1
	idxs_from = list(range(nbr_fragments))
	idxs_to = [idx + len_fragments - 1 for idx in idxs_from]

	fragments_lens_check = []
	frgts_handled = []
	# for idx_from, idx_to in zip(idxs_from, idxs_to):
	# 	df_fragment = df.loc[idx_from:idx_to]
	# 	df_fragment.reset_index(inplace = True, drop = True)
	# 	df_frgt_ = handle_columns_v2(df = df_fragment,
	# 		close_column_name = close_column_name, 
	# 		target_col_name = target_col_name, 
	# 		freqs_seasonal = freqs_seasonal)
	# 	df_frgt_.reset_index(inplace = True, drop = True)
	# 	frgts_handled.append(df_frgt_)
	# 	df_frgt_check = df_frgt_.copy()
	# 	df_frgt_check.dropna(inplace = True)
	# 	fragments_lens_check.append(df_frgt_check.shape[0])

	k_range = len(idxs_from)
	step_k_range = 1
	for idx_from, idx_to in zip(idxs_from, idxs_to):
		start_timestep = time.time()
		df_fragment = df.loc[idx_from:idx_to]
		df_fragment.reset_index(inplace = True, drop = True)
		df_frgt_ = handle_columns_v2(df = df_fragment,
			close_column_name = close_column_name, 
			target_col_name = target_col_name, 
			freqs_seasonal = freqs_seasonal)
		df_frgt_.reset_index(inplace = True, drop = True)
		frgts_handled.append(df_frgt_)
		df_frgt_check = df_frgt_.copy()
		df_frgt_check.dropna(inplace = True)
		fragments_lens_check.append(df_frgt_check.shape[0])

		taken_timestep = time.time() - start_timestep
		print("Step : ", step_k_range, " out of ", k_range, "--- Taken timestep :", taken_timestep, " s")
		step_k_range += 1

	fragments_lens_check = list(set(fragments_lens_check))
	assert len(fragments_lens_check) == 1, 'les longueurs des df_frgt_ s ne sont pas identiques !'
	assert fragments_lens_check[0] >= len_fragments-1, "il y a au-moins un NaN dans au-moins un df_frgt_ !"	
	return frgts_handled


def scale_fragments(df_fragments):
	return [data_scaler(df = item)[0] for item in df_fragments]



###############################################################
######################################################################
############################################################################
######################################################################
###############################################################

def manage_frame(frame, data_cols_names, 
				close_column_name, freqs_seasonal, 
				target_col_name, look_back):

	assert target_col_name in frame.columns.tolist(), "The frame must have already target column."

	frame = frame.copy()

	### HANDLE COLUMNS:
	frame = handle_columns_v2(df = frame,
							close_column_name = close_column_name, 
							freqs_seasonal = freqs_seasonal, 
							target_col_name = target_col_name)

	### SCALE:
	frame, _ = data_scaler(df = frame)

	### SPLIT X AND Y:
	data_x_y = split_x_y(df = frame, 
		data_cols_names = data_cols_names, 
		target_col_name = target_col_name, 
		look_back = look_back)

	x_ = data_x_y['dataX']
	y_ = data_x_y['dataY']

	### SPLIT TRAIN AND TEST:
	X_train = x_[:-1]
	y_train = y_[:-1]

	X_test = x_[x_.shape[0]-1:x_.shape[0]]
	y_test = y_[y_.shape[0]-1:y_.shape[0]]

	return {"X_train":X_train,
			"y_train":y_train,
			"X_test":X_test,
			"y_test":y_test}

def load_simulated_model(model_filepath):
	try:
		with open(model_filepath, "rb") as f:
			loaded_ = pickle.load(f)
		return loaded_
	except FileNotFoundError:
		return False

class SimulatorModel:
	def __init__(self):
		pass

	def fit(self, X_train, y_train, 
			epochs, batch_size, 
			validation_split, 
			shuffle, verbose, 
			model_filepath = None,
			callbacks = None):

		x_gfyrjmxnbhfifjd = random.randint(100, 900)

		losses = []
		val_losses = []
		loss = random.randint(80,90)/10000
		val_loss = random.randint(85,100)/10000

		for epoch in range(epochs):
			loss /= 1.2
			val_loss /= 1.1
			if verbose:
				print(f"Epoch {epoch+1}/{epochs}")
				print(f"{x_gfyrjmxnbhfifjd}/{x_gfyrjmxnbhfifjd}[=============================] - {random.randint(1,3)/10}s - {random.randint(1,3)/1000}ms/step - loss: {round(loss, 5)} - val_loss: {round(val_loss, 5)}")
			time.sleep(random.randint(1,2)/10_000)

			losses.append(loss)
			val_losses.append(val_loss)

		if model_filepath is not None:
			with open(model_filepath, 'wb') as f:
				pickle.dump(SimulatorModel(), f)
			print(f"\nSimulated Model {model_filepath} successfully saved !\n")

		return {'loss':losses, 'val_loss':val_losses}

	def compile(self, loss, optimizer):
		# model.compile(loss='mean_squared_error', optimizer='adam')
		pass

	# def add():
	# 	pass

	# def save():
	# 	pass

	def predict(self, X_test):
		y_pred = [random.uniform(0, 1) for _ in range(X_test.shape[0])]
		y_pred = np.array(y_pred)
		return y_pred

def get_accuracy(df_results, 
	close_column_name, target_shift, y_pred_col_name, verbose, save_to = None):
	
	def get_rapprochement(y_pred):
		diff_0 = abs(y_pred - 0.0)
		diff_0_5 = abs(y_pred - 0.5)
		diff_1 = abs(y_pred - 1.0)

		all_diffs = [diff_0, diff_0_5, diff_1]
		min_ = min(all_diffs)
		if min_ == diff_0:
			return 0.0
		elif min_ == diff_0_5:
			return 0.5
		else:
			return 1.0

	df = df_results.copy()
	len_df = df.shape[0]
	df['future_price'] = df[close_column_name].shift(-abs(target_shift))
	df['true_sens'] = df['future_price'] - df[close_column_name]
	df['true_sens'] = np.where(df['true_sens'] > 0, 1.0,
						np.where(df['true_sens'] < 0, 0.0, 0.5))

	df.dropna(subset = ['future_price'], inplace = True)
	df.reset_index(inplace = True, drop = True)

	df['y_pred_rppchmt'] = list(map(get_rapprochement, df[y_pred_col_name]))

	### IF THE MARKET IS NEUTRAL: WE DON'T LOSE AND DON'T WIN ANYTHING, SO LET'S DROP THE
	### ROWS WHERE TRUE SENS == 0.5:

	df = df[df['true_sens'] != 0.5]

	### IF THE SIGNAL y_pred_rppchmt IS 0.5, WE DON'T TAKE ANY POSITION, SO LET'S DROP THE ROWS
	### WHERE y_pred_rppchmt == 0.5:

	df = df[df['y_pred_rppchmt'] != 0.5]

	df['result'] = df['y_pred_rppchmt'] == df['true_sens']
	wins = df['result'].tolist().count(True)
	loses = df['result'].tolist().count(False)
	total_trades = wins + loses
	try:
		accuracy = wins/total_trades
		accuracy = str(round(accuracy*100, 2)) + " %"
	except ZeroDivisionError:
		accuracy = 'No trade taken'

	try:
		ratio_trades = total_trades/len_df
		ratio_trades = str(round(ratio_trades*100, 2)) + " %"
	except ZeroDivisionError:
		ratio_trades = 'No trade taken'

	print("\n\n\tPlease wait. \n\tComputing balance sheet ...")
	if verbose:
		print("\t\t\t\tRatio Trades :", ratio_trades, "\n")
		print("\t\t\t\tDf length	:", len_df)
		print("\t\t\t\tWins		 :", wins)
		print("\t\t\t\tLoses		:", loses)
		print("\t\t\t\tAll Trades   :", wins + loses)
		print("\t\t\t\tAccuracy	 :", accuracy, "\n")

	if save_to is not None:
		with open(save_to, "w", encoding = "utf-8") as f:	
			# f.write(f"\n\tTotal Trades : {total_trades}")
			f.write(f"\n\tRatio Trades : {ratio_trades}\n")
			f.write(f"\tDf length	: {len_df}\n")
			f.write(f"\tWins		 : {wins}\n")
			f.write(f"\tLoses		: {loses}\n")
			f.write(f"\tAll trades   : {wins + loses}\n")
			f.write(f"\tAccuracy	 : {accuracy}\n")


def get_fit_predict_frames_n_save_result(len_frame, waitings,
						target_col_name, look_back, test_nbr,
						freqs_seasonal, epochs, batch_size_fit,
						validation_split_fit, shuffle_fit,
						verbose_fit, close_column_name,
						verbose_model_checkpoint,
						target_shift, verbose_eval_results,
						target_type, ratio_true_trend, 
						fitting_modulo, modulos_save_landmark,
						dataset_key, df_len = None,
						head_or_tail = None,
						reset_idx_frame = False,
						):
	
	print_style(f"\n  Test number: {test_nbr}\n", color = 'yellow', bold = True)

	manage_wargins()
	if uname.node == "Gilbert-PC":
		df = pd.read_csv(r'C:\Users\LENOVO\AppData\Local\Programs\Python\Python37-32\Lib\site-packages\projects\trading_and_ai\data\recent_data\EURUSD-120.0 Min--2024-3-18 9-0-0.csv')
		color = None
		informative_color = color
		alert_color = color
		good_color = color
		bold = False
		# underline = False

	if 'COLAB_GPU' in env_ or 'COLAB_JUPYTER_IP' in env_:
		informative_color = 'cyan'
		alert_color = 'red'
		good_color = 'green'
		bold = True

		common_path = "https://github.com/GilbertAK/eurusd_data/blob/main/"
		dataset_urls = {"dataset_test1":"https://github.com/GilbertAK/EURUSD_2021_10_1_2021_11_15_ohlcv_1_min/blob/main/EURUSD_2021_10_1_2021_11_15_ohlcv_1_min-.csv",
			"dataset_test12":common_path + "EURUSD-2023-07-25_2023-08-31.csv",
			"dataset_test13":common_path + "EURUSD-2023-08-23_2023-09-29.csv",
			"dataset_test14":common_path + "EURUSD-2023-09-22_2023-10-31.csv",
			"dataset_test15":common_path + "EURUSD-2023-10-24_2023-11-30.csv",
			"dataset_test16":common_path + "EURUSD-2023-11-21_2023-12-29.csv",
			"dataset_test17":common_path + "EURUSD-2023-12-13_2024-01-23.csv",
			"dataset_5_min":"https://firebasestorage.googleapis.com/v0/b/saving-data-2ee4b.appspot.com/o/drown-in-bigdata%2Fdatasets%2FEURUSD-5.0%20Min--2024-3-8%2022-0-0.csv?alt=media",
			"dataset_1_hour":"https://firebasestorage.googleapis.com/v0/b/saving-data-2ee4b.appspot.com/o/drown-in-bigdata%2Fdatasets%2FEURUSD-60.0%20Min--2024-3-18%209-0-0.csv?alt=media",
			"dataset_2_hours":"https://firebasestorage.googleapis.com/v0/b/saving-data-2ee4b.appspot.com/o/drown-in-bigdata%2Fdatasets%2FEURUSD-120.0%20Min--2024-3-18%209-0-0.csv?alt=media",
			}

		url_dataset = dataset_urls[dataset_key]
		try:
			df = pd.read_csv(url_dataset)
		except:
			url_dataset += "?raw=true"
			df = pd.read_csv(url_dataset)

	### CHECK IF GOOGLE DRIVE IS ALREADY CONNECTED:
	###____________________________________________
	if 'COLAB_GPU' in env_ or 'COLAB_JUPYTER_IP' in env_:
		start_time_while_drive = time.time()
		print_style('\nDrive not yet connected !', color = alert_color, bold = bold)
		while True:
			if not os.path.isdir(s = "/content/drive/MyDrive/"):
				time.sleep(1)			
			else:
				print_style("\nOkay, Drive successfully connected !", color = good_color, bold = bold)
				elasped_while_drive = time.time() - start_time_while_drive
				print_style(f"Elapsed time : {round(elasped_while_drive, 2)} seconds or {round(elasped_while_drive/60, 2)} minutes.\n", 
					color = informative_color, bold = bold)
				break

	### TAKE TAIL OR HEAD DF AND RESET_INDEX:
	###_______________________________________
	assert head_or_tail == 'head' or head_or_tail == 'tail' or head_or_tail == None, '"head_or_tail" must be "head", "tail" or None.'
	if head_or_tail == 'head':
		df = df.head(df_len)
	elif head_or_tail == 'tail':
		df = df.tail(df_len)
	df.reset_index(inplace = True, drop = True)

	### KEEP THE TRUE CLOSE COLUMN NAME:
	###_________________________________

	true_close = df[close_column_name]

	### PREVENT SOME ERRORS:
	###_____________________

	preventor_ecart_1 = 2
	assert len_frame >= preventor_ecart_1 * look_back, f'"len_frame" doit être >= à {preventor_ecart_1}*look_back.'
	preventor_ecart_2 = 3
	assert len_frame >= preventor_ecart_2 * max(freqs_seasonal), f'"len_frame" doit être >= à {preventor_ecart_2}*max(freqs_seasonal).'
	assert len_frame < df.shape[0], '"len_frame" doit être < à len df.'

	### PREPARE INDEXES:
	###_________________

	nbr_frames = df.shape[0] - len_frame + 1
	idxs_from = list(range(nbr_frames))
	total_nbr_steps = len(list(range(nbr_frames)))
	idxs_to = [item + len_frame -1 for item in idxs_from]
	### save the original idxs_from to be used natively later:
	original_idxs_from = idxs_from

	### PREPARE GENERAL PATHS:
	###_______________________

	general_path = "result_data/"
	if 'COLAB_GPU' in env_ or 'COLAB_JUPYTER_IP' in env_:
		general_path = "/content/drive/MyDrive/"

	### PREPARE FILEPATHS:
	###___________________

	y_preds_filepath = general_path + "y_preds_test_nbr_" + test_nbr + ".pkl"
	y_tests_filepath = general_path + "y_tests_test_nbr_" + test_nbr + ".pkl"
	times_taken_fitting_filepath = general_path + "times_taken_fitting_test_nbr_" + test_nbr + ".pkl"
	iteration_times_filepath = general_path + "iteration_times_test_nbr_" + test_nbr + ".pkl"

	### LOAD LANDMARKS FOR FIREBASE:
	###_____________________________
	files_landmarks = [
		f'finished_steps_test_nbr_{test_nbr}.pkl',
		f'y_preds_test_nbr_{test_nbr}.pkl',
		f'y_tests_test_nbr_{test_nbr}.pkl',
		f'times_taken_fitting_test_nbr_{test_nbr}.pkl',
		f'iteration_times_test_nbr_{test_nbr}.pkl',
		f'model_test_nbr_{test_nbr}.h5',
		f'saved_last_epoch_filepath_test_nbr_{test_nbr}.txt',]

	if 'COLAB_GPU' in env_ or 'COLAB_JUPYTER_IP' in env_:
		firebase_storage = FirebaseStorage(firebase_config = firebase_config)

		not_all_landmarks = 0
		for f in files_landmarks:
			print_style(f'Trying do download {f} ...', 
						color = informative_color, bold = bold)
			firebase_storage.download_file(
							cloud_file_path_name = "instantanely/landmarks/" + f, 
							local_file_path_name = general_path + f,
							)
			time.sleep(waitings[0])
			if check_file_exists(filepath = general_path + f):
				print_style(f"{f} successfully loaded !\n",
							color = good_color, bold = bold)
			else:
				not_all_landmarks += 1
				print_style(f"{f} wasn't downloaded.\n", 
							color = alert_color, bold = bold)

		if not_all_landmarks > 0:
			print_style(f"\nOne or Many landmark(s) is/are not downloaded.\n\tSo waiting for {waitings[1]} second(s).\n",
						color = informative_color, bold = bold)
			time.sleep(waitings[1])

	### LOAD LANDMARKS:
	###________________

	y_preds = load_pickle(filepath = y_preds_filepath)
	if not y_preds:
		y_preds_existed = False
		y_preds = []
	else:
		y_preds_existed = True

	y_tests = load_pickle(filepath = y_tests_filepath)
	if not y_tests:
		y_tests_existed = False
		y_tests = []
	else:
		y_tests_existed = True

	times_taken_fitting = load_pickle(filepath = times_taken_fitting_filepath)	
	if not times_taken_fitting:
		times_taken_fitting_existed = False
		times_taken_fitting = []
	else:
		times_taken_fitting_existed = True

	iteration_times = load_pickle(filepath = iteration_times_filepath)
	if not iteration_times:
		iteration_times_existed = False
		iteration_times = []
	else:
		iteration_times_existed = True

	### DATA COLUMNS NAMES:
	###_____________________

	data_cols_names_seasonality = [f'seasonality_{d}' for d in freqs_seasonal]
	data_cols_names = ['close', 
						'stationnarized_close',
						'soft_0.5', 
						'less_0.5', 
						'soft_0.5_stationnarized',
						'dwt_cA', 
						'dwt_cD', 
						'dwt_cA_stationnarized']
	data_cols_names += data_cols_names_seasonality

	### PREPARE COMPLETION FILEPATH:
	###_____________________________

	if 'COLAB_GPU' in env_ or 'COLAB_JUPYTER_IP' in env_:
		completion_path = f"/content/drive/MyDrive/finished_steps_test_nbr_{test_nbr}.pkl"
	if uname.node == 'Gilbert-PC':
		completion_path = f"result_data/finished_steps_test_nbr_{test_nbr}.pkl"

	### GET LAST STEP:
	###_______________

	last_step = load_pickle(filepath = completion_path)
	if last_step:
		last_step_existed = True
		idxs_from = [d for d in idxs_from if d > last_step[0]]
		idxs_to = [d for d in idxs_to if d > last_step[1]]

		passed_steps = len([val for val in original_idxs_from if val <= last_step[0]])
		assert len(y_preds) == passed_steps, "length of y_preds must be equals to passed_steps"
		assert len(y_tests) == passed_steps, "length of y_tests must be equals to passed_steps"
		assert len(times_taken_fitting) == passed_steps, "length of times_taken_fitting must be equals to passed_steps"
		assert len(iteration_times) == passed_steps, "length of iteration_times must be equals to passed_steps"

	else:
		last_step_existed = False

	### SIGNAL IF ALL LANDMARKS HAVE BEEN FOUND OR NOT:
	###________________________________________________
	if y_preds_existed and y_tests_existed and times_taken_fitting_existed and iteration_times_existed and last_step_existed:
		print_style("Okay, All landmarks found.", 
			color = good_color, bold = bold)
		time.sleep(waitings[0])

	else:
		for _ in range(5):
			print_style("Attention !\n\tSome or all landmarks are not found !\n\tAll execution is going to start from zero !\n",
				color = alert_color, bold = bold)
		time.sleep(waitings[1])

	######################################################################
	### Vérifier si les longueurs de idxs_from et idxs_to sont identiques:
	assert len(idxs_from) == len(idxs_to), "length of idxs_from shall be == to length of idxs_to"

	#########################################
	### SIGNAL IF CODE IS ENTIRELY EXECUTED:
	if len(idxs_from) == 0:
		print(f"\n\n\tCode execution is already finished for test nbr: {test_nbr} !!!\n\n")

	##########################################
	### RUN NEXT PART OF LOOP IF NOT FINISHED:
	elif len(idxs_from) > 0:
		###########################################################
		### Insure that lengths of y_preds and y_tests are identic:
		assert len(y_preds) == len(y_tests), "Lengths of y_preds and y_tests shall be identic."
		
		####################################
		### Initialize the progression_step:
		progression_step = 1
		for idx_from, idx_to in zip(idxs_from, idxs_to):
			true_idx_from = idx_from
			iteration_time_start = time.time()

			if reset_idx_frame:
				frame = df.loc[0:idx_to, :]
			else:
				frame = df.loc[idx_from:idx_to, :]
			frame.reset_index(inplace = True, drop = True)

			### ADD TARGET COLUMN:
			frame = add_target(df = frame,
				close_column_name = close_column_name, 
				target_type = target_type,
				drop_added_nan = False,
				target_shift = target_shift,
				ratio_true_trend = ratio_true_trend)

			data_x_y_train_test = manage_frame(frame = frame,
												data_cols_names = data_cols_names, 
												target_col_name = target_col_name, 
												look_back = look_back,
												close_column_name = close_column_name,
												freqs_seasonal = freqs_seasonal,)

			X_train = data_x_y_train_test["X_train"]
			X_test = data_x_y_train_test["X_test"]
			y_train = data_x_y_train_test["y_train"]
			y_test = data_x_y_train_test["y_test"]

			model_filename = 'model_test_nbr_' + f'{test_nbr}.h5'
			model_filepath = general_path + model_filename
			saved_last_epoch_filepath = general_path + f'saved_last_epoch_filepath_test_nbr_{test_nbr}.txt'
			
			if isinstance(epochs, tuple):
				epochs_first_training = epochs[0]
				epochs_definitive = epochs[1]
				refit = True
			if isinstance(epochs, int):
				epochs_first_training = epochs
				epochs_definitive = epochs
				refit = False
			standard_epochs_first_training = epochs_first_training

			# print(f"Checking presence of: {model_filepath}")
			model_filepath_exists = False
			for _ in range(5):
				if check_file_exists(filepath = model_filepath):
					model_filepath_exists = True
					break
				else:
					# time.sleep(1)
					pass

			# print(f"Checking presence of: {saved_last_epoch_filepath}")
			saved_last_epoch_filepath_exists = False
			for _ in range(5):
				if check_file_exists(filepath = saved_last_epoch_filepath):
					saved_last_epoch_filepath_exists = True
					break
				else:
					# time.sleep(1)
					pass

			if model_filepath_exists and saved_last_epoch_filepath_exists:
				with open(saved_last_epoch_filepath, "r", encoding = "utf-8") as f:
					last_epochs = f.readlines()
				last_epochs = [item.strip() for item in last_epochs]
				all_previous_epochs_nbr = len(last_epochs)
			else:
				all_previous_epochs_nbr = 0

			if refit and all_previous_epochs_nbr < standard_epochs_first_training:
				#### FIT THE MODEL TO BE SAVED AND REUSED AFTER:
				if 'COLAB_GPU' in env_ or 'COLAB_JUPYTER_IP' in env_:
					if model_filepath_exists and saved_last_epoch_filepath_exists:
						from tensorflow.keras.models import load_model
						model = load_model(model_filepath)
						epochs_first_training -= all_previous_epochs_nbr
						print_style(text = f"\nOkay {model_filepath} \n\t\tand\n\t{saved_last_epoch_filepath} found !!!\n",
									color = good_color, 
									bold = bold)
						time.sleep(waitings[0])

					else:
						print("\n")
						for _ in range(5):
							print_style(text = "Fitting model will start from zero !!!", 
										color = alert_color, 
										bold = bold)
						print("\n")

						time.sleep(waitings[1])

						dropout = 0.2
						model = keras.Sequential()
						model.add(keras.layers.Bidirectional(
							keras.layers.LSTM(units=128, input_shape=(X_train.shape[1], X_train.shape[2]))))

						model.add(keras.layers.Dropout(rate=dropout))
						model.add(keras.layers.Dense(units=1))
						model.compile(loss='mean_squared_error', optimizer='adam')

						print_style("\tKeras model is created.\n", 
									color = good_color, 
									bold = bold)

					model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath = model_filepath,
													monitor = 'loss',
													mode = 'min',
													save_best_only = True,
													verbose = verbose_model_checkpoint)
					
					each_epoch = EachEpochCallback(epochs = epochs_first_training, 
												test_nbr = test_nbr,
												modulos_save_landmark = modulos_save_landmark,
												general_path = general_path,
												saved_last_epoch_filepath = saved_last_epoch_filepath,
												good_color = good_color,
												alert_color = alert_color,
												bold = bold,
												)
					history = model.fit(X_train,
										y_train,
										epochs = epochs_first_training,
										batch_size = batch_size_fit,
										validation_split = validation_split_fit,
										shuffle = shuffle_fit,
										verbose = True,
										callbacks = [model_checkpoint_callback, each_epoch],
										)

				if uname.node == "Gilbert-PC":
					model = SimulatorModel()
					history = model.fit(X_train,
										y_train,
										epochs = epochs_first_training,
										batch_size = batch_size_fit,
										validation_split = validation_split_fit,
										shuffle = shuffle_fit,
										verbose = True,
										model_filepath = model_filepath,
										# ### ### callbacks = [model_checkpoint_callback, each_epoch_callback],
										)

					for epoch_ in range(epochs_first_training):
						with open(saved_last_epoch_filepath, "a", encoding = "utf-8") as f:
							f.write(f'{epoch_+1}\n')

			### FIT THE MODEL:
			diff_progr = total_nbr_steps - len(idxs_from)
			step_progr_now = progression_step + diff_progr
			print_style(f"\n\tProgression : {step_progr_now}/{total_nbr_steps}\n", color = informative_color, bold = bold)
			
			assert fitting_modulo > 0, '"fitting_modulo" must be > 0.'
			if (fitting_modulo > 1 and (idx_from+1) == 1) or ((idx_from+1)%fitting_modulo == 0):
				fit_model = True
			else:
				fit_model = False

			if fit_model:
				start_time_fitting = time.time()

				if refit:					
					model_is_already_exists_new_check = check_file_exists(filepath = model_filepath)
					
					if model_is_already_exists_new_check:
						### LOAD THE MODEL:
						if 'COLAB_GPU' in env_ or 'COLAB_JUPYTER_IP' in env_:
							model = load_the_model(model_filepath = model_filepath)

							print_style(f"\tKeras Model {model_filename} is successfully loaded !",
										color = good_color, bold = bold)

							if verbose_fit:
								print("\n")

						if uname.node == "Gilbert-PC":
							model = load_simulated_model(model_filepath = model_filepath)

							for _ in range(3):
								print_style(f"\tSimulated model {model_filename} is successfully loaded !",
											color = alert_color, bold = bold)

							if verbose_fit:
								print("\n")

					if not model_is_already_exists_new_check:
						for _ in range(10):
							print_style(f"We should load and use the model {model_filename} but it seems that it hadn't been saved !",
								color = alert_color, bold = bold)
						time.sleep(waitings[0])

				else:
					### REINITIALIZE THE MODEL:
					if 'COLAB_GPU' in env_ or 'COLAB_JUPYTER_IP' in env_:
						dropout = 0.2
						model = keras.Sequential()
						model.add(keras.layers.Bidirectional(
								keras.layers.LSTM(units=128, 
									input_shape=(X_train.shape[1], X_train.shape[2]))))
						model.add(keras.layers.Dropout(rate=dropout))
						model.add(keras.layers.Dense(units=1))
						model.compile(loss='mean_squared_error', optimizer='adam')
						# for _ in range(10):
						print_style("\n\tKeras model is created.\n", color = good_color, bold = bold)

					if uname.node == 'Gilbert-PC':
						model = SimulatorModel()
						# for _ in range(10):
						print_style("\n\tWe are using a simulated model !!!\n", color = alert_color, bold = bold)

				###### definitive fitting / so just before prediction:
				if not verbose_fit:
					print_style("\n\tPlease wait, fitting model ...\n", color = informative_color, bold = bold)
				history = model.fit(X_train,
									y_train,
									epochs = epochs_definitive,
									batch_size = batch_size_fit,
									validation_split = validation_split_fit,
									shuffle = shuffle_fit,
									verbose = verbose_fit,
									# callbacks = [model_checkpoint_callback],
									)

				time_taken_fitting = round(time.time() - start_time_fitting, 5)
				times_taken_fitting.append(time_taken_fitting)

			### USE THE MODEL TO MAKE PREDICTION:
			y_pred = model.predict(X_test)

			### CHECKING
			assert y_pred.shape[0] == 1, 'Il devrait y avoir une seule valeur de y_pred.'
			assert y_test.shape[0] == 1, 'Il devrait y avoir une seule valeur de y_test.'

			### APPENDING DATA INTO y_preds AND y_tests LISTS:
			if uname.node == "Gilbert-PC":
				y_preds.append(y_pred[0])
				y_tests.append(y_test[0])
			if 'COLAB_GPU' in env_ or 'COLAB_JUPYTER_IP' in env_:
				y_preds.append(y_pred[0][0].astype(float))
				y_tests.append(y_test[0].astype(float))

			### save the progression
			arrived_to = true_idx_from, idx_to

			###################
			### SAVE LANDMARKS:
			###________________

			### 1/2) SAVE TO GOOGLE DRIVE:
			###____________________________

			iteration_time_taken = time.time() - iteration_time_start
			iteration_times.append(iteration_time_taken)

			save_pickle(filepath = completion_path, data = arrived_to)			
			save_pickle(filepath = y_preds_filepath, data = y_preds)
			save_pickle(filepath = y_tests_filepath, data = y_tests)
			save_pickle(filepath = times_taken_fitting_filepath, data = times_taken_fitting)
			save_pickle(filepath = iteration_times_filepath, data = iteration_times)

			### 2/2) SAVE TO FIREBASE:
			###________________________

			if 'COLAB_GPU' in env_ or 'COLAB_JUPYTER_IP' in env_:
				if progression_step%modulos_save_landmark['short'] == 0:

					files_landmarks = [
						f'finished_steps_test_nbr_{test_nbr}.pkl',
						f'y_preds_test_nbr_{test_nbr}.pkl',
						f'y_tests_test_nbr_{test_nbr}.pkl',
						f'times_taken_fitting_test_nbr_{test_nbr}.pkl',
						f'iteration_times_test_nbr_{test_nbr}.pkl',
						f'model_test_nbr_{test_nbr}.h5',
						f'saved_last_epoch_filepath_test_nbr_{test_nbr}.txt',
						]

					print_style("\nUploading landmarks to firebase storage ...", color = informative_color)
					files_2_upload = 0
					for f in files_landmarks:
						firebase_storage = FirebaseStorage(firebase_config = firebase_config)
						try:
							result_file_url_ = firebase_storage.upload_file(
										local_file_path_name = general_path + f, 
										cloud_file_path_name = "instantanely/landmarks/" + f,
										)
							files_2_upload += 1

							# print_style(f"{f} uploaded. Its url is: \n\t{result_file_url_}",
							# 	color = good_color, bold = bold)

						except FileNotFoundError:
							print_style(f"No file named: {f}, in Google Drive !!!",
								color = alert_color, bold = bold)

					if files_2_upload == len(files_landmarks):
						print_style(f"All {files_2_upload} landmarks files are successfully uploaded.", 
										color = good_color, bold = bold)
					elif files_2_upload != len(files_landmarks):
						print_style("\n\tOne or some file(s) was/were not uploaded.",
										color = alert_color, bold = bold)

			### PRINTING TIMES:
			###_________________

			avg_iteration_times = sum(iteration_times)/len(iteration_times)
			staying_steps = total_nbr_steps - len(iteration_times)
			staying_time = staying_steps*avg_iteration_times
			staying_time_seconds = staying_time

			if staying_time < 60:
				staying_time = str(round(staying_time, 2)) + " second(s)"

			elif staying_time >= 60 and staying_time < 3600:
				staying_time /= 60
				staying_time = str(round(staying_time, 3)) + " minute(s)"

			elif staying_time >= 3600 and staying_time < 86400:
				staying_time /= 3600
				staying_time = str(round(staying_time, 5)) + " hour(s)"

			elif staying_time >= 86400:
				staying_time /= 86400
				staying_time = str(round(staying_time, 5)) + " day(s)"

			time_now = time.time()
			finish_at_ = staying_time_seconds + time_now
			if 'COLAB_GPU' in env_ or 'COLAB_JUPYTER_IP' in env_:
				time_now += 7200
				finish_at_ += 7200
			datetime_now = datetime.datetime.fromtimestamp(int(time_now))
			finish_at_ = datetime.datetime.fromtimestamp(int(finish_at_))

			if verbose_fit:
				print("\n")

			print_style(f"Datetime now      : {datetime_now}", color = informative_color, bold = bold)
			print_style(f"Finish at         : {finish_at_}", color = informative_color, bold = bold)
			print_style(f"Staying time      : {staying_time}", color = informative_color, bold = bold)
			print_style("\n______________________________________________\n",
											color = informative_color, bold = bold)

			# ####### if idx_from+1 >= 7:
			# 	break

			### Increment the progression_step
			progression_step += 1


		############################
		############################
		############################
		############################
		############################
		############################
		############################
		############################
		############################
		############################
		######### ENDING OF THE LOOP

		### ARRANGE TO CLOSE:
		###___________________
		true_close = true_close.tail(len(y_preds)).tolist()

		### GET ACCURACY:
		df_results = pd.DataFrame({"y_pred":y_preds, "true_close":true_close})

		save_balance_df_results_to = general_path + f"Df_results_test_nbr_{test_nbr}.csv"
		df_results.to_csv(save_balance_df_results_to, index = False)

		save_balance_sheet_to = general_path + f"Balance_sheet_test_nbr_{test_nbr}.txt"
		get_accuracy(df_results = df_results, 
					close_column_name = 'true_close',
					y_pred_col_name = 'y_pred',
					target_shift = target_shift,
					verbose = verbose_eval_results,
					save_to = save_balance_sheet_to,
					)
		
		if 'COLAB_GPU' in env_ or 'COLAB_JUPYTER_IP' in env_:
			files_results =  [f'finished_steps_test_nbr_{test_nbr}.pkl',
					f'iteration_times_test_nbr_{test_nbr}.pkl',
					f'times_taken_fitting_test_nbr_{test_nbr}.pkl',
					f'y_preds_test_nbr_{test_nbr}.pkl',
					f'y_tests_test_nbr_{test_nbr}.pkl',
					f'model_test_nbr_{test_nbr}.h5',
					f'saved_last_epoch_filepath_test_nbr_{test_nbr}.txt',
					f'Df_results_test_nbr_{test_nbr}.csv',
					f'Balance_sheet_test_nbr_{test_nbr}.txt',]

			print_style(f"\nThere are {len(files_results)} files to upload to firebase storage :", 
				color = informative_color)
			for elt in files_results:
				print_style(f"\t- {elt}", color = informative_color)
			print("\n")

			for f in files_results:
				firebase_storage = FirebaseStorage(firebase_config = firebase_config)
				try:
					result_file_url = firebase_storage.upload_file(
								local_file_path_name = general_path + f, 
								cloud_file_path_name = "instantanely/results/" + f,
								)

					print(f"Result File {f} Uploaded on url: \n\t", result_file_url)
				except FileNotFoundError:
					print(f"No file named: {f}, in Google Drive !!!")

		print_style('\n\tFINISHED.\n', color = good_color, bold = bold)
