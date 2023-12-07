import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import stft
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import keras
import librosa as lr
from spafe.features.gfcc import gfcc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import average_precision_score
import time
import os

# Define the directory name of the audio dataset
feature_method = ['STFT']
dataset_name = ['6_dB_slider/id_00'] 
epochs_list = [5]
'''
feature_method = ['MFCC','STFT','GFCC','Mel_spectrogram']

dataset_name = ['-6_dB_slider/id_00', '-6_dB_slider/id_02', '-6_dB_slider/id_04', '-6_dB_slider/id_06', 
            '0_dB_slider/id_00', '0_dB_slider/id_02',  '0_dB_slider/id_04',  '0_dB_slider/id_06', 
            '6_dB_slider/id_00', '6_dB_slider/id_02',  '6_dB_slider/id_04',  '6_dB_slider/id_06',
            '-6_dB_fan/id_00', '-6_dB_fan/id_02', '-6_dB_fan/id_04', '-6_dB_fan/id_06', 
            '0_dB_fan/id_00', '0_dB_fan/id_02',  '0_dB_fan/id_04',  '0_dB_fan/id_06', 
            '6_dB_fan/id_00', '6_dB_fan/id_02',  '6_dB_fan/id_04',  '6_dB_fan/id_06',
            '-6_dB_valve/id_00', '-6_dB_valve/id_02', '-6_dB_valve/id_04', '-6_dB_valve/id_06', 
            '0_dB_valve/id_00', '0_dB_valve/id_02',  '0_dB_valve/id_04',  '0_dB_valve/id_06', 
            '6_dB_valve/id_00', '6_dB_valve/id_02',  '6_dB_valve/id_04',  '6_dB_valve/id_06',
            '-6_dB_pump/id_00', '-6_dB_pump/id_02', '-6_dB_pump/id_04', '-6_dB_pump/id_06', 
            '0_dB_pump/id_00', '0_dB_pump/id_02',  '0_dB_pump/id_04',  '0_dB_pump/id_06', 
            '6_dB_pump/id_00', '6_dB_pump/id_02',  '6_dB_pump/id_04',  '6_dB_pump/id_06'
            ]
epochs_list = [5, 10, 15, 20, 25, 30, 35, 40]
'''

# Extracting methods to make the spectrogram or other graphic features from WAV files
def extract_MFCC(file_path):
    audio_data, sample_rate = lr.load(file_path)
    mfccs = lr.feature.mfcc(y = audio_data, sr = sample_rate, n_mfcc = 200)
    return mfccs

def extract_GFCC(file_path):
    audio_data, sample_rate = lr.load(file_path)
    gfccs = gfcc(audio_data, fs = sample_rate, nfilts = 250, num_ceps = 250)
    return gfccs

def extract_STFT(file_path):
    sample_rate, audio_data = wavfile.read(file_path)
    f, t, data_stft = stft(audio_data[:,2], fs = sample_rate, nperseg = 1024, noverlap = 512)
    return f, t, data_stft

def extract_mel_spectro(file_path):
    audio_data, sample_rate = lr.load(file_path)
    mel_spectrogram = lr.feature.melspectrogram(y=audio_data, sr= sample_rate, n_fft=1024, hop_length=512, n_mels=128, power=2)
    return mel_spectrogram

# Loop for all cases = number of feature_method * number of dataset_name
start_time = time.perf_counter()
count = 0

# Make a dataframe to save the results
col_names = ["Method", "Dataset", "ROC_curve_AUC_test", "PR_curve_AP_test", "PR_curve_Chance_level",
           "TP_train", "FP_train","FN_train","TN_train", "TP_test", "FP_test", "FN_test","TN_test",
           "Loss_test", "Accuracy_test", "Loss_train", "Accuracy_train", "epochs"]
df_saved_results = pd.DataFrame(columns = col_names)

# Main part for learning and saving the results

for epoch_n in epochs_list:
    for feature_ in feature_method:
        for dataset_ in dataset_name:
            keras.utils.set_random_seed(40)
            
            print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
            print(f'▶▶ Starting [{feature_}] method and [{dataset_}] dataset.')
            
            normal_dir = f'./data/{dataset_}/normal'
            abnormal_dir = f'./data/{dataset_}/abnormal'
            
            # Get the list of the wav file names in the directory
            wav_files_no = os.listdir(normal_dir)
            wav_files_ab = os.listdir(abnormal_dir)
            
            if feature_ == 'MFCC':
                scaler = StandardScaler()
                data_Z = []
                for file in wav_files_no:
                    data_mfcc = extract_MFCC(os.path.join(normal_dir, file))
                    data_mfcc2 = scaler.fit_transform(data_mfcc)
                    data_Z.append(data_mfcc2)
                for file in wav_files_ab:
                    data_mfcc = extract_MFCC(os.path.join(abnormal_dir, file))
                    data_mfcc2 = scaler.fit_transform(data_mfcc)
                    data_Z.append(data_mfcc2)
                    
            elif feature_ == 'GFCC':
                data_Z = []
                for file in wav_files_no:
                    data_gfcc = extract_GFCC(os.path.join(normal_dir, file))
                    data_Z.append(data_gfcc)
                for file in wav_files_ab:
                    data_gfcc = extract_GFCC(os.path.join(abnormal_dir, file))
                    data_Z.append(data_gfcc)
                    
            elif feature_ == 'STFT':
                data_Z = []
                for file in wav_files_no:
                    data_f,data_t,data_stft = extract_STFT(os.path.join(normal_dir, file))
                    data_Z.append(data_stft)
                for file in wav_files_ab:
                    data_f,data_t,data_stft = extract_STFT(os.path.join(abnormal_dir, file))
                    data_Z.append(data_stft)
                    
            elif feature_ == 'Mel_spectrogram':
                data_Z = []
                for file in wav_files_no:
                    data_mel = extract_mel_spectro(os.path.join(normal_dir, file))
                    data_Z.append(data_mel)
                for file in wav_files_ab:
                    data_mel = extract_mel_spectro(os.path.join(abnormal_dir, file))
                    data_Z.append(data_mel)
            else:
                print('Wrong feature extraction method: check the option.')
                break
            
            # Convert the list to a numpy array
            data_Z = np.array(data_Z)
            
            # Create labels for the dataset: 0 : normal, 1 : abnormal
            label_no = np.zeros(len(wav_files_no))
            label_ab = np.ones(len(wav_files_ab))
            
            # Make the same number of samples for normal and abnormal (both in the train and test dataset)
            # This code will work for only abnormal cases are smaller then normal cases
            selected_normal = data_Z[0:len(label_no)][np.random.choice(len(label_no), len(label_ab), replace = False)]
            abnormal = data_Z[len(label_no):len(label_no)+len(label_ab)]
            data_Z = np.concatenate(((selected_normal,abnormal)))
            label_no = np.zeros(len(label_ab))
            label_ab = np.ones(len(label_ab))
            
            X_train_no, X_test_no, y_train_no, y_test_no = train_test_split(selected_normal, label_no, test_size = 0.5, random_state=42)
            X_train_ab, X_test_ab, y_train_ab, y_test_ab = train_test_split(abnormal, label_ab,  test_size = 0.5, random_state=42)
            X_train = np.concatenate((X_train_no, X_train_ab))
            X_test = np.concatenate((X_test_no, X_test_ab))
            y_train = np.concatenate((y_train_no, y_train_ab))
            y_test = np.concatenate((y_test_no, y_test_ab))
            
            # Define the CNN model layers: we can make any combination by adding or deleting the number of layers.
            
            model = Sequential()
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(data_Z.shape[1], data_Z.shape[2], 1)))
            model.add(MaxPooling2D(pool_size=(3, 3)))
            model.add(Flatten())
            model.add(Dense(64, activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            
            # Compile the model
            model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
            
            # Reshape the data to add a channel dimension (1 for grayscale)
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
            X_all = data_Z.reshape(data_Z.shape[0], data_Z.shape[1], data_Z.shape[2],1)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            
            # Train the model
            model.fit(X_train, y_train, epochs=epoch_n, batch_size=16)
            
            # Evaluate the model
            y_train_pred = model.predict(X_train)
            y_pred = model.predict(X_test)
            train_loss, train_accuracy = model.evaluate(X_train, y_train)
            test_loss, test_accuracy = model.evaluate(X_test, y_test)
            print(f"\n▶ [Train dataset] Overall accuracy test: Train Loss: {train_loss}, Train Accuracy: {train_accuracy:3.5f}")
            print(f"\n▶ [Test dataset] Overall accuracy test: Test Loss: {test_loss}, Test Accuracy: {test_accuracy:3.5f}")
            
            # Set the threshold to classify the TP, FP, TN, FN.
            threshold = 0.5
            
            # Results for train dataset
            train_m1 = keras.metrics.TruePositives(thresholds=threshold, name=None, dtype=None)
            train_m1.update_state(y_train, y_train_pred)
            train_m2 = keras.metrics.FalsePositives(thresholds=threshold, name=None, dtype=None)
            train_m2.update_state(y_train, y_train_pred)
            train_m3 = keras.metrics.FalseNegatives(thresholds=threshold, name=None, dtype=None)
            train_m3.update_state(y_train, y_train_pred)
            train_m4 = keras.metrics.TrueNegatives(thresholds=threshold, name=None, dtype=None)
            train_m4.update_state(y_train, y_train_pred)
            
            # Results for test dataset
            test_m1 = keras.metrics.TruePositives(thresholds=threshold, name=None, dtype=None)
            test_m1.update_state(y_test, y_pred)
            test_m2 = keras.metrics.FalsePositives(thresholds=threshold, name=None, dtype=None)
            test_m2.update_state(y_test, y_pred)
            test_m3 = keras.metrics.FalseNegatives(thresholds=threshold, name=None, dtype=None)
            test_m3.update_state(y_test, y_pred)
            test_m4 = keras.metrics.TrueNegatives(thresholds=threshold, name=None, dtype=None)
            test_m4.update_state(y_test, y_pred)
    
            # Set the figure saving folder
            Fig_save_dir = f'./output_Fig/{epoch_n}'
            dataset_2 = dataset_.replace('/', '_')
            
            # Plot ROC curve
            base = [0 for _ in range(len(y_test))]
            base_auc = roc_auc_score(y_test, base)
            pred_auc = roc_auc_score(y_test, y_pred)
            print('Model predicted: ROC AUC=%.3f' % (pred_auc))
            
            base_fpr, base_tpr, _ = roc_curve(y_test, base)
            pred_fpr, pred_tpr, _ = roc_curve(y_test, y_pred)
            plt.figure(figsize=(5,5))
            plt.plot(base_fpr, base_tpr, linestyle='--', label='Base', color = 'b')
            plt.plot(pred_fpr, pred_tpr, label='Model predicted (test dataset)', marker = "o", markersize = 5, color = 'g', alpha = 0.5)
            plt.xlabel('False Positive Rate', fontsize = 13)
            plt.ylabel('True Positive Rate', fontsize = 13)
            plt.text(0.25, 0.26, f'▶ Feature method: {feature_}', fontsize = 12)
            plt.text(0.25, 0.20, f'▶ Dataset name: {dataset_}', fontsize = 12)
            plt.text(0.25, 0.14, f'▶ AUC: {pred_auc: 3.3f}', fontsize = 12)
            plt.legend()
            plt.savefig(f'{Fig_save_dir}/{feature_}_{dataset_2}_ROC_curve.jpg', dpi = 180)
            plt.show()
            
            # Plot PR curve
            display = PrecisionRecallDisplay.from_predictions(
                y_test, y_pred, name="Model predicted", plot_chance_level=True
            )
            plt.savefig(f'{Fig_save_dir}/{feature_}_{dataset_2}_PR_curve.jpg', dpi = 180)
            plt.show()
            ap = average_precision_score(y_test, y_pred)
            chance_level = sum(y_test) / len(y_test)
            
            
            # Save the all above results into a dictioinary for the post-review and analysis process
            
            new_data = {"Method":feature_, "Dataset":dataset_, 
                       "ROC_curve_AUC_test":pred_auc, "PR_curve_AP_test" : ap, "PR_curve_Chance_level":chance_level,
                       "TP_train":train_m1.result().numpy(), "FP_train":train_m2.result().numpy(),
                       "FN_train":train_m3.result().numpy(),"TN_train":train_m4.result().numpy(),
                       "TP_test":test_m1.result().numpy(), "FP_test":test_m2.result().numpy(),
                       "FN_test":test_m3.result().numpy(),"TN_test":test_m4.result().numpy(),
                       "Loss_test":test_loss, "Accuracy_test":test_accuracy,
                       "Loss_train":train_loss, "Accuracy_train":train_accuracy, "epochs": epoch_n
                       }
            df_saved_results = df_saved_results.append(new_data, ignore_index=True)
            print("\n▷ The following data has been added in the dataframe [df_saved_results]\n")
            print(new_data)
    
            # Plot the results: train dataset
            threshold_line = threshold*np.ones(len(y_train))
            s = y_train.argsort()
            y_true_2 = y_train[s]
            y_pred_2 = y_train_pred[s]
            y_compare = np.hstack((y_true_2.reshape(len(y_train),1), y_pred_2))
            
            plt.figure(figsize=(8,4))
            plt.title("")
            plt.plot(y_compare[:,1], "o", alpha = 0.5, color = 'r', markersize = 5, label = 'Predicted values')
            plt.plot(y_compare[:,0], ".", alpha = 0.5, color = 'b', markersize = 5, label = 'Target values (1 = abnormal, 0 = normal)')
            plt.plot(threshold_line, ":", color = 'black', label = f'Threshold line = {threshold}')
            plt.xlabel('Train dataset samples', fontsize = 13)
            plt.ylabel('Predicted value [0,1]', fontsize = 13)
            plt.ylim(0,1)
            plt.legend(loc = (0.02, 0.1), fontsize = 13)
            plt.text(0.02, 0.55, f'▶ Feature method: {feature_}, Dataset name: {dataset_}', fontsize = 12)
            plt.savefig(f'{Fig_save_dir}/{feature_}_{dataset_2}_train_result.jpg', dpi = 180)
            plt.show()
            
            # Plot the results: test dataset
            s = y_test.argsort()
            y_true_2 = y_test[s]
            y_pred_2 = y_pred[s]
            y_compare = np.hstack((y_true_2.reshape(len(y_test),1), y_pred_2))
            
            plt.figure(figsize=(8,4))
            plt.title("")
            plt.plot(y_compare[:,1], "o", alpha = 0.5, color = 'r', markersize = 5, label = 'Predicted values')
            plt.plot(y_compare[:,0], ".", alpha = 0.5, color = 'b', markersize = 5, label = 'Target values (1 = abnormal, 0 = normal)')
            #plt.plot(threshold_line, ":", color = 'black', label = f'Threshold line = {threshold}')
            plt.xlabel('Test dataset samples', fontsize = 13)
            plt.ylabel('Predicted value [0,1]', fontsize = 13)
            plt.ylim(0,1)
            plt.legend(loc = (0.02, 0.1), fontsize = 13)
            plt.text(0.02, 0.55, f'▶ Feature method: {feature_}, Dataset name: {dataset_}', fontsize = 12)
            plt.savefig(f'{Fig_save_dir}/{feature_}_{dataset_2}_test_result.jpg', dpi = 180)
            plt.show()
            
            count += 1
            total = len(feature_method)*len(dataset_name)*len(epochs_list)
            print(f'\n▷ {count} of {total} dataset is completed. Total elapsed time: %.2f sec' % (time.perf_counter()-start_time) )
