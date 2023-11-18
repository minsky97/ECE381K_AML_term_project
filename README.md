# Anomalous sound detection for industrial machine based on supervised CNN with various feature extraction methods (STFT, MFCC, Mel-spectrogram, GFCC)
[ECE381K_AML_term_project]

In this study, we develop a method to detect abnormal sounds from audio data recorded for various machine types, models, and SNRs. The dataset we used in this project comes from the MIMII dataset (https://zenodo.org/records/3384388). This includes normal and abnormal sound dataset for four type of industrial machines (i.e. valves, pumps, fans, and slide rails) with different model IDs (i.e. 00, 02, 04, and 06) and SNRs (i.e. -6 dB, 0 dB, and 6 dB).

The specific purposes of this study are:
- Proposing a new supervised anomaly detection method based on the CNN model as a classifier
- Evaluating the performance feature extraction methods (STFT, MFCCs, GFCCs, and Mel-spectrogram), which convert the audio to image data
- Assessment of the accuracy and performance of the proposed anomaly detection model
- Exploring the over-fitting issue by increasing epochs

