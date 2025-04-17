import tensorflow as tf
from keras import layers, models
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

#Parameters
sequence_length = 700
num_classes = 3
input_dim = 20
hidden_dim = 64
batch_size = 16
num_epochs = 1
learning_rate = 0.001

#Amino acid to index mapping
aa_to_index = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
}

#One-hot Encoding the amino acid sequence
def one_hot_encode_sequence(seq):
    one_hot = np.zeros((sequence_length, len(aa_to_index)))
    for i, aa in enumerate(seq):
        if i < sequence_length and aa in aa_to_index:
            one_hot[i, aa_to_index[aa]] = 1.0
    return one_hot

#Secondary structure label mapping
ss_to_label = {'H': 0, 'E': 1, 'C': 2}

def encode_labels(labels):
    encoded = np.zeros(sequence_length, dtype=np.int32)
    for i, ss in enumerate(labels):
        if i < sequence_length and ss in ss_to_label:
            encoded[i] = ss_to_label[ss]
    return encoded

#Loadign the datasets
train_df = pd.read_csv("training_secondary_structure_train.csv")
valid_df = pd.read_csv("validation_secondary_structure_valid.csv")
test_df = pd.read_csv("test_secondary_structure_cb513.csv")

train_sequences = np.array([one_hot_encode_sequence(seq) for seq in train_df['seq']])
train_labels = np.array([encode_labels(sst3) for sst3 in train_df['sst3']])
valid_sequences = np.array([one_hot_encode_sequence(seq) for seq in valid_df['seq']])
valid_labels = np.array([encode_labels(sst3) for sst3 in valid_df['sst3']])
test_sequences = np.array([one_hot_encode_sequence(seq) for seq in test_df['seq']])
test_labels = np.array([encode_labels(sst3) for sst3 in test_df['sst3']])

#Model definition
model = models.Sequential()
model.add(layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(sequence_length, input_dim)))
model.add(layers.Bidirectional(layers.LSTM(hidden_dim, return_sequences=True)))
model.add(layers.TimeDistributed(layers.Dense(num_classes, activation='softmax')))


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.summary()

#Training the model
history = model.fit(train_sequences, train_labels, validation_data=(valid_sequences, valid_labels), epochs=num_epochs, batch_size=batch_size)

#Evaluating on test data
test_loss, test_accuracy = model.evaluate(test_sequences, test_labels)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

#Generating predictions on test data
predictions = model.predict(test_sequences)
predicted_labels = np.argmax(predictions, axis=-1)  # Convert one-hot to class indices

#Flattening the predicted and true labels for metric calculation
flat_predicted_labels = predicted_labels.flatten()
flat_true_labels = test_labels.flatten()

#Calculating the Precision, Recall, and F1 Score
precision = precision_score(flat_true_labels, flat_predicted_labels, average='macro')
recall = recall_score(flat_true_labels, flat_predicted_labels, average='macro')
f1 = f1_score(flat_true_labels, flat_predicted_labels, average='macro')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

#Segment Overlap (SOV) measure calculation
def calculate_sov(true_labels, pred_labels, label):
    segments_true, segments_pred = [], []
    start, current = -1, -1

    for i in range(len(true_labels)):
        if true_labels[i] == label:
            if current == -1:
                start = i
            current = i
        else:
            if current != -1:
                segments_true.append((start, current))
                start, current = -1, -1

    if current != -1:
        segments_true.append((start, current))

    start, current = -1, -1
    for i in range(len(pred_labels)):
        if pred_labels[i] == label:
            if current == -1:
                start = i
            current = i
        else:
            if current != -1:
                segments_pred.append((start, current))
                start, current = -1, -1

    if current != -1:
        segments_pred.append((start, current))

    overlap_sum, min_sum, max_sum = 0, 0, 0

    for s_true in segments_true:
        max_overlap = 0
        for s_pred in segments_pred:
            overlap = min(s_true[1], s_pred[1]) - max(s_true[0], s_pred[0]) + 1
            if overlap > 0:
                max_overlap = max(max_overlap, overlap)
                min_len = min(s_true[1] - s_true[0] + 1, s_pred[1] - s_pred[0] + 1)
                max_len = max(s_true[1] - s_true[0] + 1, s_pred[1] - s_pred[0] + 1)
                min_sum += min_len
                max_sum += max_len
                overlap_sum += overlap

    return (overlap_sum + min_sum - max_sum) / max_sum if max_sum > 0 else 0

#Calculating SOV for each secondary structure type
sov_H = calculate_sov(flat_true_labels, flat_predicted_labels, ss_to_label['H'])
sov_E = calculate_sov(flat_true_labels, flat_predicted_labels, ss_to_label['E'])
sov_C = calculate_sov(flat_true_labels, flat_predicted_labels, ss_to_label['C'])
sov_avg = (sov_H + sov_E + sov_C) / 3

print(f"SOV (H): {sov_H:.4f}")
print(f"SOV (E): {sov_E:.4f}")
print(f"SOV (C): {sov_C:.4f}")
print(f"SOV (Average): {sov_avg:.4f}")

#Function to predict the secondary structure for a single sequence
def predict_secondary_structure(sequence):
    one_hot_sequence = one_hot_encode_sequence(sequence)
    one_hot_sequence = np.expand_dims(one_hot_sequence, axis=0)  # Add batch dimension

    predictions = model.predict(one_hot_sequence)
    predicted_labels = np.argmax(predictions, axis=-1).flatten()

    label_to_ss = {v: k for k, v in ss_to_label.items()}  # Reverse mapping
    predicted_ss = ''.join([label_to_ss[label] for label in predicted_labels])

    return predicted_ss

#Asking the user for a sequence to test and predict the Secondary Structures of Protein
sequence = input("Enter a protein sequence (up to 700 amino acids): ").upper()
sequence = sequence[:sequence_length]  # Trim sequence if it's longer than 700
predicted_structure = predict_secondary_structure(sequence)

print("Predicted Secondary Structure:")
print(predicted_structure)