import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMTagger(nn.Module):

    def __init__(self, N_input_features, hidden_dim, N_lstm_layers, dropout, # lstm
                 N_1d_filters, kernel_size, # 1d convolution
                 target_size, use_all_lstm_layers=True, # last linear layer
                 ): 


        super(LSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim
        self.N_input_features = N_input_features

        self.N_1d_filters = N_1d_filters
        self.N_lstm_layers = N_lstm_layers
        self.use_all_lstm_layers = use_all_lstm_layers

        # batchnorm 1d
        self.batchnorm = nn.BatchNorm1d(N_input_features*N_1d_filters) # Input (N,C) or (N,C,L)
        print('No batchnorm')

        # conv1d layer
        self.conv = nn.Conv1d(in_channels=N_input_features,
                              out_channels=N_input_features * N_1d_filters,
                              kernel_size=kernel_size,
                              padding_mode='reflect',
                              padding=kernel_size,
                              groups=N_input_features)  # each channel has it's own filter

        # LSTM definition
        self.lstm = nn.LSTM(N_input_features * N_1d_filters,
                            hidden_dim * N_1d_filters,
                            num_layers=N_lstm_layers,
                            dropout=dropout)

        # Final linear layer
        if use_all_lstm_layers:
            self.hidden2tag = nn.Linear(hidden_dim * N_1d_filters * N_lstm_layers, target_size)
        else:
            self.hidden2tag = nn.Linear(hidden_dim * N_1d_filters, target_size)


    def forward(self, gesture_sequence):
        '''
        :param gesture_sequence: shape [N (batch size), L (seq length), C (number of features)]
        :return:
        '''
        orig_gestrue_shapes = gesture_sequence.shape
        # print('Original gesture_sequence', gesture_sequence.shape)

        # convolving
        # for convolution we need input shape - (N, C, L)
        gesture_sequence = self.conv(torch.transpose(gesture_sequence, 1, 2))
        # gesture_sequence = self.batchnorm(gesture_sequence)
        # after convolution we go back to (N, L, C)
        gesture_sequence = torch.transpose(gesture_sequence, 1, 2)

        # LSTM FORWARD
        # transpose for LSTM to (L, N, С)
        reshaped_tensor = torch.transpose(gesture_sequence, 0, 1)
        # print('reshaped_tensor', reshaped_tensor.shape)

        lstm_out, (h_n, c_n) = self.lstm(reshaped_tensor)  # sequence len, batch, embed_size

        # LINEAR FORWARD - reshape
        # using hidden states from all LSTM layers
        if self.use_all_lstm_layers:
            liner_input = c_n.view(c_n.shape[1], c_n.shape[2] * c_n.shape[0])

        # using hidden states from only last LSTM layer
        else:
            liner_input = c_n[-1].view(c_n.shape[1], c_n.shape[2])

        # linear forward for point-wise
        # reshaping into batchsize, Number_of_words, hidden_embedding
        #         last_layer_input = lstm_out.view(-1, len(gesture_sequence),
        #                                          self.hidden_dim*self.N_filters)


        # LINEAR FORWARD - forward
        tag_space = self.hidden2tag(liner_input)

        # scoring
        # CROSS ENTROPY ALWAYS WITHOUT SOFTMAX !!!
        tag_scores = tag_space

        return tag_scores
