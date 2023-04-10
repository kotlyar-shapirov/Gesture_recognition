import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMTaggerProb(nn.Module):

    def __init__(self, N_input_features, hidden_dim, dropout, bi, num_layers, # lstm
                 N_1d_filters, kernel_size, # 1d convolution
                 target_size, # last linear layer
                 ): 

        assert kernel_size%2 == 1, 'kernel_size must be odd number'
        super(LSTMTaggerProb, self).__init__()

        self.hidden_dim = hidden_dim
        self.N_input_features = N_input_features
        self.N_1d_filters = N_1d_filters

        # dropout
        self.dropout1= nn.Dropout(0.3)
        self.dropout2= nn.Dropout(0.3)
        print('dropout both')

        # conv1d layer
        self.conv = nn.Conv1d(in_channels=N_input_features,
                              out_channels=N_input_features * N_1d_filters,
                              kernel_size=kernel_size,
                              padding_mode='reflect',
                              padding=kernel_size//2,   # makes input and output size stay the same
                              groups=N_input_features)  # each channel has it's own filter

        self.conv2lstm = nn.Linear(N_input_features * N_1d_filters, hidden_dim)

        # LSTM definition
        self.lstm = nn.LSTM(hidden_dim ,
                            hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=bi)

        # Final linear layer
        if bi:
            self.N_lstm_layers = 2
        else:
            self.N_lstm_layers = 1
        lin_inpt_size = hidden_dim * self.N_lstm_layers
        self.hidden2hidden = nn.Linear(lin_inpt_size, lin_inpt_size)
        self.hidden2tag = nn.Linear(lin_inpt_size, target_size)
        



    def forward(self, gesture_sequence):
        '''
        :param gesture_sequence: shape [N (batch size), L (seq length), C (number of features)]
        :return:
        '''
        orig_gestrue_shapes = gesture_sequence.shape
        # print('Original gesture_sequence', gesture_sequence.shape)

        # dropout for joints
        gesture_sequence = self.dropout1(gesture_sequence)

        # convolving
        # for convolution we need input shape - (N, C, L)
        gesture_sequence = self.conv(torch.transpose(gesture_sequence, 1, 2))
        # after convolution we go back to (N, L, C)
        gesture_sequence = torch.transpose(gesture_sequence, 1, 2)

        # dimentionality reduction
        gesture_sequence = self.conv2lstm(gesture_sequence)

        # LSTM FORWARD
        # transpose for LSTM to (L, N, С)
        reshaped_tensor = torch.transpose(gesture_sequence, 0, 1)
        # print('reshaped_tensor', reshaped_tensor.shape)

        lstm_out, (h_n, c_n) = self.lstm(reshaped_tensor)  # sequence len, batch, embed_size
        # LINEAR FORWARD - reshape
        # using hidden states from all LSTM layers - transpose back to (N, L, С)
        linear_input = lstm_out.transpose(0,1)

        # LINEAR FORWARD - forward
        # linear_input1 = nn.ReLU()(self.hidden2hidden(linear_input)) + linear_input
        tag_space = self.hidden2tag(self.dropout2(linear_input))

        # scoring
        # CROSS ENTROPY ALWAYS WITHOUT SOFTMAX !!!
        tag_scores = tag_space

        return tag_scores
