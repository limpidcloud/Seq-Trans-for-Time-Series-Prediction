import torch


def calculate_seq_num(seq_len, total_len, cha):
    seq_num = []
    for length in seq_len:
        seq_num.append(total_len // length * cha)
    return seq_num


def get_l_c(l_pred):
    if l_pred < 80:
        return l_pred
    else:
        A = l_pred // 80
        B = l_pred // 30
        l_c = 0
        for i in range(A + 1, B):
            if l_pred % i == 0:
                l_c = l_pred // i
                break
        return l_c


def get_config(name):
    epochs = 20
    n_head = 4
    n_layer = 2  # number of second-level layer
    d = 64  # embed dim
    out_channel = 1
    l_pred = None  # l_pred
    channel = 0  # m, number of attributes of sequence
    phase = [12, 31, 24, 4]
    if name == 'Taxi':
        l_pred = 288
        phase = [31, 7, 24, 12]
        channel = 1
    elif name == 'ETT':
        l_pred = 96
        channel = 7
    elif name == 'ECL':
        l_pred = 96
        channel = 10
    elif name == 'Weather':
        l_pred = 144
        phase = [12, 31, 24, 6]
        channel = 16
    l_c = get_l_c(l_pred)  # compression length
    assert l_c > 0 and channel > 0 and len(phase) > 0
    L = 4 * l_pred  # input length of sequence
    kernel = l_c // 10 if (l_c // 10) % 2 == 1 else l_c // 10 + 1
    seq_len = [L // 4, L // 2, L]  # l_(i)
    seq_num = calculate_seq_num(seq_len, L, channel)  # n_(i)
    d_conv = max(channel, 4)  # output channel of convolutional layers
    return epochs, n_head, n_layer, channel, out_channel, l_pred, L, l_c, d, kernel, d_conv, seq_len, seq_num, phase


data_name = 'ETT'
device = torch.device('cuda', 0)
batch_size = 32
epoch, head_num, layer_num, in_dim, out_dim, predict_len, sequence_len, aim_len, embed_dim, kernel_len, conv_num, seq_len_list, seq_num_list, phase_list = get_config(data_name)
tmp_list = []
for i in seq_num_list:
    tmp_list.append(i//in_dim)