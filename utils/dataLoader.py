import numpy as np

class DataLoader():
    def __init__(self, data_file, batch_size, seq_len):
        self.data = np.load(data_file, encoding='latin1')
        self.batch_size = batch_size
        self.seq_len = seq_len

    def generate_batch(self):
        x_batch = []
        y_batch = []

        for i in range(self.batch_size):
            stroke = self.data[np.random.randint(0, self.data.size + 1)]

            start_idx = np.random.randint(0, len(stroke) - self.seq_len)
            x_batch.append(stroke[start_idx: start_idx+self.seq_len])
            y_batch.append(stroke[start_idx+1:start_idx+1+self.seq_len])

        return x_batch, y_batch
