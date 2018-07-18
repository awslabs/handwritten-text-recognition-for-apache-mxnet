import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import nd, autograd, gluon
import string

from handwriting_line_recognition import Network as BiLSTMNetwork
from handwriting_line_recognition import transform
from utils.noisy_forms_dataset import Noisy_forms_dataset
from utils.ngram_dataset import Ngram_dataset

alphabet_encoding = string.ascii_letters+string.digits+string.punctuation+' '
alphabet_dict = {alphabet_encoding[i]:i + 1 for i in range(len(alphabet_encoding))}

max_seq_len = 64
print_every_n = 5

class Network(gluon.Block):
    def __init__(self, **kwargs):
        super(Network, self).__init__(**kwargs)
        alphabet_size = len(string.ascii_letters+string.digits+string.punctuation+' ') + 2

        self.p_dropout = 0.5
        self.nn = nn.gluon.Sequential()
        with self.nn.name_scope():
            self.nn.add(mx.gluon.nn.Embedding(alphabet_size, 32))
            self.nn.add(mx.gluon.rnn.LSTM(100, 2, bidirectional=True))
            self.nn.add(mx.gluon.nn.Dense(64, flatten=False))
            self.nn.add(mx.gluon.nn.Dropout(self.p_dropout))
        # self.lstm_post = mx.gluon.rnn.LSTM(200, 1)
        # self.dense_post = mx.gluon.nn.Dense(64, flatten=False)

        # self.lstm_noisy = mx.gluon.rnn.LSTM(200, 1, bidirectional=False)
        # self.dense_noisy = mx.gluon.nn.Dense(64, flatten=False)
        self.decoder = mx.gluon.nn.Dense(units=alphabet_size, flatten=False)

    def forward(self, pre_label, post_label, noisy_label):
        # pre = self.embedding(pre_label)
        # pre = self.lstm(pre)
        # pre = self.dense(pre)
        # pre = gluon.nn.Dropout(self.dropout)(pre)
        
        # post = self.embedding(post_label)
        # post = self.lstm(post)
        # post = self.dense(post)
        # post = gluon.nn.Dropout(self.dropout)(post)

        noisy = self.nn(noisy)

        # out = self.decoder(noisy)

        # hs = nd.concat(*[pre, post, noisy], dim=2)
        out = self.decoder(noisy)
        return out
        
def get_ns(train):
    ctx = mx.gpu(0)
    network = BiLSTMNetwork()
    # params = mx.ndarray.load("model_checkpoint/handwriting100.params")
    # print(params.keys())
    network.load_params("model_checkpoint/handwriting100.params", ctx=ctx)

    def noise_source_transform(image, text):
        image, _ = transform(image, text)
        image = nd.array(image)
        image = image.as_in_context(ctx)
        image = image.expand_dims(axis=0)
        output = network(image)
        predict_probs = output.softmax().asnumpy()
        return predict_probs
    ns = Noisy_forms_dataset(noise_source_transform, train=train, name="OCR_noise")
    return ns

def encode(arr, is_string):
    string_encoded = []
    if is_string:
        for letter in arr:
            string_encoded.append(alphabet_dict[letter])
    else:
        for word in arr:
            if word == 0:
                empty_character = 0
                string_encoded.append(empty_character)
            else:
                for letter in word:
                    string_encoded.append(alphabet_dict[letter])
            string_encoded.append(alphabet_dict[" "])
    return string_encoded

def decode(prediction):
    results = []
    for i, index in enumerate(prediction):
        if index == 0: #index == len(alphabet_dict) or 
            continue
        else:
            results.append(alphabet_encoding[int(index)-1])
    word = ''.join([letter for letter in results])
    return word

def transform(pre_text, post_text, noisy_text, label):
    pre_text_encoded = np.zeros(max_seq_len, dtype=np.float32)
    post_text_encoded = np.zeros(max_seq_len, dtype=np.float32)
    noisy_text_encoded = np.zeros(max_seq_len, dtype=np.float32)
    label_encoded = np.zeros(max_seq_len, dtype=np.float32)

    pre_text_ = encode(pre_text, is_string=False)
    pre_text_encoded[-len(pre_text_):] = pre_text_
    post_text_ = encode(post_text, is_string=False)
    post_text_encoded[-len(post_text_):] = post_text_

    noisy_text_ = encode(noisy_text, is_string=True)
    noisy_text_encoded[-len(noisy_text_):] = noisy_text_

    label_ = encode(label, is_string=True)
    label_encoded[-len(label_):] = label_
    return pre_text_encoded, post_text_encoded, noisy_text_encoded, label_encoded

def run_epoch(e, network, dataloader, trainer, print_name, update_network, save_network, print_output):
    total_loss = nd.zeros(1, ctx)
    for i, (pre_text_encoded, post_text_encoded, noisy_text_encoded, y) in enumerate(dataloader):
        pre_text_encoded = pre_text_encoded.as_in_context(ctx)
        post_text_encoded = post_text_encoded.as_in_context(ctx)
        noisy_text_encoded = noisy_text_encoded.as_in_context(ctx)
        y = y.as_in_context(ctx)

        with autograd.record():
            output = network(pre_text_encoded, post_text_encoded, noisy_text_encoded)
            loss = loss_func(output, y)

        if update_network:
            loss.backward()
            trainer.step(y.shape[0])

        total_loss += loss.mean()

    epoch_loss = float(total_loss.asscalar())/len(dataloader)

    if print_output and e % print_every_n == 0 and e > 0:
        words_to_show = 5
        indexes = np.random.rand(words_to_show) * y.shape[0]

        for i in range(words_to_show):
            if i == 0:
                n = 9
            elif i == 1:
                n = 24
            else:
                n = int(indexes[i])
            
            out_np = output.asnumpy()[n, :]
            y_np = y.asnumpy()[n, :]
            noisy_np = noisy_text_encoded.asnumpy()[n, :]
            out_np_max = np.argmax(out_np, axis=1)

            out_decoded = decode(out_np_max)
            y_decoded = decode(y_np)
            noisy_decoded = decode(noisy_np)
            print("{} Actual: {}, predicted: {}, noisy: {}".format(n, y_decoded, out_decoded, noisy_decoded))
    # if save_network and e % save_every_n == 0 and e > 0:
    #     network.save_params("{}/{}".format(checkpoint_dir, checkpoint_name))

    return epoch_loss


if __name__ == "__main__":
    train_ns = get_ns(train=True)
    ng_train_ds = Ngram_dataset(train_ns, "3train")

    test_ns = get_ns(train=False)
    ng_test_ds = Ngram_dataset(test_ns, "3test")

    ctx = mx.gpu(0)
    batch_size = 32
    learning_rate = 0.001
    epochs = 500
    
    # pre_values, post_values, noisy_value, actual_value = ng_train_ds[0]
    # pre_values, post_values, noisy_value, actual_value = transform(pre_values, post_values, noisy_value, actual_value)
    print("Train ns size {}, test ns size {}".format(len(train_ns), len(test_ns)))
    
    train_data = gluon.data.DataLoader(ng_train_ds.transform(transform), batch_size, shuffle=True, last_batch="discard")
    test_data = gluon.data.DataLoader(ng_test_ds.transform(transform), batch_size, shuffle=False, last_batch="discard")

    loss_func = gluon.loss.SoftmaxCrossEntropyLoss()
    net = Network()
    net.hybridize()
    net.collect_params().initialize(mx.init.Normal(), ctx=ctx)

    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate})

    for e in range(epochs):
        train_loss = run_epoch(e, net, train_data, trainer, print_name="train", 
                               update_network=True, save_network=True, print_output=False)
        test_loss = run_epoch(e, net, test_data, trainer, print_name="test", 
                              update_network=False, save_network=False, print_output=True)
        if e % print_every_n == 0 and e > 0:
            print("Epoch {0}, train_loss {1:.6f}, test_loss {2:.6f}".format(e, train_loss, test_loss))
