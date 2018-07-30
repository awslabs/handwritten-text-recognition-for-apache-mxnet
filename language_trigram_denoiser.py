import multiprocessing
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
import string
from mxboard import SummaryWriter
import argparse

from handwriting_line_recognition import Network as BiLSTMNetwork
from handwriting_line_recognition import transform as handwriting_transform
from utils.noisy_forms_dataset import Noisy_forms_dataset
from utils.ngram_dataset import Ngram_dataset

unused_symbols = ["#", "$", "*", "-", "<", "=", ">", "@", "[", "\\",
                  "]", "^", "_", "`", "{", "|", "}", "~"]
capitals = ["A", "B", "C", "D", "E", "F", "G", "H",
            "I", "J", "K", "L", "M",  "N", "O", "P", "Q", "R", "S", "T",
            "U", "V", "W", "X", "Y", "Z"]

def remove_string(string, values_to_remove):
    output_string = string
    for sym in values_to_remove:
        output_string = output_string.replace(sym, "")
    return output_string

alphabet_encoding = string.ascii_letters+string.digits+string.punctuation+' '
alphabet_encoding = remove_string(alphabet_encoding, values_to_remove=unused_symbols)
alphabet_encoding = remove_string(alphabet_encoding, values_to_remove=capitals)

alphabet_dict = {alphabet_encoding[i]:i for i in range(len(alphabet_encoding))}
alphabet_size = len(alphabet_dict) + 1

print_every_n = 5
print_text_every_n = 50
save_every_n = 50

def get_ns(train):
    network = BiLSTMNetwork(ctx=ctx)
    # params = mx.ndarray.load("model_checkpoint/handwriting_line_good.params")
    # print(params.keys())
    network.load_params("model_checkpoint/handwriting_line200.params", ctx=ctx)

    def noise_source_transform(image, text):
        image, _ = handwriting_transform(image, text)
        image = nd.array(image)
        image = image.as_in_context(ctx)
        image = image.expand_dims(axis=0)
        output = network(image)
        predict_probs = output.softmax().asnumpy()
        return predict_probs
    ns = Noisy_forms_dataset(noise_source_transform, train=train, name="OCR_noise")
    return ns

def transform(pre_text, post_text, noisy_text, label):
    # replace capitals to lower case
    pre_text = pre_text.lower()
    post_text = post_text.lower()
    noisy_text = noisy_text.lower()
    label = label.lower()

    pre_text_encoded = [alphabet_dict[char] for char in pre_text]
    post_text_encoded = [alphabet_dict[char] for char in post_text]
    noisy_text_index = alphabet_dict[noisy_text]
    y_index = alphabet_dict[label]
    return pre_text_encoded, post_text_encoded, noisy_text_index, y_index

class Trigram_denoiser_network(gluon.Block):
    def __init__(self, **kwargs):
        super(Trigram_denoiser_network, self).__init__(**kwargs)
        with self.name_scope():
            hidden_states = 100
            lstm_layers = 1

            self.embedding = mx.gluon.nn.Embedding(alphabet_size, hidden_states)
            self.encoder_pre = mx.gluon.rnn.LSTM(hidden_states, lstm_layers)
            self.encoder_post = mx.gluon.rnn.LSTM(hidden_states, lstm_layers)
            self.decoder = mx.gluon.nn.Dense(units=alphabet_size, flatten=False)

    def predict_from_context(self, values, encoder):
        values = self.embedding(values)
        values = values.transpose((1, 0, 2))
        values = encoder(values)
        values = values.transpose((1, 0, 2))
        values = values.flatten()
        return values
    
    def forward(self, pre, post, noisy, y):
        pre = self.predict_from_context(pre, encoder=self.encoder_pre)
        post = self.predict_from_context(post, encoder=self.encoder_post)
        context = nd.concat(*[pre, post], dim=1)
        out = self.decoder(context)
        out = mx.nd.softmax(out, axis=1)
        return out

class Trigram_denoiser_network_attention(gluon.Block):
    def __init__(self, n, **kwargs):
        super(Trigram_denoiser_network_attention, self).__init__(**kwargs)
        with self.name_scope():
            hidden_states = 100
            lstm_layers = 1

            self.embedding = mx.gluon.nn.Embedding(alphabet_size, hidden_states)
            self.encoder_pre = mx.gluon.rnn.LSTM(hidden_states, lstm_layers)
            self.encoder_post = mx.gluon.rnn.LSTM(hidden_states, lstm_layers)
            self.decoder = mx.gluon.nn.Dense(units=alphabet_size, flatten=False)
            self.attn = mx.gluon.nn.Dense(n)
            self.attn_applied = mx.gluon.nn.Dense(hidden_states)

    def predict_from_context(self, values, encoder):
        values = self.embedding(values)
        values = values.transpose((1, 0, 2))
        values = encoder(values)
        values = values.transpose((1, 0, 2))
        return values

    def predict_attention_from_context(self, noisy_embedded, context_hs, context):
        weights = mx.nd.concat(*[noisy_embedded, context_hs], dim=1)
        attn_weights = self.attn(weights)
        attn_weights = mx.nd.softmax(attn_weights, axis=1)
        attn_applied = mx.nd.linalg_gemm2(attn_weights.expand_dims(1), context)
        return attn_applied
    
    def forward(self, pre, post, noisy, y):
        noisy_embedded = self.embedding(noisy)
        pre = self.predict_from_context(pre, encoder=self.encoder_pre)
        post = self.predict_from_context(post, encoder=self.encoder_post)

        pre_hs = pre[:, -1, :]
        post_hs = post[:, -1, :]

        attn_pre = self.predict_attention_from_context(noisy_embedded, pre_hs, pre)
        attn_post = self.predict_attention_from_context(noisy_embedded, post_hs, post)

        context = nd.concat(*[attn_pre[:, 0, :], attn_post[:, 0, :]], dim=1)
        context = self.attn_applied(context)
        context = mx.nd.relu(context)
        
        out = self.decoder(context)
        out = mx.nd.softmax(out, axis=1)
        return out
        
def run_epoch(e, network, dataloader, trainer, print_name, update_network, save_network, print_output):
    total_loss = nd.zeros(1, ctx)
    for i, (pre, post, x, y) in enumerate(dataloader):
        pre = pre.as_in_context(ctx)
        post = post.as_in_context(ctx)        
        x = x.as_in_context(ctx)
        y = y.as_in_context(ctx)

        with autograd.record():
            output = network(pre, post, x, y)
            loss = loss_func(output, y)

        if update_network:
            loss.backward()
            trainer.step(y.shape[0])

        total_loss += loss.mean()

    epoch_loss = float(total_loss.asscalar())/len(dataloader)

    if print_output and e % print_text_every_n == 0 and e > 0:
        text = ""
        for n in range(y.shape[0]):
            out_np = output.asnumpy()[n, :]
            y_np = y.asnumpy()[n]
            x_np = x.asnumpy()[n]
            out_np_max = np.argmax(out_np)
            pre_np = pre.asnumpy()[n]
            post_np = post.asnumpy()[n]

            alphabet_encoding_with_eos = alphabet_encoding + " "
            
            out_decoded = alphabet_encoding_with_eos[out_np_max]
            y_decoded = alphabet_encoding_with_eos[y_np]
            x_decoded = alphabet_encoding_with_eos[x_np]            
            pre_decoded = [alphabet_encoding_with_eos[a] for a in pre_np]
            post_decoded = [alphabet_encoding_with_eos[a] for a in post_np]

            pre_decoded = "".join(pre_decoded)
            post_decoded = "".join(post_decoded)

            output_text = pre_decoded + "'" + y_decoded + "'" + post_decoded + "| predicted: {} - noisy: {}".format(out_decoded, x_decoded)
            text += output_text + "\n"
        print(text)
        
    with SummaryWriter(logdir="./logs", verbose=False, flush_secs=5) as sw:
        sw.add_scalar('loss', {print_name: epoch_loss}, global_step=e)

    if save_network and e % save_every_n == 0 and e > 0:
        network.save_parameters("{}/{}".format(checkpoint_dir, checkpoint_name))
    return epoch_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu_id", default=0,
                        help="ID of the GPU to use")
    parser.add_argument("-m", "--model", default="att", choices=["att", "dense"],
                        help="type of model to use")
    parser.add_argument("-c", "--checkpoint_dir", default="model_checkpoint",
                        help="Directory to store the checkpoints")
    parser.add_argument("-n", "--checkpoint_name", default="language_model.params",
                        help="Name to store the checkpoints")

    args = parser.parse_args()
    n = 5
    
    gpu_id = int(args.gpu_id)
    ctx = mx.gpu(gpu_id)

    model_type = args.model
    checkpoint_dir, checkpoint_name = args.checkpoint_dir, args.checkpoint_name

    train_ns = get_ns(train=True)
    ng_train_ds = Ngram_dataset(train_ns, "char_5train", output_type="character", n=n)

    test_ns = get_ns(train=False)
    ng_test_ds = Ngram_dataset(test_ns, "char_5test", output_type="character", n=n)

    batch_size = 32
    learning_rate = 0.0005
    epochs = 3000

    print("Train ns size {}, test ns size {}".format(len(ng_train_ds), len(ng_test_ds)))
    
    train_data = gluon.data.DataLoader(ng_train_ds.transform(transform), batch_size, shuffle=True, last_batch="discard",
                                       num_workers=int(multiprocessing.cpu_count()/2))
    test_data = gluon.data.DataLoader(ng_test_ds.transform(transform), batch_size, shuffle=False, last_batch="discard",
                                      num_workers=int(multiprocessing.cpu_count()/2))

    if model_type == "dense":
        net = Trigram_denoiser_network()
    else:
        net = Trigram_denoiser_network_attention(n=n)
    net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    net.hybridize()

    trainer = gluon.Trainer(net.collect_params(), 'adam', {"learning_rate": learning_rate})
    loss_func = gluon.loss.SoftmaxCrossEntropyLoss()
    
    for e in range(epochs):
        train_loss = run_epoch(e, net, train_data, trainer, print_name="train", 
                               update_network=True, save_network=True, print_output=False)
        test_loss = run_epoch(e, net, test_data, trainer, print_name="test", 
                              update_network=False, save_network=False, print_output=True)
        if e % print_every_n == 0 and e > 0:
            print("Epoch {0}, train_loss {1:.6f}, test_loss {2:.6f}".format(e, train_loss, test_loss))
