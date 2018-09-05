import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
import string
from mxboard import SummaryWriter

from handwriting_line_recognition import Network as BiLSTMNetwork
from handwriting_line_recognition import transform
from utils.noisy_forms_dataset import Noisy_forms_dataset
from utils.ngram_dataset import Ngram_dataset

from utils.seq2seq import Seq2seq

alphabet_encoding = string.ascii_letters+string.digits+string.punctuation+' '
alphabet_dict = {alphabet_encoding[i]:i + 1 for i in range(len(alphabet_encoding))}

max_seq_len = 96
out_seq_len = 96
print_n = 50
print_every_n = 5
print_text_every_n = 20

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
    x_encoded = np.zeros(max_seq_len, dtype=np.float32)
    y_encoded = np.zeros(out_seq_len, dtype=np.float32)

    # Add a space to sepate pre, noisy and post text
    if pre_text[0] != 0:
        pre_text[0] = "<" + pre_text[0] + " "
    if post_text[0] != 0:
        post_text[0] = " " + post_text[0] + ">"
    
    pre_text_ = encode(pre_text, is_string=False)
    post_text_ = encode(post_text, is_string=False)
    noisy_text_ = encode(noisy_text, is_string=True)
    x = pre_text_ + noisy_text_ + post_text_
    if len(x) > max_seq_len:
        x_encoded = x[:max_seq_len]
    else:
        x_encoded[-len(x):] = x
    
    label_ = encode(label, is_string=True)
    y_encoded[:len(label_)] = label_

    x_test = decode(x_encoded)
    y_test = decode(y_encoded)
    return x_encoded, y_encoded

def run_epoch(e, network, dataloader, trainer, print_name, update_network, save_network, print_output):
    total_loss = nd.zeros(1, ctx)
    for i, (x, y) in enumerate(dataloader):
        x = x.as_in_context(ctx)
        y = y.as_in_context(ctx)

        with autograd.record():
            output = network(x, y)
            loss = loss_func(output, y)

        if update_network:
            loss.backward()
            trainer[0].step(y.shape[0])
            trainer[1].step(y.shape[0])

        total_loss += loss.mean()
        batch_loss += loss.mean()

        # if i % print_n == 0 and i > 0:
        #     mean_batch_loss = float(batch_loss.asscalar()/print_n)
        #     print('{} Batches {}: {:.6f}'.format(print_name, i, mean_batch_loss))
        #     batch_loss = nd.zeros(1, ctx)
        #     nd.waitall()

    epoch_loss = float(total_loss.asscalar())/len(dataloader)

    if print_output and e % print_text_every_n == 0 and e > 0:
        text = "predicted\t| actual\t| noisy \n ---- | ---- | ---- \n"
        for n in range(y.shape[0]):
            out_np = output.asnumpy()[n, :]
            y_np = y.asnumpy()[n, :]
            x_np = x.asnumpy()[n, :]
            out_np_max = np.argmax(out_np, axis=1)

            out_decoded = decode(out_np_max)
            y_decoded = decode(y_np)
            x_decoded = decode(x_np)

            output_text = out_decoded + "\t| " + y_decoded + "\t| " + x_decoded
            text += output_text + "\n"
        with SummaryWriter(logdir="./logs", verbose=False, flush_secs=5) as sw:
            sw.add_text(tag='{}_text'.format(print_name), text=text, global_step=e)
            print("output {}".format(text))

    # if save_network and e % save_every_n == 0 and e > 0:
    #     network.save_params("{}/{}".format(checkpoint_dir, checkpoint_name))
    with SummaryWriter(logdir="./logs", verbose=False, flush_secs=5) as sw:
        sw.add_scalar('loss', {print_name: epoch_loss}, global_step=e)

    return epoch_loss

if __name__ == "__main__":
    train_ns = get_ns(train=True)
    ng_train_ds = Ngram_dataset(train_ns, "2train", n=2)

    test_ns = get_ns(train=False)
    ng_test_ds = Ngram_dataset(test_ns, "2test", n=2)

    ctx = mx.gpu(0)
    batch_size = 32
    learning_rate = 0.00001
    epochs = 500
    
    # pre_values, post_values, noisy_value, actual_value = ng_train_ds[0]
    # pre_values, post_values, noisy_value, actual_value = transform(pre_values, post_values, noisy_value, actual_value)
    print("Train ns size {}, test ns size {}".format(len(ng_train_ds), len(ng_test_ds)))
    
    train_data = gluon.data.DataLoader(ng_train_ds.transform(transform), batch_size, shuffle=True, last_batch="discard")
    test_data = gluon.data.DataLoader(ng_test_ds.transform(transform), batch_size, shuffle=False, last_batch="discard")
    
    alphabet_size = len(string.ascii_letters+string.digits+string.punctuation+' ') + 2
    net = Seq2seq(input_size=alphabet_size, hidden_size=60, output_size=alphabet_size, sos_token=alphabet_dict["<"], eos_token=[">"], max_seq_len=max_seq_len)
    
    net.initialize(init=mx.init.Xavier(), ctx=ctx)
    net.hybridize()

    trainer_encoder = gluon.Trainer(net.encoder.collect_params(), 'adam', {"learning_rate": learning_rate})
    trainer_decoder = gluon.Trainer(net.decoder.collect_params(), 'adam', {"learning_rate": learning_rate})

    trainer = (trainer_encoder, trainer_decoder)
    
    loss_func = gluon.loss.SoftmaxCrossEntropyLoss()
    
    for e in range(epochs):
        train_loss = run_epoch(e, net, train_data, trainer, print_name="train", 
                               update_network=True, save_network=True, print_output=False)
        test_loss = run_epoch(e, net, test_data, trainer, print_name="test", 
                              update_network=False, save_network=False, print_output=True)
        if e % print_every_n == 0 and e > 0:
            print("Epoch {0}, train_loss {1:.6f}, test_loss {2:.6f}".format(e, train_loss, test_loss))
