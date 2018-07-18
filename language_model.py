import mxnet as mx
from mxnet import nd
import numpy as np

from autocorrect import spell as simple_spellcheck

from handwriting_line_recognition import Network as BiLSTMNetwork
from handwriting_line_recognition import transform
from handwriting_line_recognition import decode as topK_decode
from handwriting_line_recognition import alphabet_dict

from utils.noisy_sentences_dataset import Noisy_sentences_dataset

from utils.max_flow import FlowNetwork

def get_ns(train):
    ctx = mx.gpu(0)
    network = BiLSTMNetwork()
    # params = mx.ndarray.load("model_checkpoint/temp.params")
    # print(params.keys())
    # network.load_params("model_checkpoint/temp.params", ctx=ctx)
    # import pdb; pdb.set_trace();    

    def noise_source_transform(image, text):
        image, _ = transform(image, text)
        image = nd.array(image)
        image = image.as_in_context(ctx)
        image = image.expand_dims(axis=0)
        output = network(image)
        predict_probs = output.softmax().asnumpy()
        return predict_probs
    ns = Noisy_sentences_dataset(noise_source_transform, train=train, name="OCR_noise")
    return ns

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def topk_decoded(prob):
    predicted_text = topK_decode(np.argmax(prob, axis=2))[0]
    return predicted_text

def simple_spellchecker(prob):
    predicted_text = topK_decode(np.argmax(prob, axis=2))[0]
    output = ""
    for word in predicted_text.split(" "):
        output += simple_spellcheck(word) + " "
    return output


if __name__ == "__main__":
    train_ns = get_ns(train=True)

    from hmmlearn.hmm import MultinomialHMM
    hmm = MultinomialHMM(n_components=3)
    
    for i in range(len(train_ns)):
        prob, text = train_ns[i]
        predicted_text = topK_decode(np.argmax(prob, axis=2))[0]
        import pdb; pdb.set_trace();        
    
    test_ns = get_ns(train=False)
    
    distances = []
    for i in range(len(test_ns)):
        if i % 100 == 0:
            print("{}/{}".format(i, len(test_ns)))
        prob, text = test_ns[i]
        topk_text = topk_decoded(prob)
        topk_distance = levenshtein_distance(topk_text, text)

        simple_spellchecker_text = simple_spellchecker(prob)
        ss_distance = levenshtein_distance(simple_spellchecker_text, text)

        # max_flow_text = max_flow(prob)
        # mf_distance = levenshtein_distance(max_flow_text, text)

        distances.append([topk_distance, ss_distance])#, mf_distance])

    distances = np.array(distances)
    mean_distance = np.mean(distances, axis=0)
    import pdb; pdb.set_trace();    
