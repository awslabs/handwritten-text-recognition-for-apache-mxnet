import os
import tarfile
import urllib
import sys
import time
import glob
import pickle
import xml.etree.ElementTree as ET
import cv2
import json
import numpy as np
import pandas as pd
import zipfile
import matplotlib.pyplot as plt

from mxnet.gluon.data import dataset
from mxnet import nd

class IAMDataset(dataset.ArrayDataset):
    """ The IAMDataset provides images of handwritten passages written by multiple
    individuals. The data is available at http://www.fki.inf.unibe.ch

    The passages can be parsed into separate words, lines, or the whole form.
    The dataset should be separated into writer independent training and testing sets.

    Parameters
    ----------
    parse_method: str, Required
        To select the method of parsing the images of the passage
        Available options: [form, line, word]

    credentials: (str, str), Default None 
        Your (username, password) for the IAM dataset. Register at
        http://www.fki.inf.unibe.ch/DBs/iamDB/iLogin/index.php
        By default, IAMDataset will read it from credentials.json
    
    root: str, default: dataset/iamdataset
        Location to save the database

    train: bool, default True
        Whether to load the training or testing set of writers.

    output_data_type: str, default text
        What type of data you want as an output: Text or bounding box.
        Available options are: [text, bb]
     
    output_parse_method: str, default None
        If the bounding box (bb) was selected as an output_data_type, 
        this parameter can select which bb you want to obtain.
        Available options: [form, line, word]

    transform: function, default None
        A user defined callback that transforms each sample. For example:
    ::
        transform=lambda data, label: (data.astype(np.float32)/255, label)

    """
    MAX_IMAGE_SIZE_FORM = (1120, 800)
    def __init__(self, parse_method, credentials=None,
                 root=os.path.join('dataset', 'iamdataset'), 
                 train=True, output_data="text",
                 output_parse_method=None, transform=None):

        _parse_methods = ["form", "line", "word"]
        error_message = "{} is not a possible parsing method: {}".format(
            parse_method, _parse_methods)
        assert parse_method in _parse_methods, error_message
        self._parse_method = parse_method
        url_partial = "http://www.fki.inf.unibe.ch/DBs/iamDB/data/{data_type}/{filename}.tgz"
        if self._parse_method == "form":
            self._data_urls = [url_partial.format(data_type="forms", filename="forms" + a) for a in ["A-D", "E-H", "I-Z"]]
        elif self._parse_method == "line":
            self._data_urls = [url_partial.format(data_type="lines", filename="lines")]
        elif self._parse_method == "word":
            self._data_urls = [url_partial.format(data_type="words", filename="words")]
        self._xml_url = "http://www.fki.inf.unibe.ch/DBs/iamDB/data/xml/xml.tgz"

        if credentials == None:
            if os.path.isfile("credentials.json"):
                with open('credentials.json') as f:
                    credentials = json.load(f)
                self._credentials = (credentials["username"], credentials["password"])
            else:
                assert False, "Please enter credentials for the IAM dataset in credentials.json or as arguments"
        else:
            self._credentials = credentials
        
        self._train = train

        _output_data_types = ["text", "bb"]
        error_message = "{} is not a possible output data: {}".format(
            output_data, _output_data_types)
        assert output_data in _output_data_types, error_message
        self._output_data = output_data

        if self._output_data == "bb":
            _parse_methods = ["form", "line", "word"]
            error_message = "{} is not a possible output parsing method: {}".format(
                output_parse_method, _parse_methods)
            assert output_parse_method in _parse_methods, error_message
            self._output_parse_method = output_parse_method

        self.image_data_file_name = os.path.join(root, "image_data-{}-{}.plk".format(self._parse_method, self._output_data))
        self._root = root
        if not os.path.isdir(root):
            os.makedirs(root)

        data = self._get_data()
        super(IAMDataset, self).__init__(data)

    def _download(self, url):
        toolbar_width = 40
        password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, url, self._credentials[0], self._credentials[1])
        handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
        opener = urllib.request.build_opener(handler)
        urllib.request.install_opener(opener)
        opener.open(url)
        filename = os.path.basename(url)

        print("Downloading {}: ".format(filename)) 
        def reporthook(count, block_size, total_size):
            percentage = float(count * block_size) / total_size * 100
            # Taken from https://gist.github.com/sibosutd/c1d9ef01d38630750a1d1fe05c367eb8
            sys.stdout.write('\r')
            sys.stdout.write("Completed: [{:{}}] {:>3}%"
                             .format('-' * int(percentage / (100.0 / toolbar_width)),
                                     toolbar_width, int(percentage)))
            sys.stdout.flush()

        urllib.request.urlretrieve(url,
                                   reporthook=reporthook,
                                   filename=os.path.join(self._root, filename))[0]
        sys.stdout.write("\n")

    def _download_xml(self):
        archive_file = os.path.join(self._root, os.path.basename(self._xml_url))
        if not os.path.isfile(archive_file):
            self._download(self._xml_url)
            tar = tarfile.open(archive_file, "r:gz")
            tar.extractall(os.path.join(self._root, "xml"))
            tar.close()

    def _download_data(self):
        for url in self._data_urls:
            archive_file = os.path.join(self._root, os.path.basename(url))
            if not os.path.isfile(archive_file):
                self._download(url)
                tar = tarfile.open(archive_file, "r:gz")
                tar.extractall(os.path.join(self._root, self._parse_method))
                tar.close()

    def _download_subject_list(self):
        url = "http://www.fki.inf.unibe.ch/DBs/iamDB/tasks/largeWriterIndependentTextLineRecognitionTask.zip"
        archive_file = os.path.join(self._root, os.path.basename(url))
        if not os.path.isfile(archive_file):
            self._download(url)
            zip_ref = zipfile.ZipFile(archive_file, 'r')
            zip_ref.extractall(os.path.join(self._root, "subject"))
            zip_ref.close()
        
    def _pre_process_image(self, img_in):    
        im = cv2.imread(img_in, cv2.IMREAD_GRAYSCALE)
        # reduce the size of form images so that it can fit in memory.
        if self._parse_method == "form":
            size = im.shape[:2] # old_size is in (height, width) format
            if size[0] > self.MAX_IMAGE_SIZE_FORM[0] or size[1] > self.MAX_IMAGE_SIZE_FORM[1]:
                ratio_w = float(self.MAX_IMAGE_SIZE_FORM[0])/size[0]
                ratio_h = float(self.MAX_IMAGE_SIZE_FORM[1])/size[1]
                ratio = min(ratio_w, ratio_h)
                new_size = tuple([int(x*ratio) for x in size])
                im = cv2.resize(im, (new_size[1], new_size[0]))
                size = im.shape
            
            delta_w = max(0, self.MAX_IMAGE_SIZE_FORM[1] - size[1])
            delta_h = max(0, self.MAX_IMAGE_SIZE_FORM[0] - size[0])
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)
            
            color = im[0][0]
            if color < 230:
                color = 230
            im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=float(color))

        img_arr = np.asarray(im)
        return img_arr

    def _get_bb_of_item(self, item):
        # Find the coordinates of the left and right-most letters for the
        # bounding box of the item
        character_list = [a for a in item.iter("cmp")]
        x1 = np.min([int(a.attrib['x']) for a in character_list])
        y1 = np.min([int(a.attrib['y']) for a in character_list])
        x2 = np.max([int(a.attrib['x']) + int(a.attrib['width']) for a in character_list])
        y2 = np.max([int(a.attrib['y']) + int(a.attrib['height'])for a in character_list])
        return [x1, y1, x2 - x1, y2 - y1]
    
    def _get_output_data(self, item):
        output_data = []
        if self._output_data == "text":
            if self._parse_method == "form":
                text = ""
                for line in item.iter('machine-print-line'):
                    text += line.attrib["text"] + "\n"
                output_data.append(text)
            else:
                output_data.append(item.attrib['text'])
        else:
            for item_output in item.iter(self._output_parse_method):
                bb = self._get_bb_of_item(item_output)
                output_data.append(bb)
        output_data = np.array(output_data)
        return output_data
            
    def _process_data(self):
        image_data = []
        xml_files = glob.glob(self._root + "/xml/*.xml")
        toolbar_width = 40
        print("Processing data...")
        def reporthook(count, total_size):
            percentage = float(count) / total_size * 100
            # Taken from https://gist.github.com/sibosutd/c1d9ef01d38630750a1d1fe05c367eb8
            sys.stdout.write('\r')
            sys.stdout.write("Completed: [{:{}}] {:>3}%"
                             .format('-' * int(percentage / (100.0 / toolbar_width)),
                                     toolbar_width, int(percentage)))
            sys.stdout.flush()

        for i, xml_file in enumerate(xml_files):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for item in root.iter(self._parse_method):
                if self._parse_method == "form":
                    image_id = item.attrib["id"]
                else:
                    tmp_id = item.attrib["id"]
                    tmp_id_split = tmp_id.split("-")
                    image_id = os.path.join(tmp_id_split[0], tmp_id_split[0] + "-" + tmp_id_split[1], tmp_id)
                image_filename = os.path.join(self._root, self._parse_method, image_id + ".png")
                image_arr = self._pre_process_image(image_filename)
                output_data = self._get_output_data(item)
                image_data.append([item.attrib["id"], image_arr, output_data])
                reporthook(i, len(xml_files))
        image_data = pd.DataFrame(image_data, columns=["subject", "image", "output"])
        image_data.to_pickle(self.image_data_file_name, protocol=2)
        return image_data

    def _process_subjects(self):
        train_subject_lists = ["trainset", "validationset1", "validationset2"]
        test_subject_lists = ["testset"]
        train_subjects = []
        test_subjects = []
        for train_list in train_subject_lists:
            subject_list = pd.read_csv(os.path.join(self._root, "subject", train_list+".txt"))
            train_subjects.append(subject_list.values)
        for test_list in test_subject_lists:
            subject_list = pd.read_csv(os.path.join(self._root, "subject", test_list+".txt"))
            test_subjects.append(subject_list.values)

        train_subjects = np.concatenate(train_subjects)
        test_subjects = np.concatenate(test_subjects)
        if self._parse_method == "form":
            new_train_subjects = []
            for i in train_subjects:
                form_subject_number = i[0].split("-")[0] + "-" + i[0].split("-")[1]
                new_train_subjects.append(form_subject_number)
            new_test_subjects = []
            for i in test_subjects:
                form_subject_number = i[0].split("-")[0] + "-" + i[0].split("-")[1]
                new_test_subjects.append(form_subject_number)
            train_subjects, test_subjects = new_train_subjects, new_test_subjects
                
        return train_subjects, test_subjects
                
    def _get_data(self):
        # Get the data
        if not os.path.isdir(self._root):
            os.makedirs(self._root)

        if os.path.isfile(self.image_data_file_name):
            images_data = pickle.load(open(self.image_data_file_name, 'rb'))
        else:
            self._download_xml()
            self._download_data()
            images_data = self._process_data()

        # Split data into train and test
        self._download_subject_list()
        train_subjects, test_subjects = self._process_subjects()
        if self._train:
            data = images_data[np.in1d(images_data["subject"], train_subjects)]
        else:
            data = images_data[np.in1d(images_data["subject"], test_subjects)]

        return data

    def __getitem__(self, idx):
        return (self._data[0].iloc[idx].image, self._data[0].iloc[idx].output)
