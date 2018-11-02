# Copyright 2017 Natural Language Processing Group, Nanjing University, zhaocq.nlp@gmail.com.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Classes for reading in data. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod

import numpy
import six
import tensorflow as tf

from njunmt.data.data_reader import LineReader
from njunmt.utils.constants import Constants
from njunmt.utils.constants import concat_name
from njunmt.utils.misc import padding_batch_data
from njunmt.utils.expert_utils import repeat_n_times


def do_bucketing(pivot, *args):
    """ Sorts the `pivot` and args by length of `pivot`.

    Args:
        pivot: The pivot.
        args: A list of others.

    Returns: The same as inputs.
    """
    tlen = numpy.array([len(t) for t in pivot])
    tidx = tlen.argsort()
    _pivot = [pivot[i] for i in tidx]
    _args = []
    for ele in args:
        _args.append([ele[i] for i in tidx])
    return _pivot, _args


def pack_feed_dict(name_prefixs, origin_datas, paddings, input_fields):
    """

    Args:
        name_prefixs: A prefix string of a list of strings.
        origin_datas: Data list or a list of data lists.
        paddings: A padding id or a list of padding ids.
        input_fields: A list of input fields dict.

    Returns: A dict for while loop.
    """
    data = dict()
    data["feed_dict"] = dict()

    def map_fn(n, d, p):
        # n: name prefix
        # d: data list
        # p: padding symbol
        data[concat_name(n, Constants.IDS_NAME)] = d
        n_samples = len(d)
        n_devices = len(input_fields)
        n_samples_per_gpu = n_samples // n_devices
        if n_samples % n_devices > 0:
            n_samples_per_gpu += 1

        def _feed_batchs(_start_idx, _inpf):

            if _start_idx * n_samples_per_gpu >= n_samples:
                return 0
            x, x_len = padding_batch_data(
                d[_start_idx * n_samples_per_gpu:(_start_idx + 1) * n_samples_per_gpu], p)
            data["feed_dict"][_inpf[concat_name(n, Constants.IDS_NAME)]] = x
            data["feed_dict"][_inpf[concat_name(n, Constants.LENGTH_NAME)]] = x_len
            return len(x_len)

        parallels = repeat_n_times(
            n_devices, _feed_batchs,
            list(range(n_devices)), input_fields)
        data["feed_dict"]["parallels"] = parallels

    if isinstance(name_prefixs, six.string_types):
        map_fn(name_prefixs, origin_datas, paddings)
    else:
        [map_fn(n, d, p) for n, d, p in zip(name_prefixs, origin_datas, paddings)]
    return data


@six.add_metaclass(ABCMeta)
class TextInputter(object):
    """Base class for inputters. """

    def __init__(self):
        pass

    @abstractmethod
    def make_feeding_data(self, *args, **kwargs):
        """ Processes the data file and return an iterable instance for loop. """
        raise NotImplementedError


class TextLineInputter(TextInputter):
    """ Class for reading in lines.  """

    def __init__(self,
                 line_readers,
                 padding_id,
                 batch_size):
        """ Initializes the parameters for this inputter.

        Args:
            line_readers: A LineReader instance or a list of LineReader instances.
            padding_id: An integer for padding.
            batch_size: An integer value indicating the number of
              sentences passed into one step. Sentences will be padded by EOS.

        Raises:
            ValueError: if `batch_size` is None.
        """
        super(TextLineInputter, self).__init__()
        self._readers = line_readers
        self._batch_size = batch_size
        if self._batch_size is None:
            raise ValueError("batch_size should be provided.")
        self._padding_id = padding_id

    def _make_feeding_data_from(self,
                                reader,
                                input_fields,
                                name_prefix):
        """ Processes the data file and return an iterable instance for loop.

        Args:
            reader: A LineReader instance.
            input_fields: A dict of placeholders.
            name_prefix: A string, the key name prefix for feed_dict.

        Returns: An iterable instance that packs feeding dictionary
                   for `tf.Session().run` according to the `filename`.
        """
        assert isinstance(reader, LineReader)
        ss_buf = []
        while True:
            encoded_ss = reader.next()
            if encoded_ss == "":
                break
            if encoded_ss is None:
                continue
            ss_buf.append(encoded_ss)
        reader.close()
        data = []
        batch_data_idx = 0

        while batch_data_idx < len(ss_buf):
            data.append(pack_feed_dict(
                name_prefixs=name_prefix,
                origin_datas=ss_buf[batch_data_idx: batch_data_idx + self._batch_size],
                paddings=self._padding_id,
                input_fields=input_fields))
            batch_data_idx += self._batch_size
        return data

    def make_feeding_data(self, input_fields,
                          name_prefix=Constants.FEATURE_NAME_PREFIX):
        """ Processes the data file(s) and return an iterable
        instance for loop.

        Args:
            input_fields: A dict of placeholders.
            name_prefix: A string, the key name prefix for feed_dict.

        Returns: An iterable instance or a list of iterable
                   instances according to the `data_field_name`
                   in the constructor.
        """
        if isinstance(self._readers, list):
            return [self._make_feeding_data_from(
                reader, input_fields, name_prefix)
                    for reader in self._readers]
        return self._make_feeding_data_from(
            self._readers, input_fields, name_prefix)


class ParallelTextInputter(TextInputter):
    """ Class for reading in parallel texts.  """

    def __init__(self,
                 features_reader,
                 labels_reader,
                 features_padding_id,
                 labels_padding_id,
                 batch_size=None,
                 batch_tokens_size=None,
                 shuffle_every_epoch=None,
                 fill_full_batch=False,
                 bucketing=True):
        """ Initializes the parameters for this inputter.

        Args:
            features_reader: A LineReader instance for features.
            labels_reader: A LineReader instance for labels.
            features_padding_id: An integer for features padding.
            labels_padding_id: An integer for labels padding.
            batch_size: An integer value indicating the number of
              sentences passed into one step. Sentences will be padded by EOS.
            batch_tokens_size: An integer value indicating the number of
              words of each batch. If provided, sentence pairs will be batched
              together by approximate sequence length.
            shuffle_every_epoch: A string type. If provided, use it as postfix
              of shuffled data file name.
            fill_full_batch: Whether to ensure each batch of data has `batch_size`
              of datas.
            bucketing: Whether to sort the sentences by length of labels.

        Raises:
            ValueError: if both `batch_size` and `batch_tokens_size` are
              not provided.

        """
        super(ParallelTextInputter, self).__init__()
        self._features_reader = features_reader
        self._labels_reader = labels_reader
        self._features_padding_id = features_padding_id
        self._labels_padding_id = labels_padding_id
        self._batch_size = batch_size
        self._batch_tokens_size = batch_tokens_size
        self._shuffle_every_epoch = shuffle_every_epoch
        self._fill_full_batch = fill_full_batch
        self._bucketing = bucketing
        if self._batch_size is None and self._batch_tokens_size is None:
            raise ValueError("Either batch_size or batch_tokens_size should be provided.")
        if (self._batch_size is not None) and (self._batch_tokens_size is not None):
            tf.logging.info("batching data according to batch_tokens_size={}, "
                            "and use batch_size={} as an auxiliary variable.".format(batch_tokens_size, batch_size))
        if batch_tokens_size is None:
            self._cache_size = self._batch_size * 128  # 80 * 128 = 10240
        else:
            self._cache_size = self._batch_tokens_size * 6  # 4096 * 6 := 25000
            if batch_size is None:
                self._batch_size = 32

    def _small_parallel_data(self, input_fields):
        """ Function for reading small scale parallel data for evaluation.

        Args:
            input_fields: A dict of placeholders or a list of dicts.

        Returns: A list of feeding data.
        """
        ss_buf = []
        tt_buf = []
        while True:
            ss = self._features_reader.next()
            tt = self._labels_reader.next()
            if ss == "" or tt == "":
                break
            if ss is None or tt is None:
                continue
            ss_buf.append(ss)
            tt_buf.append(tt)
        self._features_reader.close()
        self._labels_reader.close()
        if self._bucketing:
            tt_buf, ss_buf = do_bucketing(tt_buf, ss_buf)
            ss_buf = ss_buf[0]
        data = []
        batch_data_idx = 0
        while batch_data_idx < len(ss_buf):
            data.append(
                pack_feed_dict(
                    name_prefixs=[Constants.FEATURE_NAME_PREFIX, Constants.LABEL_NAME_PREFIX],
                    origin_datas=[ss_buf[batch_data_idx: batch_data_idx + self._batch_size],
                                  tt_buf[batch_data_idx: batch_data_idx + self._batch_size]],
                    paddings=[self._features_padding_id, self._labels_padding_id],
                    input_fields=input_fields))
            batch_data_idx += self._batch_size
        return data

    def make_feeding_data(self,
                          input_fields,
                          in_memory=False):
        """ Processes the data files and return an iterable
              instance for loop.
        Args:
            input_fields: A dict of placeholders or a list of dicts.
            in_memory: Whether to load all data into memory.

        Returns: An iterable instance.
        """
        if in_memory and self._fill_full_batch:
            raise ValueError(
                "`in_memory` option with _SmallParallelData fn now only deal with small evaluation data. "
                "`fill_full_batch` for ParallelTextInputter is available for training data only.")
        if in_memory and self._shuffle_every_epoch:
            raise ValueError(
                "`in_memory` option with _SmallParallelData fn now only deal with small evaluation data. "
                "`shuffle_every_epoch` for ParallelTextInputter is available for training data only.")
        if in_memory:
            return self._small_parallel_data(input_fields)
        return self._BigParallelDataIterator(
            input_fields=input_fields,
            **self.__dict__)

    class _BigParallelDataIterator(object):
        """ An iterator class for reading parallel data. """

        def __init__(self,
                     input_fields,
                     **kwargs):
            """ Initializes.

            Args:
                input_fields: A dict of placeholders or a list of dicts.
                **kwargs: The attributes of the parent ParallelTextInputter.
            """
            for k, v in kwargs.items():
                setattr(self, k, v)

            self._reset()

            self._features_buffer = []
            self._labels_buffer = []
            self._features_len_buffer = []
            self._labels_len_buffer = []
            self._end_of_data = False
            self._input_fields = input_fields

        def __iter__(self):
            return self

        def _reset(self):
            """ shuffle features & labels file. """
            if not hasattr(self, "_shuffled_features_file"):
                self._shuffled_features_file = "features_file." + str(self._shuffle_every_epoch)
                self._shuffled_labels_file = "labels_file." + str(self._shuffle_every_epoch)
            argsort_index = self._features_reader.reset(
                do_shuffle=self._shuffle_every_epoch,
                shuffle_to_file=self._shuffled_features_file,
                argsort_index=None)
            _ = self._labels_reader.reset(
                do_shuffle=self._shuffle_every_epoch,
                shuffle_to_file=self._shuffled_labels_file,
                argsort_index=argsort_index)

        def __next__(self):
            """ capable for python3 """
            return self.next()

        def next(self):
            if self._end_of_data:
                self._end_of_data = False
                self._reset()
                raise StopIteration

            assert len(self._features_buffer) == len(self._labels_buffer), "Buffer size mismatch"
            if len(self._features_buffer) < self._batch_size:
                cnt = len(self._features_buffer)
                while cnt < self._cache_size:
                    ss = self._features_reader.next()
                    tt = self._labels_reader.next()
                    if ss == "" or tt == "":
                        break
                    if ss is None or tt is None:
                        continue
                    cnt += 1
                    self._features_buffer.append(ss)
                    self._labels_buffer.append(tt)
                if len(self._features_buffer) == 0 or len(self._labels_buffer) == 0:
                    self._end_of_data = False
                    self._reset()
                    raise StopIteration
                if self._bucketing:
                    # sort by len
                    self._labels_buffer, self._features_buffer \
                        = do_bucketing(self._labels_buffer, self._features_buffer)
                    self._features_buffer = self._features_buffer[0]
                self._features_len_buffer = [len(s) for s in self._features_buffer]
                self._labels_len_buffer = [len(t) for t in self._labels_buffer]
            if self._fill_full_batch and len(self._features_buffer) < self._batch_size:
                self._end_of_data = False
                self._reset()
                raise StopIteration
            local_batch_size = self._batch_size
            if self._batch_tokens_size is not None:  # batching data by num of tokens
                sum_s = numpy.sum(self._features_len_buffer[: local_batch_size])
                sum_t = numpy.sum(self._labels_len_buffer[: local_batch_size])
                while True:
                    if sum_s >= self._batch_tokens_size or sum_t >= self._batch_tokens_size:
                        break
                    if self._batch_tokens_size - sum_s < 20 or self._batch_tokens_size - sum_t < 20:
                        break
                    if local_batch_size >= len(self._features_len_buffer):
                        break
                    sum_s += self._features_len_buffer[local_batch_size]
                    sum_t += self._labels_len_buffer[local_batch_size]
                    local_batch_size += 1
            features = self._features_buffer[:local_batch_size]
            labels = self._labels_buffer[:local_batch_size]
            if len(features) < local_batch_size:
                del self._features_buffer[:]
                del self._labels_buffer[:]
                del self._features_len_buffer[:]
                del self._labels_len_buffer[:]
            else:
                del self._features_buffer[:local_batch_size]
                del self._labels_buffer[:local_batch_size]
                del self._features_len_buffer[:local_batch_size]
                del self._labels_len_buffer[:local_batch_size]
            if len(features) <= 0 or len(labels) <= 0:
                self._end_of_data = False
                self._reset()
                raise StopIteration
            ret_data = pack_feed_dict(
                name_prefixs=[Constants.FEATURE_NAME_PREFIX, Constants.LABEL_NAME_PREFIX],
                origin_datas=[features, labels],
                paddings=[self._features_padding_id, self._labels_padding_id],
                input_fields=self._input_fields)
            if self._fill_full_batch:
                ret_data["feed_dict"].pop("parallels")
            return ret_data
