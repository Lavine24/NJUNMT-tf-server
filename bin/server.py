# -*- coding: utf-8 -*-
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
"""Entrance for inference from a trained NMT model. """
import tensorflow as tf

from njunmt.utils.configurable import define_tf_flags
from njunmt.utils.configurable import update_configs_from_flags
from njunmt.utils.configurable import load_from_config_path

from .translate_server import TranslateServer, TranslateRequestHandler

# define arguments for infer.py
# format: {arg_name: [type, default_val, helper]}
INFER_ARGS = {
    "config_paths": ["string", "", """Path to a yaml configuration files defining FLAG values.
                                   Multiple files can be separated by commas. Files are merged recursively.
                                   Setting a key in these files is equivalent to
                                   setting the FLAG value with the same name."""],
    "infer": ["string", "", """A yaml-style string defining the inference options."""],
    "infer_data": ["string", "", """A yaml-style string defining the inference data files."""],
    "model_dir": ["string", "models", """The path to load models. """],
    "weight_scheme": ["string", "average", """The weight scheme for ensemble, by default: average."""],
    "server_address": ["string", "", """IP and port, such as 123.456.789.0:1234"""],
}

FLAGS = define_tf_flags(INFER_ARGS)


def main(_argv):
    # load flags from config file
    model_configs = load_from_config_path(FLAGS.config_paths)
    # replace parameters in configs_file with tf FLAGS
    model_configs = update_configs_from_flags(model_configs, FLAGS, INFER_ARGS.keys())

    model_dirs = FLAGS.model_dir.strip().split(",")

    ip, port = FLAGS.server_address.split(":")
    addr = (ip, int(port))

    server = TranslateServer(addr, TranslateRequestHandler)
    server.init_experiment(model_configs=model_configs,
                           model_dirs=model_dirs,
                           weight_scheme=FLAGS.weight_scheme)
    server.serve_forever()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
