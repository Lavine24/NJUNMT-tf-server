# -*- coding: UTF-8 -*- 

# Copyright 2018, Natural Language Processing Group, Nanjing University, 
#
#       Author: Zheng Zaixiang
#       Contact: zhengzx@nlp.nju.edu.cn 
#           or zhengzx.142857@gmail.com
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from njunmt.ensemble_experiment import *
import json
import sys
import errno
import socket

if sys.version_info[0] < 3:
    import SocketServer as socketserver
else:
    import socketserver


def wrap_message(**args):
    # return bytes(json.dumps(args), encoding="UTF-8")
    return json.dumps(args).encode()


def unwrap_message(json_str):
    try:
        return json.loads(json_str.decode())
    except:
        return {"command": "control", "content": "error"}


class SimpleEnsembleExperiment(EnsembleExperiment):
    def __init__(self,
                 model_configs,
                 model_dirs,
                 weight_scheme="average"):
        """ Initializes the ensemble experiment.

        Args:
            model_configs: A dictionary of all configurations.
            model_dirs: A list of model directories (checkpoints).
            weight_scheme: A string, the ensemble weights. See
              `EnsembleModel.get_ensemble_weights()` for more details.
        """
        super(EnsembleExperiment, self).__init__()
        self._model_dirs = model_dirs
        self._weight_scheme = weight_scheme
        infer_options = parse_params(
            params=model_configs["infer"],
            default_params=self.default_inference_options())
        
        self._model_configs = model_configs
        self._model_configs["infer"] = infer_options
        
        print_params("Model parameters: ", self._model_configs)
        
        self.experiment_spec = {
            'model_configs': model_configs
        }

        self.init_experiment()
        print("Start listening...")
    
    def init_vocab(self):
        vocab_source = Vocab(
            filename=self._model_configs["infer"]["source_words_vocabulary"],
            bpe_codes=self._model_configs["infer"]["source_bpecodes"])
        vocab_target = Vocab(
            filename=self._model_configs["infer"]["target_words_vocabulary"],
            bpe_codes=self._model_configs["infer"]["target_bpecodes"])
        
        return vocab_source, vocab_target
    
    def init_model(self, sess, vocab_source, vocab_target):
        print("Building model...")
        estimator_spec = model_fn_ensemble(
            self._model_dirs, vocab_source, vocab_target,
            weight_scheme=self._weight_scheme,
            inference_options=self._model_configs["infer"])

        predict_op = estimator_spec.predictions

        sess.run(tf.global_variables_initializer())
        print("Done.")

        return sess, predict_op, estimator_spec
    
    def init_experiment(self):
        """ Runs ensemble model. """
        print("Initialize experiment...")
        sess = self._build_default_session()
        vocab_source, vocab_target = self.init_vocab()
        sess, predict_op, estimator_spec = self.init_model(sess, vocab_source, vocab_target)
        
        self.experiment_spec.update(**{
            "session": sess,
            "predict_op": predict_op,
            "vocab_source": vocab_source,
            "vocab_target": vocab_target,
            "estimator_spec": estimator_spec,
            "model_info": {"model_dir": ", ".join(self._model_dirs)}
        })
        print("Done.")
    
    def reload_model(self, model_dirs):
        print("Reloading model...")

        self._model_dirs = model_dirs.split(",")

        self.experiment_spec['session'].close()
        tf.reset_default_graph()
        self.init_experiment()

        print("Done.")


class TranslateServer(socketserver.TCPServer):
    def init_experiment(self, **args):
        experiment = SimpleEnsembleExperiment(**args)
        self._experiment = experiment
        self.experiment_spec = experiment.experiment_spec

    def reload_model(self, model_dirs):
        self._experiment.reload_model(model_dirs)


class TranslateRequestHandler(socketserver.BaseRequestHandler, object):
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """

    def __init__(self, request, client_address, server):
        self.experiment_spec = server.experiment_spec
        super(TranslateRequestHandler, self).__init__(request, client_address, server)

    def preprocess_raw(self, raw_data):
        """
        :param raw_data: json string
            {
                "command": str (control, translate),
                "data": str
            }

        :return: processed dict
            {
                "command": str (control, translate),
                "data": tf feeding_data (if not translate then None)
            }
        """
        msg = unwrap_message(raw_data)

        if msg["command"] == "translate":
            lines = msg["content"].strip().split("\n")

            text_inputter = TextLineInputter(
                line_readers=[LineReader(
                    data=lines,
                    preprocessing_fn=lambda x: self.experiment_spec["vocab_source"].convert_to_idlist(x))],
                padding_id=self.experiment_spec["vocab_source"].pad_id,
                batch_size=self.experiment_spec["model_configs"]["infer"]["batch_size"])

            feeding_data = text_inputter.make_feeding_data(self.experiment_spec["estimator_spec"].input_fields)

            return {"command": "translate", "content": feeding_data}
        return msg

    def handle(self):
        # self.request is the TCP socket connected to the client
        print("User from ({}:{}) connected.:".format(*self.client_address))

        while True:
            try:
                raw_data = self.request.recv(1024).strip()  # json string
                print(raw_data)

                # preprocess raw_data to request (dict)
                request = self.preprocess_raw(raw_data)
                print(request)

                if request["command"] == "translate":
                    trans_outputs = []
                    sources = []
                    for feeding_data in request["content"]:
                        source, trans_output, trans_score = infer(
                            sess=self.experiment_spec["session"],
                            prediction_op=self.experiment_spec["predict_op"],
                            infer_data=feeding_data,
                            output=None,
                            vocab_source=self.experiment_spec["vocab_source"],
                            vocab_target=self.experiment_spec["vocab_target"],
                            delimiter=self.experiment_spec["model_configs"]["infer"]["delimiter"],
                            output_attention=False,
                            tokenize_output=self.experiment_spec["model_configs"]["infer"]["char_level"],
                            verbose=True)
                        sources.extend(source)
                        trans_outputs.extend(trans_output)

                    sources = "\n".join(sources)
                    trans_outputs = "\n".join(trans_outputs)
                    response = wrap_message(status="success", info="", source=sources, translation=trans_outputs,
                                             model_info=self.experiment_spec["model_info"])

                elif request["command"] == "control":
                    if request["content"] == "close":
                        break

                elif request["command"] == "reload":
                    new_model_dirs = request["content"]
                    self.server.reload_model(new_model_dirs)
                    response = wrap_message(status="success",
                                             info="Reloaded model from {}".format(new_model_dirs),
                                             model_info=self.experiment_spec["model_info"])

                self.request.sendall(response)

            except Exception as e:
                response = wrap_message(status="error")
                try:
                    self.request.sendall(response)
                except socket.error as e:
                    print("Close connection from {}:{}.".format(*self.client_address))
                    break

