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

import socket
import jieba
import re
import json


# jieba.initialize()


def wrap_message(**args):
    return bytes(json.dumps(args), encoding="UTF-8")


class NJUNMTClient(object):
    def __init__(self, addr=("210.28.133.11", 21274)):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addr = addr
        # 建立连接:
        self.socket.connect(addr)
        # 接收欢迎消息:
        # print(self.socket.recv(1024).decode('utf-8'))
        self.reg_remove_space = re.compile('[ \t]+')

    def close(self):
        request = wrap_message(command="control", data="close")
        self.socket.sendall(request)
        self.socket.close()

    def preprocess(self, s):
        out = ' '.join(jieba.cut(s))
        out = self.reg_remove_space.sub(' ', out)
        return out

    def request(self, commnad, content, user_ip='127.0.0.1', debug=False):
        """
            :param source (str): 输入的待翻译文本
            :param debug (bool): 是否打印中间过程

            :return translation (str): 返回译文
        """
        if debug:
            print("{} source: {}".format(user_ip, content))

        # 对文本进行前处理
        if commnad == "translate":
            content = self.preprocess(content)

        request = wrap_message(command=commnad, content=content)

        if debug:
            print("{} source (preprocessed): {}".format(user_ip, content))

        # 发送处理后的文本至服务端
        self.socket.sendall(request)

        # 接收服务端返回的翻译结果
        response = str(self.socket.recv(1024 * 1024 * 8).strip(), 'UTF-8')

        if debug:
            print("{} translation: {}".format(user_ip, response))

        response = json.loads(response.strip())

        # print(response["translation"])
        return response


if __name__ == "__main__":
    """ 交互式使用示例 """
    client = NJUNMTClient(("127.0.0.1", 1234))
    while True:
        try:
            source = input("NJUNMT.INFO: Please input sentence to translate.\nsource: ")
            response = client.request("translate", source)
            print(response)
        except KeyboardInterrupt:
            break
