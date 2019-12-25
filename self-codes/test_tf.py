# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 18:12:39 2017

@author: hoge
"""

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))