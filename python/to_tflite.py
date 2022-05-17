import numpy as np
import tensorflow as tf

input_model = "./testa/saved_model.pb"
input_name = "gaussian_noise"
output_node_name = "dense_1"

output_model = "./output.tflite"

converter = tf.lite.TFLiteConverter.from_saved_model("./testa");
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_quant_model = converter.convert()
with open(output_model, 'wb') as o_:
    o_.write(tflite_quant_model)


