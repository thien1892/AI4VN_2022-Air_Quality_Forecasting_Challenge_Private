import tensorflow_probability as tfp
import tensorflow as tf

def mdape(y_true, y_pred):
   squared_difference = tf.abs((y_true - y_pred)/ y_true)
   return tfp.stats.percentile(squared_difference, 50.0, interpolation='midpoint')

def loss_custom(y_true, y_pred):
    mae = tf.keras.losses.mean_absolute_error(y_true, y_pred)
    rmse = tf.sqrt(tf.keras.losses.mean_squared_error(y_true, y_pred))
    return (1.5 * mae + rmse) /(1.5 + 1)