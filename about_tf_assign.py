
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

def syntax_test():
	ops.reset_default_graph()
	initializer =tf.keras.initializers.he_normal()
	x = tf.get_variable('test', shape = [3, 2], initializer = initializer)

	sess = tf.Session()

	sess.run(x.initializer)

	print("Variable x has value {} and shape {}".format( sess.run(x), x.shape) )

	sess.run( tf.assign( x, np.array( [ [1, 2, 3], [4, 5., 6] ] ), validate_shape = False ) )

	print("After tf.assign(x, <new numpy array>, x has value {}".format( sess.run(x) ) )

	sess.close()


# <USAGE>
# >>> import about_tf_assign as a0
# >>> a0.syntax_test()

# And you shall see :
'''
Variable x has value [[ 1.2451764  -0.34545043]
 [-0.1047621   0.45586464]
 [ 0.39827967  0.19549114]] and shape (3, 2)
After tf.assign(x, <new numpy array>, x has value [[1. 2. 3.]
 [4. 5. 6.]]
'''