import tensorflow as tf
from tensorflow.keras import layers, Model


def _conv_bn_relu(x, filters, kernel_size, name):
	conv = layers.Conv2D(
		filters,
		kernel_size=kernel_size,
		strides=(1, 1),
		padding="valid",
		use_bias=False,
		name=f"{name}_conv",
	)(x)
	bn = layers.BatchNormalization(name=f"{name}_bn")(conv)
	return layers.Activation("relu", name=f"{name}_relu")(bn)


def _dense_bn_relu(x, units, name):
	d = layers.Dense(units, use_bias=False, name=f"{name}_dense")(x)
	bn = layers.BatchNormalization(name=f"{name}_bn")(d)
	return layers.Activation("relu", name=f"{name}_relu")(bn)


def build_pointnet_classifier(
	num_particles,
	feature_dim=3,
	output_dim=5,
	dropout_rate=0.3,
	base_width=16,
):
	"""
	PointNet classifier adapted for jet tagging.
	- Input: (batch, num_particles, feature_dim) where feature_dim=3 for (pt, eta, phi)
	- Output: (batch, output_dim)

	base_width controls the channel widths:
	  conv layers: [base_width, base_width*2, base_width*4]
	  fc layers:   [base_width*2, base_width]
	Default base_width=16 reproduces the original architecture.
	Use base_width=20 for ~1.3M FLOPs.
	"""
	inputs = layers.Input((num_particles, feature_dim), name="features")

	# Add a channel dimension to match Conv2D expectations: (B, N, F, 1)
	x = layers.Lambda(lambda t: tf.expand_dims(t, axis=-1), name="expand_dims")(inputs)

	# Point functions (MLPs implemented as 2D convs)
	x = _conv_bn_relu(x, base_width,     (1, feature_dim), name="conv1")
	x = _conv_bn_relu(x, base_width * 2, (1, 1),           name="conv2")
	x = _conv_bn_relu(x, base_width * 4, (1, 1),           name="conv3")

	# Symmetric function: max pooling across N (particle) dimension
	x = layers.GlobalMaxPooling2D(name="global_max_pool")(x)

	# MLP on global feature
	x = _dense_bn_relu(x, base_width * 2, name="fc1")
	x = _dense_bn_relu(x, base_width,     name="fc2")
	x = layers.Dropout(dropout_rate, name="dropout")(x)

	activation = "sigmoid" if output_dim == 1 else "softmax"
	outputs = layers.Dense(output_dim, activation=activation, name="head")(x)
	return Model(inputs, outputs, name="PointNetJetClassifier")