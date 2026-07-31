import tensorflow as tf
from tensorflow.keras import layers, Model
import math


# ========== Core Components ==========

class Serialization2D(layers.Layer):
    """
    Serialization and sorting for point sequences.
    - 'morton': Z-order (Morton) curve over the quantized spatial grid. 2D on (eta, phi)
                for jets; 3D on (x, y, z) when coord_dim=3.
    - 'pt':     sort by pt descending            (jets only)
    - 'kt':     sort by pt * sqrt(eta^2 + phi^2) descending (jets only)

    Inputs:
      x:      features [B, N, C]
      coords: jets (weighted_input=True): [B, N, 2] = (eta, phi) or [B, N, 3] = (pt, eta, phi)
              generic (weighted_input=False): [B, N, coord_dim] = (x, y, z)

    NOTE: the class name is kept for checkpoint/layer-name compatibility with the existing
    jet runs even though it now also handles 3D.
    """
    def __init__(self, grid_size=0.05, sort_by="morton", coord_dim=2, weighted_input=True, **kwargs):
        super().__init__(**kwargs)
        if coord_dim not in (2, 3):
            raise ValueError("coord_dim must be 2 or 3")
        if not weighted_input and sort_by in ("pt", "kt"):
            raise ValueError(f'sort_by="{sort_by}" requires a weight channel (weighted_input=True)')
        self.grid_size = grid_size
        self.sort_by = sort_by
        self.coord_dim = coord_dim
        self.weighted_input = weighted_input

    def _interleave_bits(self, axes):
        """Z-order (Morton) code interleaving the bits of `len(axes)` quantized axes."""
        d = len(axes)
        axes = [tf.cast(a, tf.int64) for a in axes]
        z = tf.zeros_like(axes[0])

        one = tf.constant(1, dtype=tf.int64)
        # Keep the total code width under 63 bits regardless of dimensionality.
        n_bits = 63 // d
        # Interleave bits with a static Python loop so graph tracing stays simple.
        for i in range(n_bits):
            shift = tf.constant(i, dtype=tf.int64)
            for axis_i, a in enumerate(axes):
                bit = (tf.bitwise.right_shift(a, shift) & one)
                z = z | tf.bitwise.left_shift(bit, d * i + axis_i)
        return z

    def call(self, inputs):
        # Support both call(x, coords) and call([x, coords])
        if isinstance(inputs, (list, tuple)):
            x, coords = inputs
        else:
            raise ValueError("Serialization2D expects inputs=[x, coords]")
        B = tf.shape(x)[0]
        N = tf.shape(x)[1]
        if coords.shape.rank is None or coords.shape.rank != 3:
            raise ValueError("coords must be rank-3: [B, N, coord_dim (+1 for a weight channel)]")

        # Default: morton
        if self.sort_by == "morton":
            # Spatial axes are the trailing coord_dim channels, so this works whether or
            # not a leading weight (pt) channel is present.
            spatial = [coords[..., i - self.coord_dim] for i in range(self.coord_dim)]
            grid = []
            for a in spatial:
                a_min = tf.reduce_min(a, axis=1, keepdims=True)
                grid.append(tf.cast((a - a_min) / self.grid_size, tf.int32))
            morton_code = self._interleave_bits(grid)
            sort_idx = tf.argsort(morton_code, axis=1)  # ascending
        elif self.sort_by == "pt":
            if coords.shape[-1] < 3:
                raise ValueError("pt sorting requires coords[...,0]=pt with coords last-dim=3")
            pt = coords[..., 0]
            # sort by pt descending
            sort_idx = tf.argsort(pt, axis=1, direction="DESCENDING")
        elif self.sort_by == "kt":
            if coords.shape[-1] < 3:
                raise ValueError("kt sorting requires coords[...,0:3]=(pt,eta,phi)")
            pt = coords[..., 0]
            eta = coords[..., 1]
            phi = coords[..., 2]
            kt = pt * tf.sqrt(eta * eta + phi * phi)
            sort_idx = tf.argsort(kt, axis=1, direction="DESCENDING")
        else:
            raise ValueError(f"Unknown sort_by: {self.sort_by}")

        # Prepare indices for gathering
        batch_idx = tf.range(B)[:, None]
        batch_idx = tf.tile(batch_idx, [1, N])
        gather_idx = tf.stack([batch_idx, sort_idx], axis=-1)

        x_sorted = tf.gather_nd(x, gather_idx)
        coords_sorted = tf.gather_nd(coords, gather_idx)
        return x_sorted, coords_sorted

class GeometricCPE(layers.Layer):
    """
    Convolutional Position Encoding (xCPE) that respects jet geometry.
    Uses a depthwise convolution on the quantized coordinate grid: 2D on (eta, phi) for
    jets, 3D on (x, y, z) when coord_dim=3.
    This serves as the efficient positional injection mechanism in PTv3.
    """
    def __init__(self, channels, kernel_size=3, grid_size=0.05, coord_dim=2,
                 max_grid_cells=1 << 20, **kwargs):
        super().__init__(**kwargs)
        if coord_dim not in (2, 3):
            raise ValueError("coord_dim must be 2 or 3")
        self.channels = channels
        self.kernel_size = kernel_size
        self.grid_size = grid_size
        self.coord_dim = coord_dim
        self.max_grid_cells = max_grid_cells

        # Conv on the spatial grid (mimicking sparse conv)
        conv_cls = layers.Conv2D if coord_dim == 2 else layers.Conv3D
        self.conv2d = conv_cls(
            channels,
            kernel_size=kernel_size,
            padding="same",
            groups=channels,  # Depthwise convolution for efficiency
            use_bias=True
        )
        self.pointwise = layers.Dense(channels)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, coords):
        """
        Args:
            x:      Features [B, N, C]
            coords: Coordinates [B, N, coord_dim]
        """
        B = tf.shape(x)[0]
        N = tf.shape(x)[1]
        C = self.channels
        D = self.coord_dim

        residual = x

        # Quantize to grid
        grid_idx = []
        for i in range(D):
            a = coords[..., i]
            a_min = tf.reduce_min(a, axis=1, keepdims=True)
            grid_idx.append(tf.cast((a - a_min) / self.grid_size, tf.int32))

        # Get grid dimensions
        dims = [tf.maximum(tf.reduce_max(g) + 1, 1) for g in grid_idx]

        # Cap total cells so a mis-set grid_size cannot allocate an enormous buffer.
        if self.max_grid_cells is not None:
            total = dims[0]
            for d in dims[1:]:
                total = total * d
            cap = tf.constant(self.max_grid_cells, dtype=total.dtype)
            shrink = tf.maximum(
                tf.cast(tf.math.ceil(tf.pow(tf.cast(total, tf.float32) / tf.cast(cap, tf.float32),
                                            1.0 / float(D))), tf.int32),
                1,
            )
            shrink = tf.where(total > cap, shrink, tf.ones_like(shrink))
            grid_idx = [g // shrink for g in grid_idx]
            dims = [tf.maximum(tf.reduce_max(g) + 1, 1) for g in grid_idx]

        grid_idx = [tf.clip_by_value(g, 0, d - 1) for g, d in zip(grid_idx, dims)]

        # Scatter points onto the grid [B, *dims, C]
        batch_idx = tf.range(B)[:, None]
        batch_idx = tf.tile(batch_idx, [1, N])

        indices = tf.stack([batch_idx] + grid_idx, axis=-1)  # [B, N, 1+D]

        indices = tf.reshape(indices, [-1, D + 1])
        features = tf.reshape(x, [-1, C])

        # Create a sparse tensor equivalent
        grid = tf.scatter_nd(
            indices,
            features,
            tf.stack([B] + dims + [tf.constant(C, dtype=tf.int32)])
        )

        # Apply the depthwise convolution
        grid = self.conv2d(grid)

        # Gather back to points
        out = tf.gather_nd(grid, tf.reshape(indices, [B, N, D + 1]))

        out = self.pointwise(out)
        out = self.norm(out)

        return residual + out

class QuantizedRPE(layers.Layer):
    """
    RPE using quantized relative positions with learnable table (reintroduced if needed).
    """
    def __init__(self, num_heads, quantization_bins=32, coord_dim=2, wrap_last_coord=None, **kwargs):
        super().__init__(**kwargs)
        if coord_dim not in (2, 3):
            raise ValueError("coord_dim must be 2 or 3")
        self.num_heads = num_heads
        self.bins = quantization_bins
        self.coord_dim = coord_dim
        # The last axis is periodic (phi) for jets only.
        self.wrap_last_coord = (coord_dim == 2) if wrap_last_coord is None else wrap_last_coord
        
        # Learnable bias table for quantized relative positions
        # Each dimension (eta, phi) gets its own set of bins
        self.rpe_table = self.add_weight(
            name="rpe_table",
            shape=[coord_dim * quantization_bins, num_heads],
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True
        )
    
    def call(self, coords):
        """
        Args:
            coords: [B, T, coord_dim]. For jets coords[...,0]=eta, coords[...,1]=phi.
        Returns:
            bias: [B, H, T, T]
        """
        pi = tf.constant(math.pi, dtype=coords.dtype)
        half = self.bins // 2
        bias = None

        for axis in range(self.coord_dim):
            c = coords[..., axis]  # [B, T]
            rel = c[:, :, None] - c[:, None, :]  # [B, T, T]

            is_periodic = self.wrap_last_coord and axis == self.coord_dim - 1
            if is_periodic:
                rel = tf.math.floormod(rel + pi, 2 * pi) - pi
                rel_range = pi
            else:
                # Avoid division by zero
                rel_range = tf.maximum(tf.reduce_max(tf.abs(rel)), tf.cast(1e-6, coords.dtype))

            bins = tf.cast(rel / rel_range * half, tf.int32)
            bins = tf.clip_by_value(bins, -half, half - 1)
            idx = bins + half + axis * self.bins

            axis_bias = tf.gather(self.rpe_table, idx)  # [B, T, T, H]
            bias = axis_bias if bias is None else bias + axis_bias

        # Combine biases
        bias = tf.transpose(bias, [0, 3, 1, 2])  # [B, H, T, T]
        
        return bias


class PatchedAttention(layers.Layer):
    """Local attention with patching on the serialized sequence."""
    def __init__(self, d_model, num_heads, patch_size, dropout=0.0, use_rpe=False, coord_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.patch_size = patch_size
        self.use_rpe = use_rpe
        
        self.wq = layers.Dense(d_model, use_bias=True)
        self.wk = layers.Dense(d_model, use_bias=True)
        self.wv = layers.Dense(d_model, use_bias=True)
        self.wo = layers.Dense(d_model, use_bias=True)
        self.dropout = layers.Dropout(dropout)
        self.rpe = QuantizedRPE(num_heads, coord_dim=coord_dim) if use_rpe else None
    
    def _split_heads(self, x, num_heads):
        b, t, d = tf.unstack(tf.shape(x)[:3])
        x = tf.reshape(x, [b, t, num_heads, d // num_heads])
        return tf.transpose(x, [0, 2, 1, 3])
    
    def _merge_heads(self, x):
        b, h, t, dh = tf.unstack(tf.shape(x))
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [b, t, h * dh])
    
    def call(self, x, coords, training=False):
        B, T, D = tf.unstack(tf.shape(x))
        P = self.patch_size
        
        # Pad if necessary
        pad_len = (P - T % P) % P
        if pad_len > 0:
            x = tf.pad(x, [[0, 0], [0, pad_len], [0, 0]])
            coords = tf.pad(coords, [[0, 0], [0, pad_len], [0, 0]])
        
        T_padded = T + pad_len
        num_patches = T_padded // P
        
        # Reshape to patches
        x_patched = tf.reshape(x, [B, num_patches, P, D])
        x_patched = tf.reshape(x_patched, [B * num_patches, P, D])
        coord_dim = coords.shape[-1]
        coords_patched = tf.reshape(coords, [B, num_patches, P, coord_dim])
        coords_patched = tf.reshape(coords_patched, [B * num_patches, P, coord_dim])
        
        # Attention within patches
        q = self._split_heads(self.wq(x_patched), self.num_heads)
        k = self._split_heads(self.wk(x_patched), self.num_heads)
        v = self._split_heads(self.wv(x_patched), self.num_heads)
        
        dk = tf.cast(self.d_head, x.dtype)
        scores = tf.einsum("bhtd,bhTd->bhtT", q, k) / tf.math.sqrt(dk)
        
        if self.use_rpe:
            bias = self.rpe(coords_patched)
            scores = scores + bias
        
        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.dropout(weights, training=training)
        
        out = tf.einsum("bhtT,bhTd->bhtd", weights, v)
        out = self._merge_heads(out)
        out = tf.ensure_shape(out, [None, None, self.d_model])
        out = self.wo(out)
        
        # Reshape back
        out = tf.reshape(out, [B, num_patches, P, self.d_model])
        out = tf.reshape(out, [B, T_padded, self.d_model])
        
        # Remove padding
        if pad_len > 0:
            out = out[:, :-pad_len, :]
        
        return out


class PTv3Block(layers.Layer):
    """Transformer block with CPE."""
    def __init__(self, d_model, d_ff, num_heads, patch_size, cpe_k=3, grid_size=0.05, dropout=0.0, use_rpe=False, coord_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.coord_dim = coord_dim
        self.cpe = GeometricCPE(d_model, kernel_size=cpe_k, grid_size=grid_size, coord_dim=coord_dim)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = PatchedAttention(d_model, num_heads, patch_size, dropout=dropout, use_rpe=use_rpe, coord_dim=coord_dim)
        self.drop1 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_ff, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])
        self.drop2 = layers.Dropout(dropout)
    
    def call(self, inputs, training=False):
        x, coords = inputs
        
        # CPE
        x = self.cpe(x, coords)
        
        # Attention
        y = self.attn(self.norm1(x), coords, training=training)
        x = x + self.drop1(y, training=training)
        
        # FFN
        y = self.ffn(self.norm2(x))
        x = x + self.drop2(y, training=training)
        
        return [x, coords]


class SerializedPooling2D(layers.Layer):
    """
    Pools along the serialized 1D sequence (Grid Pooling approximation).
    The input sequence is assumed to be already sorted by the serialization code.
    """
    def __init__(self, out_dim, stride=2, **kwargs):
        super().__init__(**kwargs)
        self.stride = stride
        self.proj = layers.Dense(out_dim)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs):
        x, coords = inputs
        B, N = tf.shape(x)[0], tf.shape(x)[1]
        channels = x.shape[-1]
        coord_channels = tf.shape(coords)[-1]
        
        # Pad and group (This is now simpler as it just pads the sorted sequence)
        N_out = N // self.stride
        remainder = N % self.stride
        if remainder != 0:
            pad_len = self.stride - remainder
            # Pad at the end of the sequence
            x = tf.pad(x, [[0, 0], [0, pad_len], [0, 0]])
            coords = tf.pad(coords, [[0, 0], [0, pad_len], [0, 0]])
            N_out = (N + pad_len) // self.stride
        
        # Reshape for pooling
        x_grouped = tf.reshape(x, [B, N_out, self.stride, channels])
        coords_grouped = tf.reshape(coords, [B, N_out, self.stride, coord_channels])
        
        # Pool (Max features, Mean coordinates)
        x_pooled = tf.reduce_max(x_grouped, axis=2)
        coords_pooled = tf.reduce_mean(coords_grouped, axis=2)
        
        # Project and Normalize
        x_pooled = tf.ensure_shape(x_pooled, [None, None, channels])
        x_pooled = self.proj(x_pooled)
        x_pooled = self.norm(x_pooled)
        
        return [x_pooled, coords_pooled]


# ========== Full Model ==========

def build_ptv3_serialized_jet_classifier(
    num_particles=150,
    output_dim=5,
    enc_dims=[64, 128, 256],
    enc_layers=[2, 2, 2],
    enc_heads=[4, 8, 8],
    enc_patch_sizes=[50, 25, 12],
    enc_strides=[3, 2],
    cpe_k=3, # Changed default to 3x3 for simpler conv, was 8
    grid_size=0.05,
    use_rpe=False, # Changed default to False, aligning with PTv3 principle
    dropout=0.0,
    aggregation="max",
    serialize_by="morton",
    use_pool=True,
    assume_serialized_input=False,
    coord_dim=2,
    weighted_input=True,
):
    """
    Build the hierarchical PTv3-inspired classifier with serialization.

    Input layout depends on `weighted_input`:
      - True  (jets): [pt, eta, phi]; pt is kept only as a serialization key.
      - False (generic point clouds, e.g. ModelNet): [x, y, z], all channels spatial.
    """
    feature_dim = coord_dim + (1 if weighted_input else 0)

    features_input = layers.Input((num_particles, feature_dim), name="features")

    # Keep the full tuple only for serialization keys; blocks operate on the spatial coords.
    coord_start = 1 if weighted_input else 0
    sort_coords = features_input[..., 0:feature_dim]
    coords = features_input[..., coord_start:coord_start + coord_dim]

    # Initial projection
    x = layers.Dense(enc_dims[0], activation="relu")(features_input)

    # ------------------ START Serialization Injection ------------------
    # Step 1: Serialize the initial point cloud.
    # All subsequent layers (Blocks and Pooling) operate on this sorted sequence.
    # Choose sorting with sort_by: "morton" (default), "pt", or "kt".
    if not assume_serialized_input:
        x, sort_coords = Serialization2D(
            grid_size=grid_size,
            sort_by=serialize_by,
            coord_dim=coord_dim,
            weighted_input=weighted_input,
        )([x, sort_coords])
        coords = sort_coords[..., coord_start:coord_start + coord_dim]
    # ------------------- END Serialization Injection -------------------
    
    # Hierarchical encoder
    for i in range(len(enc_dims)):
        # Transformer blocks
        for _ in range(enc_layers[i]):
            x, coords = PTv3Block(
                d_model=enc_dims[i],
                d_ff=enc_dims[i] * 4,
                num_heads=enc_heads[i],
                patch_size=enc_patch_sizes[i],
                cpe_k=cpe_k,
                grid_size=grid_size,
                dropout=dropout,
                use_rpe=use_rpe,
                coord_dim=coord_dim,
            )([x, coords])
        
        # Downsample (except last stage)
        if i < len(enc_dims) - 1:
            if use_pool:
                # Downsampling uses SerializedPooling2D
                x, coords = SerializedPooling2D(
                    out_dim=enc_dims[i + 1],
                    stride=enc_strides[i]
                )([x, coords])
            else:
                # No pooling: project channels only, keep sequence length and coords
                x = layers.Dense(enc_dims[i + 1])(x)
                x = layers.LayerNormalization(epsilon=1e-6)(x)
            
            # NOTE: In the official PTv3, SerializedPooling calculates the new,
            # coarser serialization codes and implicitly sorts the pooled points.
            # This implementation relies on the pooling operation's local grouping
            # (due to the prior Z-order sort) to maintain the spatial structure.
    
    # Aggregation
    if aggregation == "mean":
        x = tf.reduce_mean(x, axis=1)
    else:
        x = tf.reduce_max(x, axis=1)
    
    # Classifier head
    x = layers.Dense(enc_dims[-1], activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    
    activation = "sigmoid" if output_dim == 1 else "softmax"
    outputs = layers.Dense(output_dim, activation=activation)(x)
    
    return Model(inputs=features_input, outputs=outputs)


# Example usage
if __name__ == "__main__":
    model = build_ptv3_serialized_jet_classifier(
        num_particles=150,
        output_dim=5,
        enc_dims=[64, 128, 256],
        enc_layers=[2, 2, 2],
        enc_heads=[4, 8, 8],
        enc_patch_sizes=[50, 25, 12],
        enc_strides=[3, 2],
        cpe_k=3,
        use_rpe=False,
        dropout=0.1,
        aggregation="max",
        serialize_by="morton"
    )
    
    model.summary()
    
    # Test forward pass
    batch_size = 4
    dummy_input = tf.random.normal([batch_size, 150, 3])
    output = model(dummy_input)
    print(f"\nOutput shape: {output.shape}")
