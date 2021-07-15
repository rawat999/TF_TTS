"""Multi-band Melgan Network"""

import numpy as np
import tensorflow as tf
from models import BaseModel
from utils import GroupConv1D, WeightNormalization


def get_initializer(initializer_seed=42):
    """
    Creates a 'tf.initializers.glorot_normal' with the given seed.
    :param initializer_seed: (int) Initializer seed
    :return: GlorotNormal initializer with seed = 'initializer_seed'
    """
    return tf.keras.initializers.GlorotNormal(seed=initializer_seed)


class TFReflectionPadd1d(tf.keras.layers.Layer):
    """Reflection Pad 1D layer"""
    def __init__(self, padding_size, padding_type="REFLECT", **kwargs):
        """
        Initialize Reflection Pad 1D layer
        :param padding_size: interger value for padding
        :param padding_type: string value of padding ("CONSTANT","REFLECT", or "SYMMETRIC")
        Default is "REFLECT"
        :param kwargs: additional parameter for keras Layer class
        """
        super().__init__(**kwargs)
        self.padding_size = padding_size
        self.padding_type = padding_type

    def call(self, x):
        """
        Calculate forward propagation.
        :param x: (Tensor) Input tensor (B, T, C)
        :return: padded tensor (B, T + 2*padding_size, C)
        """
        return tf.pad(x, [[0, 0], [self.padding_size, self.padding_size], [0, 0]], self.padding_type,)


class TFConvTranspose1d(tf.keras.layers.Layer):
    """Tensorflow ConvTranspose1d layer"""
    def __init__(self, filters, kernel_size, strides, padding, is_weight_norm, initializer_seed, **kwargs):
        """
        Initialize TFConvTranspose1d layer
        :param filters: (int) Number of filters
        :param kernel_size: (int) kernel size
        :param strides: (int) stride width
        :param padding: (str) Padding type ("Same" or "Valid")
        :param is_weight_norm: (bool) Whether applying Weight-Normalization or not.
        :param initializer_seed: (int) random seed for initializer
        :param kwargs: additional arguments for keras layer
        """
        super().__int__(**kwargs)
        self.conv1d_transpose = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding="same", kernel_initializer=get_initializer(initializer_seed))

        if is_weight_norm:
            self.conv1d_transpose = WeightNormalization(self.conv1d_transpose) # TODO

    def call(self, x):
        """
        Calculate forward propagation.
        :param x: (Tensor) Input tensor (B, T, C)
        :return:
            Tensor: Output tensor (B, T', C')
        """
        x = tf.expand_dims(x, 2)
        x = self.conv1d_transpose(x)
        x = tf.squeeze(x, 2)
        return x


class TFResidualStack(tf.keras.layers.Layer):
    """Tensorflow ResidualStack Layer"""
    def __init__(self, kernel_size, filters, dilation_rate, use_bias, nonlinear_activation, nonlinear_activation_params, is_weight_norm, initializer_seed, **kwargs):
        """
        Initialize TFResidualStack Layer
        :param kernel_size: (int) kernel_size
        :param filters: (int) Number of filters
        :param dilation_rate: (int) Dilation rate
        :param use_bias: Whether to add bias parameter in convolution layers.
        :param nonlinear_activation: (str) Activation function name
        :param nonlinear_activation_params: (dict) Hyperparameters for activation function.
        :param is_weight_norm: (bool) whether weight normalizetion apply
        :param initializer_seed: (int) seed value for initializer
        :param kwargs: addtional parameter for keras layer
        """
        super()._init__(**kwargs)
        self.blocks = [getattr(tf.keras.layers, nonlinear_activation)(**nonlinear_activation_params),
                       TFReflectionPadd1d((kernel_size-1)//2*dilation_rate),
                       tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, use_bias=use_bias, kernel_initializer=get_initializer(initializer_seed)),
                       getattr(tf.keras.layers, nonlinear_activation)(**nonlinear_activation_params),
                       tf.keras.layers.Conv1D(filters=filters, kernel_size=1, use_bias=use_bias, kernel_initializer=get_initializer(initializer_seed))]

        self.shortcut = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, use_bias=use_bias, kernel_initializer=get_initializer(initializer_seed), name="shortcut")

        # apply weightnorm
        if is_weight_norm:
            self._apply_weightnorm(self.blocks)
            self.shortcut = WeightNormalization(self.shortcut) # TODO

    def call(self, x):
        """
        Calculate forward propagation
        :param x:
        x (Tensor): Input tensor (B, T, C)
        :return:
        Tensor: Output tensor (B, T, C)
        """
        _x = tf.identity(x)
        for layer in self.blocks:
            _x = layer(_x)
        shortcut = self.shortcut(x)
        return shortcut + _x

    def _apply_weightnorm(self, list_layers):
        """Try apply weightnorm for all layer in list_layers."""
        for i in range(len(list_layers)):
            try:
                layer_name = list_layers[i].name.lower()
                if "conv1d" in layer_name or "dense" in layer_name:
                    list_layers[i] = WeightNormalization(list_layers[i])  # TODO

            except Exception:
                pass

def design_prototype_filter(tabs=62, cutoff_ratio=0.15, beta=9.0):
    """
    This function design the prototype of cosine modulated filterbanks
    using Kaiser window approach
    :param tabs(int): The number of filter tabs
    :param cutoff_ratio(float): cut-off frequency ratio
    :param beta(float): Beta coefficient for kaiser window
    :return:
    ndarray: Impulse response of prototype filter (tabs +1,)
    """

    # validation of arguments
    assert tabs % 2 == 0, "The Number of tabs must be even number"
    assert 0.0< cutoff_ratio < 1.0,  "Cutoff ratio must be > 0.0 and < 1.0"

    # initial filter
    wc = np.pi * cutoff_ratio
    with np.errstate(invalid="ignore"):
        h_i = np.sin(wc * (np.arange(tabs+1) - 0.5*tabs)) / (np.pi * (np.arange(tabs+1) - 0.5 * tabs))

    # fix NaN due to indeterminate form
    h_i[tabs // 2] = np.cos(0) * cutoff_ratio

    # apply kaiser window
    w = np.kaiser(tabs + 1, beta)
    h = h_i * w
    return h


class TFPQMF(tf.keras.layers.Layer):
    """Pseudo-Quadrature Mirror Filters (PQMF) using tensorflow"""
    def __init__(self, config, **kwargs):
        """
        Initialize PQMF layer
        :param config(class): MultiBandMelGeneratorConfig
        :param kwargs: addition parameters for layer class
        """
        super().__init__(**kwargs)
        subbands = config.subbands
        taps = config.taps
        cutoff_ratio = config.cutoff_ratio
        beta = config.beta

        # define filter coefficient
        h_proto = design_prototype_filter(taps, cutoff_ratio, beta)  # (63,)
        h_analysis = np.zeros((subbands, len(h_proto)))  # 10, 63
        h_synthesis = np.zeros((subbands, len(h_proto)))  # 10, 63
        for k in range(subbands):
            h_analysis[k] = (2 * h_proto * np.cos((2 * k + 1) * (np.pi / (2 * subbands)) * (np.arange(taps + 1) - (taps/2))
                                                  + (-1) ** k * np.pi / 4))
            h_synthesis[k] = (2 * h_proto * np.cos((2 * k + 1) * (np.pi / (2 * subbands)) * (np.arange(taps + 1) - (taps/2))
                                                   - (-1) ** k * np.pi / 4))

        # [subbands, 1, taps + 1] == [filter_width, in_channels, out_channels]
        analysis_filter = np.expand_dims(h_analysis, 1)  # 10,1,63
        analysis_filter = np.transpose(analysis_filter, (2, 1, 0))  # 63, 1, 10

        synthesis_filter = np.expand_dims(h_synthesis, 0)  # 1, 10, 63
        synthesis_filter = np.transpose(synthesis_filter, (2, 1, 0))  # 63, 10, 1

        # filter for downsampling & upsampling
        updown_filter = np.zeros((subbands, subbands, subbands), dtype=np.float32)
        for k in range(subbands):
            updown_filter[0, k, k] = 1.0

        self.subbands = subbands  # 10
        self.taps = taps  # 62
        self.analysis_filter = analysis_filter.astype(np.float32)  # 63,1,10
        self.synthesis_filter = synthesis_filter.astype(np.float32)  # 63, 10, 1
        self.updown_filter = updown_filter.astype(np.float32)  # 10,10,10

    @tf.function(experimental_relax_shapes=True,input_signature=[tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32)],)
    def analysis(self, x):
        """Analysis with PQMF.
        Args:
            x (Tensor): Input tensor (B, T, 1).
        Returns:
            Tensor: Output tensor (B, T // subbands, subbands).
        """
        x = tf.pad(x, [[0, 0], [self.taps // 2, self.taps // 2], [0, 0]])  # (B, T + tabs, 1)
        x = tf.nn.conv1d(x, self.analysis_filter, stride=1, padding="VALID")
        x = tf.nn.conv1d(x, self.updown_filter, stride=self.subbands, padding="VALID")
        return x

    @tf.function(experimental_relax_shapes=True,input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)],)
    def synthesis(self, x):
        """Synthesis with PQMF.
        Args:
            x (Tensor): Input tensor (B, T // subbands, subbands).
        Returns:
            Tensor: Output tensor (B, T, 1).
        """
        x = tf.nn.conv1d_transpose(
            x,
            self.updown_filter * self.subbands,
            strides=self.subbands,
            output_shape=(
                tf.shape(x)[0],
                tf.shape(x)[1] * self.subbands,
                self.subbands,
            ),
        )
        x = tf.pad(x, [[0, 0], [self.taps // 2, self.taps // 2], [0, 0]])
        return tf.nn.conv1d(x, self.synthesis_filter, stride=1, padding="VALID")



class TFMBMelGANGenerator(BaseModel):
    def __init__(self, config, **kwargs):
        """
        Initialize TFMBMelGANGenerator model
        :param config: config object of MBMelgan generator
        :param kwargs: additional arguments for BaseModel
        """
        super.__init__(**kwargs)

        # check hyper parameter is valid or not
        assert config.filters >= np.prod(config.upsample_scales)
        assert config.filters % (2 ** len(config.upsample_scales)) == 0

        # add initial layer
        layers = []
        layers += [TFReflectionPadd1d((config.kernel_size - 1)//2, padding_type=config.padding_type, name="first_reflect_padding"),
                   tf.keras.layers.Conv1D(filters=config.filters, kernel_size=config.kernel_size,
                                          use_bias=config.use_bias, kernel_initializer=get_initializer(config.initializer_seed))]

        # add upsampling layers
        for i, upsample_scale in enumerate(config.upsample_scales):
            layers += [getattr(tf.keras.layers, config.nonlinear_activation)(**config.nonlinear_activation_params),
                       TFConvTranspose1d(filters=config.filters//(2 ** (i + 1)), kernel_size=upsample_scale * 2, strides=upsample_scale, padding="same", is_weight_norm=config.is_weight_norm, initializer_seed=config.initializer_seed, name="conv_transpose_._{}".format(i))]

        # add residual stack layer
        for j in range(config.stacks):
            layers += [TFResidualStack(kernel_size=config.stack_kernel_size,
                                       filters=config.filters//(2 ** (i + 1)),
                                       dilation_rate=config.stack_kernel_size**j,
                                       use_bias=config.use_bias,
                                       nonlinear_activation=config.nonlinear_activation,
                                       nonlinear_activation_params=config.nonlinear_activation_params,
                                       is_weight_norm=config.is_weight_norm,
                                       initializer_seed=config.initializer_seed,
                                       name="residual_stack_._{}._._{}".format(i, j))]

        # add final layer
        layers += [getattr(tf.keras.layers, config.nonlinear_activation)(**config.nonlinear_activation_params),
                   TFReflectionPadd1d((config.kernel_size-1)//2,
                                      padding_type=config.padding_type,
                                      name="last_reflect_padding"),
                   tf.keras.layers.Conv1D(filters=config.out_channels,
                                          kernel_size=config.out_channels,
                                          use_bias=config.use_bias,
                                          kernel_initializer=get_initializer(config.initializer_seed),
                                          dtype=tf.float32,
                                          ),
                   ]

        if config.use_final_nonlinear_activation:
            layers += [tf.keras.layers.Activation("tanh", dtype=tf.float32)]

        if config.is_weight_norm is True:
            self._apply_weightnorm(layers)

        # combine whole layers in Sequential API
        self.melgan = tf.keras.models.Sequential(layers)

        # object for Pseudo-Quadrature Mirror Filters (PQMF)
        self.pqmf = TFPQMF(config=config, dtype=tf.float32, name="pqmf")

    def call(self, mels, **kwargs):
        """
        Calculate forward propagation.
        :param mels: (Tensor) (B, T, channels)
        :param kwargs: additional args
        :return: Tensor: Output tensor (B, T ** prod(upsample_scales), out_channels)
        """
        return self.inference(mels)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 80], dtype=tf.float32, name="mels")])
    def inference(self, mels):
        mb_audios = self.melgan(mels)
        return self.pqmf.synthesis(mb_audios)

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, None, 80], dtype=tf.float32, name="mels")])
    def inference_tflite(self, mels):
        mb_audios = self.melgan(mels)
        return self.pqmf.synthesis(mb_audios)

    def _apply_weightnorm(self, list_layers):
        """Try apply weightnorm for all layer in list_layers"""
        for i in range(len(list_layers)):
            try:
                layer_name = list_layers[i].name.lower()
                if "conv1d" in layer_name or "dense" in layer_name:
                    list_layers[i] = WeightNormalization(list_layers[i])
            except Exception:
                pass


    def _build(self):
        """Build model by passing fake input."""
        fake_mels = tf.random.uniform(shape=[1, 100, 80], dtype=tf.float32)
        self(fake_mels)


class TFMBMelGANDiscriminator(tf.keras.layers.Layer):
    """Tensorflow MBMelGAN discriminator layer"""
    def __init__(self, out_channels=1, kernel_size=[5,3], filters=16, max_downsample_filters=1024,
                 use_bias=True, downsample_scales=[4, 4, 4, 4], nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"alpha": 0.2}, padding_type="REFLECT", is_weight_norm=True,
                 initializer_seed=0.02, **kwargs):
        """
        Initialize MBMelGAN Discriminator Layer
        :param out_channels: (int) Number of output channels.
        :param kernel_size: (list) List of two kernel sizes. The prod will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
                For example if kernel_sizes = [5, 3], the first layer kernel size will be 5 * 3 = 15.
                the last two layers' kernel size will be 5 and 3, respectively.
        :param filters: (int) Initial number of filters for conv layer.
        :param max_downsample_filters: (int) Maximum number of filters for downsampling layers
        :param use_bias: (bool) Whether to add bias parameter in convolution layers.
        :param downsample_scales: (list) List of downsampling scales.
        :param nonlinear_activation: (str) Activation function module name.
        :param nonlinear_activation_params: (dict) Hyperparameters for activation function.
        :param padding_type: (str) Padding type (support only "REFLECT", "CONSTANT", "SYMMETRIC")
        :param is_weight_norm: (bool) whether to apply weight normalization
        :param initializer_seed: (int) seed value for initializer
        """
        super().__init__(**kwargs)
        discriminator = []

        # check kernel_size is valid
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        assert kernel_size[1] % 2 == 1

        # add first layer
        discriminator += [TFReflectionPadd1d((np.prod(kernel_size) - 1) // 2, padding_type=padding_type),
                          tf.keras.layers.Conv1D(filters=filters, kernel_size=int(np.prod(kernel_size)),
                                                 use_bias=use_bias,
                                                 kernel_initializer=get_initializer(initializer_seed)),
                          getattr(tf.keras.layers, nonlinear_activation)(**nonlinear_activation_params)]


        # add downsample layers
        in_chs = filters
        with tf.keras.utils.CustomObjectScope({"GroupConv1D": GroupConv1D}):
            for downsample_scale in downsample_scales:
                out_chs = min(in_chs * downsample_scale, max_downsample_filters)
                discriminator += [GroupConv1D(filters=out_chs, kernel_size=downsample_scale * 10 +1, strides=downsample_scale, padding="same", use_bias=use_bias, groups=in_chs//4, kernel_initializer=get_initializer(initializer_seed))]
                discriminator += [getattr(tf.keras.layers, nonlinear_activation)(**nonlinear_activation_params)]
                in_chs = out_chs


        # add final layers
        out_chs = min(in_chs * 2, max_downsample_filters)
        discriminator += [tf.keras.layers.Conv1D(filters=out_chs, kernel_size=kernel_size[0],padding="same",use_bias=use_bias, kernel_initializer=get_initializer(initializer_seed))]
        discriminator += [getattr(tf.keras.layers, nonlinear_activation)(**nonlinear_activation_params)]
        discriminator += [tf.keras.layers.Conv1D(filters=out_channels, kernel_size=kernel_size[1], padding="same", use_bias=use_bias, kernel_initializer=get_initializer(initializer_seed))]

        if is_weight_norm:
            self._apply_weightnorm(discriminator)

        self.discriminator = discriminator

    def call(self, x, **kwargs):
        """
        Calculate forward propagation
        :param x: (Tensor) Input noise signal (B, T, 1).
        :param kwargs: additional parameters for keras layer calling method
        :return: (list) List of output tensors of each layer.
        """
        outs = []
        for f in self.discriminator:
            x = f(x)
            outs += [x]
        return outs

    def _apply_weightnorm(self, list_layers):
        """Try apply weightnorm for all layer in list_layers."""
        for i in range(len(list_layers)):
            try:
                layer_name = list_layers[i].name.lower()
                if "conv1d" in layer_name or "dense" in layer_name:
                    list_layers[i] = WeightNormalization(list_layers[i]) #TODO
            except Exception:
                pass


class TFMBMelGANMultiScaleDiscriminator(BaseModel):
    """MB MelGAN multi-scale discriminator model"""
    def __init__(self, config, **kwargs):
        """
        Initialize MultiBand MelGAN multi-scale discriminator model.
        :param config: config object for MB MelGAN discriminator
        :param kwargs: additional parameters
        """
        super().__init__(**kwargs)
        self.discriminator = []

        # add discriminator
        for i in range(config.scales):
            self.discriminator += [TFMBMelGANDiscriminator(out_channels=config.out_channels,
                                                           kernel_size=config.kernel_sizes,
                                                           filters=config.filters,
                                                           max_downsample_filters=config.max_downsample_filters,
                                                           use_bias=config.use_bias,
                                                           downsample_scales=config.downsample_scales,
                                                           nonlinear_activation=config.nonlinear_activation,
                                                           nonlinear_activation_params=config.nonlinear_activation_params,
                                                           padding_type=config.padding_type,
                                                           is_weight_norm=config.is_weight_norm,
                                                           initializer_seed=config.initializer_seed,
                                                           name="mbmelgan_discriminator_scale_._{}".format(i))]
            self.pooling = getattr(tf.keras.layers, config.downsample_pooling)(**config.downsample_pooling_params)

    def call(self, x, **kwargs):
        """
        Calculate forward propagation.
        :param x: (Tensor) Input noise signal (B, T, 1).
        :param kwargs: additional parameters
        :return: (list) List of list of each discriminator outputs,
        which consists of each layer outputs tensors.
        """
        outs = []
        for f in self.discriminator:
            outs += [f(x)]
            x = self.pooling(x)
        return outs
