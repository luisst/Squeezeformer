def tf_normalize_signal(signal: tf.Tensor) -> tf.Tensor:
    """
    TF Normailize signal to [-1, 1] range
    Args:
        signal: tf.Tensor with shape [None]

    Returns:
        normalized signal with shape [None]
    """
    gain = 1.0 / (tf.reduce_max(tf.abs(signal), axis=-1) + 1e-9)
    return signal * gain


def tf_preemphasis(signal: tf.Tensor, coeff=0.97):
    """
    TF Pre-emphasis
    Args:
        signal: tf.Tensor with shape [None]
        coeff: Float that indicates the preemphasis coefficient

    Returns:
        pre-emphasized signal with shape [None]
    """
    if not coeff or coeff <= 0.0: return signal
    s0 = tf.expand_dims(signal[0], axis=-1)
    s1 = signal[1:] - coeff * signal[:-1]
    return tf.concat([s0, s1], axis=-1)


def compute_log_mel_spectrogram(self, signal):
    spectrogram = self.stft(signal)
    linear_to_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=self.num_feature_bins,
        num_spectrogram_bins=spectrogram.shape[-1],
        sample_rate=self.sample_rate,
        lower_edge_hertz=0.0, upper_edge_hertz=(self.sample_rate / 2)
    )
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_weight_matrix, 1)
    return self.power_to_db(mel_spectrogram)


def tf_normalize_audio_features(audio_feature: tf.Tensor, per_frame=False) -> tf.Tensor:
    """
    TF Mean and variance features normalization
    Args:
        audio_feature: tf.Tensor with shape [T, F]

    Returns:
        normalized audio features with shape [T, F]
    """
    axis = 1 if per_frame else None
    mean = tf.reduce_mean(audio_feature, axis=axis, keepdims=True)
    std_dev = tf.math.sqrt(tf.math.reduce_variance(audio_feature, axis=axis, keepdims=True) + 1e-9)
    return (audio_feature - mean) / std_dev



if self.normalize_signal:
    signal = tf_normalize_signal(signal)
signal = tf_preemphasis(signal, self.preemphasis)

if self.feature_type == "log_mel_spectrogram":
    features = self.compute_log_mel_spectrogram(signal)
else:
    raise ValueError("feature_type must be 'log_mel_spectrogram'")

features = tf.expand_dims(features, axis=-1)

if self.normalize_feature:
    features = tf_normalize_audio_features(features, per_frame=self.normalize_per_frame)

return features

