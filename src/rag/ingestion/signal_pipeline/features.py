"""
Digital Signal Processing (DSP) tools for EEG data. Includes filters and feature extraction (power, SNR, SEF)
"""
import numpy as np
import pandas as pd
from scipy import signal
import logging

class SignalFeatures:
    def __init__(self, logger: logging.Logger,
                 fs: int,
                 freq_range: list[float],
                 notch_freq: int | None):
        """
        Calculates filter coefficients in advance.
        """
        self.logger = logger
        self.fs = fs
        self.freq_range = freq_range
        self.notch_freq = notch_freq
        self.bandpass_coeffs, self.notch_coeffs = self._prepare_filters()

    def _prepare_filters(self):
        """Designs the bandpass and notch filter coefficients."""
        nyquist = 0.5 * self.fs
        low_cut = self.freq_range[0] / nyquist
        high_cut = self.freq_range[1] / nyquist

        bandpass_b, bandpass_a = signal.butter(5, [low_cut, high_cut], btype="band")
        notch_b, notch_a = signal.iirnotch(w0=self.notch_freq, Q=30, fs=self.fs)

        return (bandpass_b, bandpass_a), (notch_b, notch_a)

    def _apply_filters(self, data):
        """Applies notch and bandpass filters to a single channel."""
        if np.isnan(data).any():
            nan_ratio = np.isnan(data).mean()
            if nan_ratio < 1.0:
                mean_val = np.nanmean(data)
                data = np.nan_to_num(data, nan=mean_val)
            else:
                data = np.zeros_like(data)
                return data

        # prevent phase shift
        filtered = signal.filtfilt(*self.notch_coeffs, data)
        filtered = signal.filtfilt(*self.bandpass_coeffs, filtered)
        return filtered

    def _calculate_metrics(self, data):
        """Calculates power, SNR, and SEF for a filtered channel."""
        power = float(np.mean(data ** 2))

        # Simple SNR estimation
        # It's not a "real" SNR, it's a proxy.
        kernel = np.ones(20) / 20
        smoothed = np.convolve(data, kernel, mode="same")
        noise = data - smoothed
        noise_std = float(np.std(noise))
        snr = power / (noise_std ** 2) if noise_std > 1e-9 else 0.0

        # Spectral Edge Frequency (SEF) at %95
        freqs = np.fft.rfftfreq(len(data), d=1 / self.fs)
        psd = np.abs(np.fft.rfft(data)) ** 2
        total_power = float(np.sum(psd))

        if total_power < 1e-9:
            return {"power": power, "snr": snr, "sef": 0.0}

        cumsum_power = np.cumsum(psd) / total_power
        sef_95 = (float(freqs[np.where(cumsum_power >= 0.95)[0][0]]) if (cumsum_power >= 0.95).any() else 0)

        return {"power": power, "snr": snr, "sef": sef_95}

    def get_eeg_features(self, df):
        all_channel_metrics = []
        data = df.copy()

        for col in data.columns:
            # Remove the ECG (Heart) channel from the signal analysis
            if col.upper() == "EKG" or not pd.api.types.is_numeric_dtype(data[col]):
                continue

            try:
                filtered = self._apply_filters(data[col].values)
                metrics = self._calculate_metrics(filtered)
                all_channel_metrics.append(metrics)
            except Exception as e:
                self.logger.warning(f"Channel '{col}' failed: {e}")
                continue

        if not all_channel_metrics:
            self.logger.warning("No valid channels processed in segment")
            return {"mean_power": 0.0, "mean_snr": 0.0, "mean_sef": 0.0}

        mean_power = float(np.nanmean([m.get("power", np.nan) for m in all_channel_metrics]))
        mean_snr = float(np.nanmean([m.get("snr", np.nan) for m in all_channel_metrics]))
        mean_sef = float(np.nanmean([m.get("sef", np.nan) for m in all_channel_metrics]))

        mean_power = 0.0 if not np.isfinite(mean_power) else mean_power
        mean_snr = 0.0 if not np.isfinite(mean_snr) else mean_snr
        mean_sef = 0.0 if not np.isfinite(mean_sef) else mean_sef

        return {"mean_power": mean_power, "mean_snr": mean_snr, "mean_sef": mean_sef}
