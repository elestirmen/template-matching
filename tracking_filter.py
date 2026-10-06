# -*- coding: utf-8 -*-
"""Lokalizasyon icin test edilebilir zamansal takip filtreleri."""

from __future__ import annotations

from typing import Tuple

import numpy as np


class ConstantVelocityKalmanFilter:
    """Durumu ``[x, y, vx, vy]`` olan sabit-hiz Kalman filtresi.

    ``motion_x``/``motion_y`` bagimsiz sensor veya kontrol sisteminden gelen,
    ilgili kare araligindaki bilinen piksel deplasmanidir. Filtrenin ogrendigi
    hiz bu girdinin yerine gecmez; kalan hareketi modeller.
    """

    def __init__(
        self,
        initial_position: Tuple[int, int],
        process_noise: float = 50.0,
        measurement_noise: float = 80.0,
    ) -> None:
        self._measurement_variance = max(1e-6, float(measurement_noise) ** 2)
        self._process_variance = max(1e-6, float(process_noise) ** 2)
        self._state = np.array(
            [float(initial_position[0]), float(initial_position[1]), 0.0, 0.0],
            dtype=np.float64,
        )
        self._covariance = np.diag(
            [
                self._measurement_variance,
                self._measurement_variance,
                self._process_variance,
                self._process_variance,
            ]
        )

    def predict(
        self,
        motion_x: float = 0.0,
        motion_y: float = 0.0,
        process_var=None,
        dt: float = 1.0,
    ) -> None:
        dt = max(1e-3, float(dt))
        transition = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self._state = transition @ self._state
        self._state[0] += float(motion_x)
        self._state[1] += float(motion_y)

        q = self._process_variance if process_var is None else max(1e-9, float(process_var))
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        process_covariance = q * np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=np.float64,
        )
        self._covariance = (
            transition @ self._covariance @ transition.T + process_covariance
        )

    def update(
        self,
        measured_x: float,
        measured_y: float,
        confidence: float = 1.0,
        gain_max: float = 1.0,
    ) -> None:
        observation = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        confidence = max(0.01, min(1.0, float(confidence)))
        measurement_covariance = np.eye(2, dtype=np.float64) * (
            self._measurement_variance / confidence
        )
        measurement = np.array([float(measured_x), float(measured_y)], dtype=np.float64)
        innovation = measurement - observation @ self._state
        innovation_covariance = (
            observation @ self._covariance @ observation.T + measurement_covariance
        )
        gain = self._covariance @ observation.T @ np.linalg.inv(innovation_covariance)

        # Eski filtrenin kazanc tavani davranisini koru. Tum kazanc matrisini
        # ayni oranla olceklemek hiz durumunun konumdan kopmasini engeller.
        cap = max(0.0, min(1.0, float(gain_max)))
        positional_gain = max(abs(float(gain[0, 0])), abs(float(gain[1, 1])))
        if positional_gain > cap and positional_gain > 0.0:
            gain *= cap / positional_gain

        self._state = self._state + gain @ innovation
        identity = np.eye(4, dtype=np.float64)
        residual = identity - gain @ observation
        self._covariance = (
            residual @ self._covariance @ residual.T
            + gain @ measurement_covariance @ gain.T
        )
        self._covariance = 0.5 * (self._covariance + self._covariance.T)

    def reset(self, initial_position: Tuple[int, int]) -> None:
        self._state = np.array(
            [float(initial_position[0]), float(initial_position[1]), 0.0, 0.0],
            dtype=np.float64,
        )
        self._covariance = np.diag(
            [
                self._measurement_variance,
                self._measurement_variance,
                self._process_variance,
                self._process_variance,
            ]
        )

    @property
    def position(self) -> Tuple[int, int]:
        return (int(round(self._state[0])), int(round(self._state[1])))

    def position_f(self) -> Tuple[float, float]:
        return (float(self._state[0]), float(self._state[1]))

    @property
    def velocity(self) -> Tuple[float, float]:
        return (float(self._state[2]), float(self._state[3]))

    @property
    def uncertainty_px(self) -> float:
        return float(np.sqrt(max(0.0, self.position_var_mean)))

    @property
    def position_var_mean(self) -> float:
        return max(
            0.0,
            float((self._covariance[0, 0] + self._covariance[1, 1]) / 2.0),
        )

    @property
    def measurement_var(self) -> float:
        return float(self._measurement_variance)
