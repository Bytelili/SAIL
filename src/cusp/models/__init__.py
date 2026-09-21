"""Trainable modules implementing Counterfactual User-State Posterior Memory."""

from .cusp_model import CUSPModel, CUSPOutput
from .posterior import DiagonalGaussianPosterior, PosteriorState

__all__ = ["CUSPModel", "CUSPOutput", "DiagonalGaussianPosterior", "PosteriorState"]
