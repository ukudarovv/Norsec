"""Аналитика Phase 2: социальные сигналы, позы, подавление FP."""

from analytics.social_signals import SOCIAL_SIGNALS, SocialSignal, detect_social_signals
from analytics.pose_signals import POSE_SIGNALS, PoseSignal, detect_pose_signals
from analytics import suppression

__all__ = [
    "SOCIAL_SIGNALS",
    "SocialSignal",
    "detect_social_signals",
    "POSE_SIGNALS",
    "PoseSignal",
    "detect_pose_signals",
    "suppression",
]
