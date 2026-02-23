# VLM-X23-A-Multimodal-Intelligent-Vision-System-Powered-by-Moondream2

A real-time AI surveillance and scene-understanding system powered by Moondream2 Vision-Language Model, classical CV detectors, and intelligent event scoring.

# Overview

VLM-X23 is a real-time multimodal intelligent monitoring system that combines:

1.Vision-Language reasoning

2.Crowd detection

3.Theft probability scoring

4.Loitering detection

5.Hazard analysis

6.OCR

7.Behavior analysis

8.Voice narration

9.Alert system with sound feedback

It uses the lightweight Moondream2 VLM for semantic understanding and integrates classical computer vision techniques for efficient real-time performance.

# Architecture

Webcam Input

      ↓
Frame Preprocessing

      ↓
Moondream2 Vision Encoder

      ↓
Multimodal Reasoning Engine

      ↓
Event Logic & Risk Scoring

      ↓
Alert System (Sound + HUD + Voice)

      ↓ 
Session Logging + Evidence Capture

# Core Features

# 1.Scene Intelligence

*Scene description

*Emotion analysis

*Behavior understanding

*Interaction analysis

*Posture detection

*Cleanliness assessment

*Surface hazard detection

*PPE / safety monitoring

*OCR text reading

# 2.Security Monitoring

*Theft detection (multi-frame scoring system)

*Crowd detection

*Loitering detection

*Smart object disappearance logic

*Motion-triggered AI inference

*Evidence image capture

# 3.Smart Alert System

*3-beep alarm system

*Cooldown logic to prevent alert spamming

*Voice narration (pyttsx3)

*HUD alert overlay

# Model Details

# 1)Vision-Language Model

*Model: Moondream2

*Model ID: vikhyatk/moondream2

*Revision: 2024-08-26

*Framework:Transformers

*Device: CUDA (if available) / CPU fallback

*Image resize optimization: 378 × 378 for faster inference

# 2)Detection Methods

*HOG Person Detector (OpenCV)

*Haar Cascade Face Detector

*VLM-based object detection fallback

*Custom tracker for ID-based movement tracking

# System Design

The system follows a modular, multi-threaded architecture:

*Main Webcam Loop

*VLM Worker Thread

*AI Auto-Monitor Thread

*Sound Alert Thread

*Voice Narration Thread

*Task Queue + Result Queue

*Session-based task isolation

This prevents UI freezing and enables asynchronous AI reasoning.

# Theft Scoring Logic

Theft detection is not based on a single frame.

It calculates a dynamic score based on:

*Repeated suspicious actions

*Object disappearance (e.g., phone missing)

*Aggressive interaction keywords

*Exit count > Entry count

*Crowd presence

*Multi-frame voting mechanism

If score ≥ 0.30 → Theft Alert Triggered.

# Controls

# | Key | Function                |

| --- | ----------------------- |

| c   | Describe scene          |

| e   | Emotion analysis        |

| a   | Activity analysis       |

| b   | Behavior                |

| l   | Cleanliness check       |

| t   | OCR text                |

| h   | Hazard detection        |

| y   | Safety check            |

| w   | Surface hazards         |

| p   | Posture                 |

| j   | Interaction             |

| x   | Theft detection         |

| o   | Loitering detection     |

| z   | Crowd monitoring toggle |

| s   | Voice toggle            |

| n   | Narrator mode           |

| m   | Motion sensor           |

| r   | Reset HUD               |

| q   | Quit                    |

# Hardware & Resource Usage

*RAM Usage: 8–12 GB recommended

*Processing Speed: ~1.5–3 sec per AI inference (CPU), faster with CUDA

*Storage Required: ~3–5 GB (model + dependencies)

*Camera: 720p / 1080p webcam

# Required Packages

pip install torch torchvision transformers opencv-python pillow numpy pyttsx3

Windows only:

winsound (built-in)

# Multi-Threading Strategy

*Dedicated AI processing thread

*Separate sound loop

*Dedicated voice engine thread

*Task queue to prevent overlapping VLM calls

*Session ID system to discard outdated tasks

# Safety Mechanisms

*Alert cooldown timers

*Session-based result filtering

*Multi-vote confirmation for theft

*Motion-based AI throttling

*Auto reset after alert duration

# How to Run

python m14(frame).py

Press keys to trigger intelligent monitoring modes.

# Research Significance

VLM-X23 demonstrates:

*Lightweight Vision-Language deployment

*Hybrid AI + classical CV architecture

*Real-time multimodal reasoning

*Practical security AI prototype

*Edge-device capable AI surveillance

# Example Outputs

*THEFT DETECTED (score=0.45)

*Crowd Detected

*LOITERING DETECTED

*Status: Hazard - Knife

*No visible text

*Clean and organized environment

# Future Improvements

*Replace HOG with YOLOv8 for better detection

*Integrate face recognition module

*Deploy as Flask / FastAPI web system

*Add database logging instead of text file

*Edge device optimization (Jetson Nano)
