# Social Sports — Sports Player Tracking Project

A computer vision and machine learning project designed to detect, identify, and track football players from real-world match footage, with a focus on amateur and small-sided football games.

The project combines **supervised deep learning, multi-object tracking, appearance-based player identification, and geometric field calibration** to transform raw football videos into structured positional data for sports analytics.

## Overview

Social Sports aims to automatically analyze football matches recorded using fixed or multi-camera setups.

The main challenge is not only detecting players in individual frames, but also **maintaining consistent player identities throughout an entire match**, even when players overlap, disappear from view, or are temporarily missed by the detection model.

The system is designed to handle real-world challenges such as low-resolution footage, motion blur, player occlusion, varying illumination, and visually similar team uniforms.

## Key Features

- **Player Detection:** YOLO-based object detection with supervised fine-tuning experiments using labeled football footage.
- **Multi-Object Tracking:** Integration and evaluation of BoT-SORT, ByteTrack, and a custom appearance-aware tracking approach.
- **Player Identification & Re-ID:** Appearance-based identity matching to maintain consistent player IDs and recover identities after temporary occlusions.
- **Custom Stable-ID Tracking:** An `AppearanceStableTracker` implementation with track recovery, spatial matching, and configurable player-count constraints.
- **Field Calibration:** Homography-based transformation of image coordinates into a 2D football pitch representation, including lens distortion correction.
- **Ball Detection:** Experimental ball detection and tracking pipeline, developed separately to address the challenges of small-object detection.
- **Data Export & Visualization:** Exporting detection and tracking information to CSV/XLSX files for debugging


<img width="641" height="355" alt="image" src="https://github.com/user-attachments/assets/190e7165-23b7-484d-9614-14902f858f5c" />
