FROM jupyter/minimal-notebook:latest

// TODO: freeze versions?
RUN pip install torch numpy pillow opencv-python

// TODO: verify jupyter notebook reachable if needed for debugging / testing / experimenting

RUN git clone https://github.com/Pongpisit-Thanasutives/Variations-of-SFANet-for-Crowd-Counting.git
