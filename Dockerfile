FROM anibali/pytorch:1.8.1-cuda11.1

# TODO: freeze versions?
RUN pip install pillow opencv-python jupyter notebook

# TODO: verify jupyter notebook reachable if needed for debugging / testing / experimenting

RUN git clone https://github.com/Pongpisit-Thanasutives/Variations-of-SFANet-for-Crowd-Counting.git
