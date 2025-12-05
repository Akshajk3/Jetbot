from controller import Robot, Camera, RangeFinder
import torch
import numpy as np
import cv2

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# webots
TIME_STEP = 32

MAX_SPEED = 6.28

robot = Robot()

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

camera: Camera = robot.getDevice("camera")
camera.enable(TIME_STEP)

range_finder: RangeFinder = robot.getDevice("range-finder")
range_finder.enable(TIME_STEP)

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

left_motor.setVelocity(0.1 * MAX_SPEED)
right_motor.setVelocity(-0.1 * MAX_SPEED)


# update loop
while robot.step(TIME_STEP) != -1:
    img = camera.getImage()
    