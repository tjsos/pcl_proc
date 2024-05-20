#!/usr/bin/env python3

#Author: Tony Jacob
#Part of RISE Project. 
#Utility functions for path_gen.py
#tony.jacob@uri.edu

import numpy as np
import cv2
import math

##Viz functions.
def compare_two_lists(list1, list2, height, width):
    """
    Viz function to plot the original canny and extracted edge

    
    Args:
        list1: List of points (raw_pixels)
        list2: List of points

    Returns:
        image: Image with points plotted.
    """
    image = np.zeros((height, width), dtype=np.uint8)
    # Iterate through the list of coordinates and draw circles
    
    for coordinates in list1:
        center = tuple(coordinates)
        cv2.circle(image, center, 1, 100)

    for coordinates in list2:
        center = tuple(coordinates)
        cv2.circle(image, center, 1, 255)
    
    return image

def compare_points_with_image(frame, points_list):
    """
    Viz function to plot list of points in existing image

    Args:
        frame: Image Array
        points_list: list of points

    Returns:
        frame: Image Array with points_list vizualised
    """
    for coordinates in points_list:
        center = tuple(coordinates)
        cv2.circle(frame, center, 1, 100, 1)
    return frame

#Rotation Matrix
def rotate_points(points, angle_radians):
    """
    Rotation Matrix
    """

    # Define the rotation matrix
    rotation_matrix = [
        [math.cos(angle_radians), math.sin(angle_radians)],
        [-math.sin(angle_radians), math.cos(angle_radians)]
    ]
    
    # Initialize a list to store rotated points
    x_list = []
    y_list = []

    
    # Apply rotation to each point
    for point in points:
        # Perform matrix multiplication to rotate the point
        rotated_x = rotation_matrix[0][0] * point[0] + rotation_matrix[0][1] * point[1]
        rotated_y = rotation_matrix[1][0] * point[0] + rotation_matrix[1][1] * point[1]
        
        # Append rotated point to the list
        x_list.append((rotated_x))
        y_list.append((rotated_y))
    
    return x_list, y_list

def find_cordinates_of_max_value(image:np.array):
    """
    Find cordinates of white pixels

    Args:
        image: Image after Canny edge detection.
    
    Returns:
        max_intensity_coordinates: [[x,y]] of edges.
    """
    max_intensity = np.max(image)
    # Get the coordinates of pixels with the maximum intensity
    max_intensity_coordinates = np.column_stack(np.where(image == max_intensity))
    # column_stack GIVES INVERTED [x,y] from that of image frame
    max_intensity_coordinates = [[y,x] for x,y in max_intensity_coordinates]
    return max_intensity_coordinates

#Fitting functions
def model_f(x, a, b, c):
    """
    The polynomial used to fit the points.
    """
    return a*x**2 + b* x**1 +c

def linear_regression_model(x, a, b):
    """
    Linear regression
    """
    return a*x + b

def calculate_slope(x_coords, y_coords):
    # Ensure there are at least two points
    if len(x_coords) < 2 or len(y_coords) < 2:
        raise ValueError("At least two points are required to calculate a slope.")
    
    if len(x_coords) != len(y_coords):
        raise ValueError("The number of x and y coordinates must be the same.")
    
    n = len(x_coords)
    sum_x = sum(x_coords)
    sum_y = sum(y_coords)
    sum_xy = sum(x * y for x, y in zip(x_coords, y_coords))
    sum_x_squared = sum(x ** 2 for x in x_coords)
    
    # Calculate the slope using the least squares formula
    numerator = n * sum_xy - sum_x * sum_y
    denominator = n * sum_x_squared - sum_x ** 2
    
    if denominator == 0:
        raise ValueError("Denominator is zero, cannot calculate slope. Check the input points.")
    
    slope = numerator / denominator
    
    return slope
