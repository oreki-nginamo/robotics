import cv2
import numpy as np

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped

def impose_range(xpix, ypix, range=80):
    dist = np.sqrt(xpix**2 + ypix**2)
    return xpix[dist < range], ypix[dist < range]

def find_yellow_rocks(img, rgb_thresh=(110, 110, 50)):
    # Apply a threshold to identify yellow rocks
    yellow_select = np.zeros_like(img[:,:,0])
    yellow_thresh = (img[:,:,0] > rgb_thresh[0]) \
                  & (img[:,:,1] > rgb_thresh[1]) \
                  & (img[:,:,2] < rgb_thresh[2])
    yellow_select[yellow_thresh] = 1
    return yellow_select
    
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    img = Rover.img
    # 1) Define source and destination points for perspective transform
     # Define calibration box in source (actual) and destination (desired) coordinates
        # These source and destination points are defined to warp the image
        # to a grid where each 10x10 pixel square represents 1 square meter
        # The destination box will be 2*dst_size on each side
    dst_size = 5
        # Set a bottom offset to account for the fact that the bottom of the image
        # is not the position of the rover but a bit in front of it
        # this is just a rough guess, feel free to change it!
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset],
                  [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset],
                  ])
    # 2) Apply perspective transform
    warped = perspect_transform(img, source, destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    navigable = color_thresh(warped)
        # ignore half of the image as bad data
    # navigable[0:int(navigable.shape[0]/2), 🙂 = 0

        # Obstacles are simply navigable inverted
    mask = np.ones_like(navigable)
    mask[:,:] = 255
    mask = perspect_transform(mask, source, destination)
    obstacles = np.absolute((np.float32(navigable)-1) * mask)
        # ignore half of the image as bad data
    # obstacles[0:int(obstacles.shape[0]/2),:] = 0

        # identify the rock
    lower_yellow = np.array([24 - 5, 100, 100])
    upper_yellow = np.array([24 + 5, 255, 255])
            # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            # Threshold the HSV image to get only upper_yellow colors
    rock_samples = cv2.inRange(hsv, lower_yellow, upper_yellow)
    rock_samples = perspect_transform(rock_samples, source, destination)

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image

    Rover.vision_image[:,:,0] = obstacles
    Rover.vision_image[:,:,1] = rock_samples
    Rover.vision_image[:,:,2] = navigable
    idx = np.nonzero(Rover.vision_image)
    Rover.vision_image[idx] = 255

    # 5) Convert map image pixel values to rover-centric coords
    xpix_navigable, ypix_navigable = rover_coords(navigable)
    xpix_obstacles, ypix_obstacles = rover_coords(obstacles)
    xpix_rocks, ypix_rocks = rover_coords(rock_samples)

    # 6) Convert rover-centric pixel values to world coordinates
    scale = 10.0
    xpix_navigable, ypix_navigable = impose_range(xpix_navigable, ypix_navigable)
    xpix_obstacles, ypix_obstacles = impose_range(xpix_obstacles, ypix_obstacles)
    navigable_x_world, navigable_y_world = pix_to_world(xpix_navigable, ypix_navigable,
                                                        Rover.pos[0], Rover.pos[1],
                                                        Rover.yaw, Rover.worldmap.shape[0], scale)
    obstacle_x_world, obstacle_y_world = pix_to_world(xpix_obstacles, ypix_obstacles,
                                                      Rover.pos[0], Rover.pos[1],
                                                      Rover.yaw, Rover.worldmap.shape[0], scale)
    rock_x_world, rock_y_world = pix_to_world(xpix_rocks, ypix_rocks,
                                              Rover.pos[0], Rover.pos[1],
                                              Rover.yaw, Rover.worldmap.shape[0], scale)

    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

        # Only update map if pitch an roll are near zero
    if (Rover.pitch < 1 or Rover.pitch > 359) and (Rover.roll < 1 or Rover.roll > 359):
        # increment = 10
        Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] = 255
        Rover.worldmap[rock_y_world, rock_x_world,1] = 255
        Rover.worldmap[navigable_y_world, navigable_x_world, 2] = 255
            # remove overlap mesurements
        nav_pix = Rover.worldmap[:, :, 2] > 0
        Rover.worldmap[nav_pix, 0] = 0
            # clip to avoid overflow
        Rover.worldmap = np.clip(Rover.worldmap, 0, 255)

    # Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles

    dist, angles = to_polar_coords(xpix_navigable, ypix_navigable)
    Rover.nav_dists = dist
    Rover.nav_angles = angles
        # Same for rock samples
    dist, angles = to_polar_coords(xpix_rocks, ypix_rocks)
    Rover.samples_dists = dist
    Rover.samples_angles = angles
    return Rover


