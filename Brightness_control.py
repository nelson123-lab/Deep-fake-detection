"""
- Each frame comes in is checked to see if the frame is bright or not.
- If the frame is not bright, then the frame is brightened.
- If the frame is already bright, then just the dark part of the frame will be brightened.
"""
import cv2
import numpy as np

from PIL import Image

def increase_brightness_for_dark(image_path, brightness_factor = 30):
    try:
        # Open the image
        im = Image.open(image_path)

        # Convert to HSV colourspace and split channels
        H, S, V = im.convert('HSV').split()

        # Increase the brightness (Value channel)
        newV = V.point(lambda i: i + int(brightness_factor * (255 - i) / 255))

        # Recombine channels and convert back to RGB
        result_image = Image.merge(mode="HSV", bands=(H, S, newV)).convert('RGB')

        # Generate output file path
        output_path = f"result_{brightness_factor}.jpg"

        # Save the result image
        result_image.save(output_path)

        return output_path

    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def adjust_brightness(image, brightness_factor):
    # Apply brightness adjustment
    adjusted_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    return adjusted_image

def classify_brightness(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate mean brightness as percentage
    mean_percent = np.mean(gray_image) * 100 / 255
    
    # Classify as "dark" or "light"
    classification = "dark" if mean_percent < 50 else "light"
    
    return classification, mean_percent

def main():
    filename = 'test_data/Real_frames/0real_0.jpg'
    
    # Load image
    im = cv2.imread(filename)
    
    if im is None:
        print(f'ERROR: Unable to load {filename}')
        return
    
    # Classify brightness
    brightness_class, mean_brightness_percent = classify_brightness(im)
    print(f'Image is {brightness_class} ({mean_brightness_percent:.1f}%)')
    
    # Adjust brightness if image is classified as "dark"
    if brightness_class == "dark":
        # Increase brightness by scaling factor
        brightness_factor = 2  # Adjust this value based on preference
        adjusted_image = adjust_brightness(im, brightness_factor)
        
        # Display or save the adjusted image
        cv2.imshow('Adjusted Image', adjusted_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save the adjusted image
        output_filename = 'adjusted_image.jpg'
        cv2.imwrite(output_filename, adjusted_image)
        increase_brightness_for_dark(output_filename, brightness_factor = 30)
    else:
        print('Image is already sufficiently bright.')
        increase_brightness_for_dark(filename)

if __name__ == "__main__":
    main()

