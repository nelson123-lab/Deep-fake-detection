import numpy as np
import cv2
import mss.tools

class MSSscreenCaptue(object):
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = self.getMonitor()
        self.width = self.monitor['width']
        self.height = self.monitor['height']
        self.screenshot_count = 0  # Counter to keep track of screenshot index

    def getMonitor(self, index=1):
        return self.sct.monitors[index]
    
    def processScreenshot(self, screenshot):
        img = np.array(screenshot)
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return imgrgb

    def shot(self, monitor):
        return self.sct.grab(monitor)

    def fullScreenshot(self):
        screenshot = self.sct.grab(self.monitor)
        return self.processScreenshot(screenshot)

    def saveScreenshot(self, filename):
        screenshot = self.fullScreenshot()
        cv2.imwrite(filename, screenshot)
        print(f"Screenshot saved as: {filename}")

    def showScreenshot(self):
        screenshot = self.fullScreenshot()
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
        cv2.imshow('Screenshot', screenshot)
        cv2.waitKey(0)  # Wait indefinitely for a key press to close the window
        cv2.destroyAllWindows()

    def getScreenSize(self):
        return (self.width, self.height)

# Example Usage
# capture = MSSscreenCaptue()
# capture.saveScreenshot('screenshot1.png')  # Save screenshot to file
# capture.showScreenshot()  # Display the screenshot in a window
