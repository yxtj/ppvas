import cv2

class FrameDiffMask():
    def __init__(self, frame=None, threshold=0.2):
        assert 0<=threshold<=1
        self.th = int(threshold*255)
        self.last = self.get_gray(frame) if frame is not None else None

    def get_gray(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray

    def apply(self, frame, last=None):
        gray = self.get_gray(frame)
        if last is None:
            last = self.last
            if len(last.shape) == 3:
                last = self.get_gray(last)
        frame_diff = cv2.absdiff(gray, self.last)
        _, mask = cv2.threshold(frame_diff, self.th, 255, cv2.THRESH_BINARY)
        self.last = gray
        return mask

    def pick_with_mask(self, frame, mask):
        color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        f = color_mask & frame
        return f

    def pick_color_diff(self, mask, frame1, frame2):
        f1 = self.pick_with_mask(frame1, mask)
        f2 = self.pick_with_mask(frame2, mask)
        f = f2-f1
        return f
