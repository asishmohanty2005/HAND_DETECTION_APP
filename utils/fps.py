import time


class FPS:

    def __init__(self):
        self.pTime = 0

    def update(self):

        cTime = time.time()

        fps = 1 / (cTime - self.pTime) if self.pTime else 0

        self.pTime = cTime

        return int(fps)