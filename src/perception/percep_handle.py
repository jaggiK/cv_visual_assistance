class PercepHandle():
    def clear(self):
        self._rgb = None
        self._left_rgb = None
        self._right_rgb = None
        self._gui_img = None

    def __init__(self):
        self.clear()
        self.frame_num = 0


