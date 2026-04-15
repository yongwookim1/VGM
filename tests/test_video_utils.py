import unittest

from PIL import Image

from src.data.video_utils import resize_frame, select_representative_frame


class VideoUtilsTest(unittest.TestCase):
    def test_resize_frame(self):
        image = Image.new("RGB", (1000, 1000), "white")
        resized = resize_frame(image, max_pixels=100 * 100)
        self.assertLessEqual(resized.size[0] * resized.size[1], 100 * 100)

    def test_select_representative_frame(self):
        frames = [Image.new("RGB", (8, 8), color) for color in ("red", "green", "blue")]
        middle = select_representative_frame(frames)
        self.assertEqual(middle.getpixel((0, 0)), (0, 128, 0))


if __name__ == "__main__":
    unittest.main()
