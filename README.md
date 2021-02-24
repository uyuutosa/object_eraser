# object_eraser

Just try below scripts to eliminate objects:

```python
import cv2

from object_eraser import ObjectEraser

eraser = ObjectEraser(size=(680,512))
result, mask = eraser.erase("assets/img.png", "assets/mask.png")
cv2.imwrite("results/result.png", result)
```