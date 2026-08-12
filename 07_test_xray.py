from PIL import Image
import matplotlib.pyplot as plt

image_path = "xray1.jpg"

image = Image.open(image_path)

print("Image size:", image.size)
print("Image mode:", image.mode)

plt.imshow(image, cmap="gray")
plt.axis("off")
plt.title("Test Chest X-ray")
plt.show()