import os
from PIL import Image, ImageDraw, ImageFont

# Create directory if it doesn't exist
os.makedirs('static/images/tumors', exist_ok=True)

# Create a new white image
img = Image.new('RGB', (200, 200), 'white')
draw = ImageDraw.Draw(img)

# Draw a gray circle
draw.ellipse([50, 50, 150, 150], outline='gray', width=2)

# Add text
draw.text((100, 100), 'No Tumor', fill='gray', anchor='mm')

# Save the image
img.save('static/images/tumors/no_tumor.jpg') 