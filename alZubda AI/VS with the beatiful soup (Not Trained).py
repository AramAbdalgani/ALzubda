import requests
from bs4 import BeautifulSoup
import os
import re
import cv2
import pytesseract
import csv
from PIL import Image
import numpy as np




config = ('-l eng --oem 1 --psm 3')

# Set the URL of the news website you want to scrape
url = "https://www.foxnews.com/"
response = requests.get(url)

# Create a BeautifulSoup object from the HTML content of the website
soup = BeautifulSoup(response.content, "html.parser")

# Initialize empty lists to store the titles, bodies, and images
titles = []
bodies = []
images = []
image_content = []
# Find all HTML tags containing titles on the website
title_tags = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'], class_=lambda x: x not in ['hidden'], id=lambda x: x not in ['hidden'])

# Extract the text content of each title tag and append it to the titles list
for title_tag in title_tags:
    title_text = title_tag.get_text().strip()
    titles.append(title_text)

# Find all HTML tags containing body text on the website
body_tags = soup.find_all(['div', 'p', 'ul', 'ol', 'li'], class_=lambda x: x not in ['hidden'], id=lambda x: x not in ['hidden'])

# Extract the text content of each body tag and append it to the bodies list
for body_tag in body_tags:
    body_text = body_tag.get_text().strip()
    bodies.append(body_text)

# Create a directory to store the images (if it doesn't already exist)
img_dir = "images"
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

# Find all HTML tags containing images on the website
img_tags = soup.find_all('img', src=True)

# Download each image and save it to the images directory
for img in img_tags:
    img_url = img.get('src')
    if not img_url.startswith('http'):
        img_url = url + img_url
    response = requests.get(img_url)
    img_name = re.sub(r'[^\w\s-]', '', os.path.basename(img_url))
    img_name_2 = img_name.replace(' ', '_')
    img_path = os.path.join(img_dir, img_name_2)

    with open(img_path, "wb") as f:
        f.write(response.content)

    img_name_jpg = img_path[:-4] + '.jpg'
    img = cv2.imread(img_path)
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        gray = cv2.medianBlur(gray, 3)
        text = pytesseract.image_to_string(gray, config=config)
        if text:
            image_content.append((img_name_jpg, text))
    else:
        print(f"Could not read image at {img_path}")
        pass

# Use Vision based Page Segmentation Algorithm to extract text regions from each image
text_regions = []
for img_path, content in image_content:
    try:
        img = cv2.imread(img_path)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            gray = cv2.medianBlur(gray, 3)
            text = pytesseract.image_to_string(gray, config=config)
            if text:
                text_regions.append(text)
        else:
            print(f"Could not read image at {img_path}")
        if img_path is None:
            pass
    except cv2.error as e:
        print(f"Error reading image at {img_path}: {e}")
        


width = 500
height = 500
color = (255, 255, 255)


image = Image.new('RGB', (width, height), color)


image.save('page.jpg', 'JPEG')


page_image = Image.open('page.jpg')
page_array = np.array(page_image.convert('L'))

threshold = cv2.adaptiveThreshold(page_array, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize=15, C=2)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (3,3))
opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)
dilate = cv2.dilate(opening, kernel, iterations=2)


contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if h > 50 and w > 50:
        crop_img = page_array[y:y+h, x:x+w]
        img_region = pytesseract.image_to_string(Image.fromarray(crop_img), lang='eng', config='--psm 6')
        if img_region:
            font_size = h / 10
            if font_size > 2:
                titles.append(img_region)
            else:
                bodies.append(img_region)
            img_name = f'image_{len(images)}.jpg'
            cv2.imwrite(img_name, crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            images.append(img_name)


with open('extracted_data.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Title', 'Body', 'Image Path', 'Image Content'])
    for i in range(len(titles)):
        if i < len(bodies) and i < len(images) and i < len(image_content):
            writer.writerow([titles[i], bodies[i], images[i], image_content[i][1]])


#print(titles)
print("done")
#print(bodies)
print(image_content)



















