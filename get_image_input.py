import os
import json

def rename_images_and_generate_json():
    files = os.listdir('.')
    
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
    images = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
    
    images.sort()  
    image_data = []
    
    for idx, image in enumerate(images, start=1):
        ext = os.path.splitext(image)[1]
        new_name = f"{idx}{ext}"
        os.rename(image, new_name)
        full_path = os.path.abspath(new_name)
        image_data.append({
            "id": idx,
            "path": full_path
        })
    
    with open('input.json', 'w') as json_file:
        json.dump({"images": image_data}, json_file, indent=4)
    
    print(f"Renamed {len(images)} images and created input.json.")

if __name__ == "__main__":
    rename_images_and_generate_json()
