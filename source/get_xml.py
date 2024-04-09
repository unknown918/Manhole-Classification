import os
import shutil

# Specify the source directory where your XML files currently are
source_dir = '../labels'

# Specify the destination directory where you want to move the XML files
dest_dir = '../test_xmls'

# Loop through the files in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith(".xml"):
        # Construct full file path
        source_file = os.path.join(source_dir, filename)
        destination_file = os.path.join(dest_dir, filename)

        # Move the XML file to the destination directory
        shutil.move(source_file, destination_file)
        print(f'Moved: {source_file} to {destination_file}')
    else:
        # If the file does not end with .xml, we skip it
        continue

print("All XML files have been moved.")
