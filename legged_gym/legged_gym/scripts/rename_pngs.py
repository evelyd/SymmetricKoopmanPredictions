import os
import argparse

def rename_png_files(directory):
    """
    Renames PNG files in the specified directory to have consistent leading zeros
    for numerical sorting (e.g., 1.png -> 0001.png, 10.png -> 0010.png).
    Assumes file names are integers followed by '.png'.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found.")
        return

    print(f"Processing PNG files in: {directory}")

    files = [f for f in os.listdir(directory) if f.endswith('.png')]

    # Determine the maximum number to find the necessary padding
    max_num = 0
    valid_files_to_process = []
    for filename in files:
        try:
            base_name = os.path.splitext(filename)[0]
            number = int(base_name)
            max_num = max(max_num, number)
            valid_files_to_process.append((filename, number))
        except ValueError:
            print(f"Warning: Skipping non-numeric file '{filename}'")
            continue

    if not valid_files_to_process:
        print("No valid numeric PNG files found to rename.")
        return

    # Calculate padding needed (e.g., 1002 needs 4 digits -> %04d)
    padding = len(str(max_num))

    print(f"Maximum image number found: {max_num}. Using {padding} digits for padding.")

    for filename, number in valid_files_to_process:
        new_filename = f"{number:0{padding}d}.png" # f-string with dynamic padding

        current_full_path = os.path.join(directory, filename)
        new_full_path = os.path.join(directory, new_filename)

        if filename != new_filename:
            try:
                os.rename(current_full_path, new_full_path)
                # print(f"Renamed '{filename}' to '{new_filename}'")
            except OSError as e:
                print(f"Error renaming '{filename}' to '{new_filename}': {e}")
        else:
            print(f"'{filename}' is already correctly formatted.")

    print("Renaming process complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename PNG files in a directory for sequential FFmpeg processing.")
    parser.add_argument("directory", help="The path to the directory containing the PNG images.")

    args = parser.parse_args()

    rename_png_files(args.directory)