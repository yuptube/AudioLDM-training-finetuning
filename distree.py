import os
from PIL import Image, ImageDraw, ImageFont
def print_tree(path, prefix=""):
    """Recursively prints a tree structure for the given filesystem path."""
    basename = os.path.basename(path) or path
    print(prefix + basename + ("/" if os.path.isdir(path) else ""))
    if os.path.isdir(path):
        entries = sorted(os.listdir(path))
        for i, name in enumerate(entries):
            sub = os.path.join(path, name)
            branch = "└──" if i < len(entries) - 1 else "└── "
            extension = "│   " if i < len(entries) - 1 else "    "
            print_tree(sub, prefix + branch.replace("├", "│").replace("└", " ") + extension)


def save_tree_as_png(root_path, output_path, font_path=None, font_size=12, padding=10):
    """
    Generate and save a directory tree of `root_path` as a PNG image at `output_path`,
    ignoring `__init__.py` files and `__pycache__` directories.

    Parameters:
    - root_path (str): Path of the directory to visualize.
    - output_path (str): File path where the PNG will be saved.
    - font_path (str, optional): Path to a .ttf font file. Defaults to PIL's default font.
    - font_size (int, optional): Font size if using a custom font. Defaults to 12.
    - padding (int, optional): Pixels of padding around the text. Defaults to 10.
    """
    def build_tree(path, prefix=""):
        lines = []
        name = os.path.basename(path) or path
        lines.append(prefix + name + ("/" if os.path.isdir(path) else ""))
        if os.path.isdir(path):
            entries = sorted(os.listdir(path))
            # Exclude __init__.py files and __pycache__ directories
            entries = [e for e in entries if e != "__init__.py" and e != "__pycache__"]
            for idx, entry in enumerate(entries):
                sub_path = os.path.join(path, entry)
                last = idx == len(entries) - 1
                branch = "└── " if last else "├── "
                extension = "    " if last else "│   "
                lines.append(prefix + branch + entry + ("/" if os.path.isdir(sub_path) else ""))
                if os.path.isdir(sub_path):
                    subtree = build_tree(sub_path, prefix + extension)
                    lines.extend(subtree[1:])
        return lines

    # Generate ASCII lines for the tree
    lines = build_tree(root_path)

    # Load font
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    # Calculate image dimensions
    sample_bbox = font.getbbox("A")
    line_height = sample_bbox[3] - sample_bbox[1]
    max_width = max(font.getlength(line) for line in lines)
    img_width = int(max_width + padding * 2)
    img_height = line_height * len(lines) + padding * 2

    # Create image with white background
    img = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(img)

    # Render each line of text
    y = padding
    for line in lines:
        draw.text((padding, y), line, font=font, fill="black")
        y += line_height

    # Save to PNG
    img.save(output_path)


print_tree("audioldm_train")
save_tree_as_png("audioldm_train" , "waveform_comparison")