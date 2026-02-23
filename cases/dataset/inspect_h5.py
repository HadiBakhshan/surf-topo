import h5py
import sys
import numpy as np

def inspect_h5(file_path):
    print("="*60)
    print(f"Inspecting file: {file_path}")
    print("="*60)

    with h5py.File(file_path, "r") as f:

        # ---- File attributes ----
        print("\n📌 File Attributes:")
        if len(f.attrs) == 0:
            print("  None")
        else:
            for attr in f.attrs:
                print(f"  {attr}: {f.attrs[attr]}")

        print("\n📂 Structure:")
        
        def print_structure(name, obj):
            print("\n" + "-"*50)
            print(f"Path: {name}")
            print(f"Type: {type(obj)}")

            if isinstance(obj, h5py.Dataset):
                print(f"Shape: {obj.shape}")
                print(f"Dtype: {obj.dtype}")
                print(f"Size: {obj.size}")
                print(f"Compression: {obj.compression}")

                # Memory usage
                size_mb = obj.size * obj.dtype.itemsize / (1024**2)
                print(f"Memory (MB): {size_mb:.2f}")

                # Dataset attributes
                if len(obj.attrs) > 0:
                    print("Attributes:")
                    for attr in obj.attrs:
                        print(f"  {attr}: {obj.attrs[attr]}")

                # Show small preview
                try:
                    preview = obj[0]
                    print("First element preview:")
                    print(preview)
                except:
                    print("Preview not available")

        f.visititems(print_structure)

    print("\n✅ Inspection complete.")
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_h5.py <path_to_file.h5>")
    else:
        inspect_h5(sys.argv[1])

# Example usage:
# python inspect_h5.py path/to/dataset.h5
