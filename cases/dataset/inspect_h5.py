# import h5py
# import sys
# import numpy as np

# def inspect_h5(file_path):
#     print("="*60)
#     print(f"Inspecting file: {file_path}")
#     print("="*60)

#     with h5py.File(file_path, "r") as f:

#         # ---- File attributes ----
#         print("\n📌 File Attributes:")
#         if len(f.attrs) == 0:
#             print("  None")
#         else:
#             for attr in f.attrs:
#                 print(f"  {attr}: {f.attrs[attr]}")

#         print("\n📂 Structure:")
        
#         def print_structure(name, obj):
#             print("\n" + "-"*50)
#             print(f"Path: {name}")
#             print(f"Type: {type(obj)}")

#             if isinstance(obj, h5py.Dataset):
#                 print(f"Shape: {obj.shape}")
#                 print(f"Dtype: {obj.dtype}")
#                 print(f"Size: {obj.size}")
#                 print(f"Compression: {obj.compression}")

#                 # Memory usage
#                 size_mb = obj.size * obj.dtype.itemsize / (1024**2)
#                 print(f"Memory (MB): {size_mb:.2f}")

#                 # Dataset attributes
#                 if len(obj.attrs) > 0:
#                     print("Attributes:")
#                     for attr in obj.attrs:
#                         print(f"  {attr}: {obj.attrs[attr]}")

#                 # Show small preview
#                 try:
#                     preview = obj[0]
#                     print("First element preview:")
#                     print(preview)
#                 except:
#                     print("Preview not available")

#         f.visititems(print_structure)

#     print("\n✅ Inspection complete.")
#     print("="*60)


# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python inspect_h5.py <path_to_file.h5>")
#     else:
#         inspect_h5(sys.argv[1])





# Example usage:
# python inspect_h5.py path/to/dataset.h5





import h5py
import sys
import numpy as np
import json

def inspect_h5(file_path, n_preview=5):

    print("=" * 60)
    print(f"Inspecting file: {file_path}")
    print("=" * 60)

    with h5py.File(file_path, "r") as f:

        # -------------------------------------------------
        # File attributes
        # -------------------------------------------------
        print("\n📌 File Attributes:")
        if len(f.attrs) == 0:
            print("  None")
        else:
            for attr in f.attrs:
                print(f"  {attr}: {f.attrs[attr]}")

        # Decode keys if stored as JSON
        input_keys = json.loads(f.attrs.get("input_keys", "[]"))
        output_keys = json.loads(f.attrs.get("output_keys", "[]"))

        print("\n📂 Structure:")
        print("-" * 50)

        for name, obj in f.items():
            print(f"\nDataset: {name}")
            print(f"Shape: {obj.shape}")
            print(f"Dtype: {obj.dtype}")
            size_mb = obj.size * obj.dtype.itemsize / (1024**2)
            print(f"Memory (MB): {size_mb:.2f}")

        # -------------------------------------------------
        # Show preview samples
        # -------------------------------------------------
        if "X" in f and "Y" in f:

            X = f["X"]
            Y = f["Y"]

            n_samples = X.shape[0]
            n_preview = min(n_preview, n_samples)

            print("\n🔎 Preview of dataset samples")
            print("-" * 50)

            for i in range(n_preview):

                print(f"\nSample {i}")

                # Input parameters
                if input_keys:
                    inputs = {k: X[i][j] for j, k in enumerate(input_keys)}
                else:
                    inputs = X[i]

                # Output values
                if output_keys:
                    outputs = {k: Y[i][j] for j, k in enumerate(output_keys)}
                else:
                    outputs = Y[i]

                print("Inputs :", inputs)
                print("Outputs:", outputs)

        # -------------------------------------------------
        # Dataset statistics
        # -------------------------------------------------
        if "X" in f and "Y" in f:

            print("\n📊 Dataset statistics")
            print("-" * 50)

            X = f["X"][:]
            Y = f["Y"][:]

            print("\nInputs statistics:")
            for i, key in enumerate(input_keys):
                col = X[:, i]
                print(f"{key:8s} | min={np.min(col):.3f} max={np.max(col):.3f} mean={np.mean(col):.3f} "
                      f"std={np.std(col):.3f} range={np.max(col)-np.min(col):.3f}")

            print("\nOutputs statistics:")
            for i, key in enumerate(output_keys):
                col = Y[:, i]
                print(f"{key:8s} | min={np.min(col):.3f} max={np.max(col):.3f} mean={np.mean(col):.3f} "
                      f"std={np.std(col):.3f} range={np.max(col)-np.min(col):.3f}")

            # Optional: check if outputs are nearly constant
            Y_ranges = np.max(Y, axis=0) - np.min(Y, axis=0)
            for i, key in enumerate(output_keys):
                if Y_ranges[i] < 1e-3:
                    print(f"⚠ Warning: Output '{key}' has very low variation (range={Y_ranges[i]:.6f})")

    print("\n✅ Inspection complete.")
    print("=" * 60)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python inspect_h5.py <path_to_file.h5> [n_preview]")
        sys.exit(1)

    file_path = sys.argv[1]

    n_preview = 5
    if len(sys.argv) > 2:
        n_preview = int(sys.argv[2])

    inspect_h5(file_path, n_preview)