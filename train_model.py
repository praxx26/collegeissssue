import sys
from pathlib import Path

from ultralytics import YOLO
import torch


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

DATA_YAML = r"C:\campus_issue\combined_dataset_7classes\data.yaml"

# Load the weights from the previous 50-epoch run
MODEL_TYPE = r"C:\campus_issue\runs\detect\campus_yolo11n\weights\last.pt"

PROJECT_DIR = r"C:\campus_issue\runs\detect"

RUN_NAME = "campus_yolo11n_continued"

# Maximum number of epochs
EPOCHS = 20

# Smaller image size = faster training
IMG_SIZE = 512

# Fixed batch size for better GPU utilization
BATCH_SIZE = 16

# Windows CPU workers (0 is often much faster on Windows to avoid overhead)
WORKERS = 0

# Stop if validation does not improve for 10 epochs
PATIENCE = 10


# ============================================================
# DATASET VERIFICATION
# ============================================================

def verify_dataset():

    print("Verifying dataset structure...")

    base_dir = Path(
        r"C:\campus_issue\combined_dataset_7classes"
    )

    if not base_dir.exists():

        print(
            f"ERROR: Dataset directory does not exist:\n"
            f"{base_dir}"
        )

        sys.exit(1)

    for split in ["train", "valid", "test"]:

        img_dir = base_dir / split / "images"
        lbl_dir = base_dir / split / "labels"

        if not img_dir.exists():

            print(
                f"ERROR: Missing image directory:\n"
                f"{img_dir}"
            )

            sys.exit(1)

        if not lbl_dir.exists():

            print(
                f"ERROR: Missing label directory:\n"
                f"{lbl_dir}"
            )

            sys.exit(1)

    yaml_file = base_dir / "data.yaml"

    if not yaml_file.exists():

        print(
            f"ERROR: Missing data.yaml:\n"
            f"{yaml_file}"
        )

        sys.exit(1)

    print(
        "Dataset verification passed!"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Verify dataset
    # --------------------------------------------------------

    verify_dataset()


    # --------------------------------------------------------
    # 2. Check GPU
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = "0"

        print("\nGPU detected!")

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print("Using device: 0")

    else:

        device = "cpu"

        print(
            "\nCUDA GPU not available."
        )

        print("Using CPU.")


    # --------------------------------------------------------
    # 3. Load YOLO11n
    # --------------------------------------------------------

    print(
        f"\nLoading pretrained model: "
        f"{MODEL_TYPE}"
    )

    model = YOLO(MODEL_TYPE)

    print(
        "Model loaded successfully!"
    )


    # --------------------------------------------------------
    # 4. Start training
    # --------------------------------------------------------

    print("\n==========================================")

    print(
        "       STARTING YOLO11n TRAINING"
    )

    print("==========================================")

    print(
        f"Dataset       : {DATA_YAML}"
    )

    print(
        f"Model         : {MODEL_TYPE}"
    )

    print(
        f"Maximum epochs: {EPOCHS}"
    )

    print(
        f"Image size    : {IMG_SIZE}"
    )

    print(
        f"Batch size    : {BATCH_SIZE} (automatic)"
    )

    print(
        f"Early stopping: {PATIENCE} epochs"
    )

    print(
        f"Device        : {device}"
    )

    print("==========================================")

    print(
        "\nTraining started...\n"
    )


    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=device,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=False,
        # Early stopping
        patience=PATIENCE,
        
    )



    # --------------------------------------------------------
    # 5. Get training directory
    # --------------------------------------------------------

    actual_run_dir = Path(
        results.save_dir
    )

    print(
        "\nTraining run directory:"
    )

    print(actual_run_dir)


    # --------------------------------------------------------
    # 6. Find best.pt
    # --------------------------------------------------------

    best_model_path = (
        actual_run_dir
        / "weights"
        / "best.pt"
    )


    if not best_model_path.exists():

        print(
            "\nERROR: best.pt was not found!"
        )

        print(
            "Expected:"
        )

        print(best_model_path)

        sys.exit(1)


    # --------------------------------------------------------
    # 7. Training complete
    # --------------------------------------------------------

    print("\n==========================================")

    print(
        "          TRAINING COMPLETED"
    )

    print("==========================================")

    print(
        "\nBest model:"
    )

    print(best_model_path)


    # --------------------------------------------------------
    # 8. Load best model
    # --------------------------------------------------------

    print(
        "\nLoading best.pt..."
    )

    best_model = YOLO(
        best_model_path
    )

    print(
        "best.pt loaded successfully!"
    )


    # --------------------------------------------------------
    # 9. Validation
    # --------------------------------------------------------

    print("\n==========================================")

    print(
        "          RUNNING VALIDATION"
    )

    print("==========================================")


    val_results = best_model.val(

        data=DATA_YAML,

        split="val",

        imgsz=IMG_SIZE,

        device=device
    )


    # --------------------------------------------------------
    # 10. Validation metrics
    # --------------------------------------------------------

    print("\n==========================================")

    print(
        "          VALIDATION RESULTS"
    )

    print("==========================================")


    try:

        print(
            f"Precision : "
            f"{val_results.box.mp:.4f}"
        )

        print(
            f"Recall    : "
            f"{val_results.box.mr:.4f}"
        )

        print(
            f"mAP50     : "
            f"{val_results.box.map50:.4f}"
        )

        print(
            f"mAP50-95  : "
            f"{val_results.box.map:.4f}"
        )

    except Exception as e:

        print(
            "Could not read validation metrics."
        )

        print(
            "Reason:",
            e
        )


    # --------------------------------------------------------
    # 11. Test evaluation
    # --------------------------------------------------------

    print("\n==========================================")

    print(
        "       RUNNING FINAL TEST EVALUATION"
    )

    print("==========================================")


    test_results = best_model.val(

        data=DATA_YAML,

        split="test",

        imgsz=IMG_SIZE,

        device=device
    )


    # --------------------------------------------------------
    # 12. Test metrics
    # --------------------------------------------------------

    print("\n==========================================")

    print(
        "            TEST RESULTS"
    )

    print("==========================================")


    try:

        print(
            f"Precision : "
            f"{test_results.box.mp:.4f}"
        )

        print(
            f"Recall    : "
            f"{test_results.box.mr:.4f}"
        )

        print(
            f"mAP50     : "
            f"{test_results.box.map50:.4f}"
        )

        print(
            f"mAP50-95  : "
            f"{test_results.box.map:.4f}"
        )

    except Exception as e:

        print(
            "Could not read test metrics."
        )

        print(
            "Reason:",
            e
        )


    # --------------------------------------------------------
    # 13. Test predictions
    # --------------------------------------------------------

    print("\n==========================================")

    print(
        "       CREATING TEST PREDICTIONS"
    )

    print("==========================================")


    test_images_dir = Path(
        r"C:\campus_issue\combined_dataset_7classes\test\images"
    )


    if (
        test_images_dir.exists()
        and any(test_images_dir.iterdir())
    ):

        best_model.predict(

            source=str(
                test_images_dir
            ),

            conf=0.50,

            save=True,

            project=str(
                actual_run_dir
            ),

            name="test_predictions",

            device=device
        )


        print(
            "\nTest predictions saved to:"
        )

        print(
            actual_run_dir
            / "test_predictions"
        )

    else:

        print(
            "No test images found."
        )


    # --------------------------------------------------------
    # 14. Final information
    # --------------------------------------------------------

    print("\n==========================================================")

    print(
        "                 FINAL MODEL"
    )

    print("==========================================================")

    print(
        "\nBEST MODEL:"
    )

    print("best.pt")

    print(
        "\nPATH:"
    )

    print(best_model_path)

    print(
        "\nTRAINING RUN:"
    )

    print(actual_run_dir)

    print(
        "\n=========================================================="
    )

    print(
        "\nThis best.pt file can be used "
        "to detect new campus images."
    )


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    main()