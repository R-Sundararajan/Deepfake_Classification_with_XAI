from pathlib import Path
from explainability.visualization import visualize
from utils.config import TEST_DIR, OUTPUT_DIR

def run_visualizations():
    real_images = list((TEST_DIR / "real").glob("*.jpg"))
    fake_images = list((TEST_DIR / "fake").glob("*.jpg"))

    if not real_images or not fake_images:
        print("Test images not found.")
        return

    real_img = real_images[5000]
    fake_img = fake_images[3000]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Generating Heatmap for REAL Image ---")
    visualize(real_img, save_path=OUTPUT_DIR / "heatmap_real.png")

    print("\n--- Generating Heatmap for FAKE Image ---")
    visualize(fake_img, save_path=OUTPUT_DIR / "heatmap_fake.png")

if __name__ == "__main__":
    run_visualizations()