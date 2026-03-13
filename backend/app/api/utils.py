from pathlib import Path

import os
_PRODUCT_IMAGES_DIR = Path(os.getenv("PRODUCT_IMAGES_DIR", str(Path(__file__).resolve().parent.parent.parent.parent / "product_images")))

def build_prestashop_image_url(id_image: int) -> str:
    from app.core.config import logger
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        local_path = _PRODUCT_IMAGES_DIR / f"{id_image}{ext}"
        if local_path.exists():
            return f"http://localhost:8000/product_images/{id_image}{ext}"

    digits = "/".join(str(id_image))
    return f"https://uniartminerales.com/img/p/{digits}/{id_image}-home_default.jpg"
