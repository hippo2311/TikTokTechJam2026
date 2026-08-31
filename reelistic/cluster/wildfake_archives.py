"""Approved WildFake archive inventory used by download and manifest tooling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WildFakeArchive:
    key: str
    remote_path: str
    compressed_gb: float
    label: int
    family: str


APPROVED_ARCHIVES = (
    WildFakeArchive("adm", "Images/Diffusion_based/ADM.zip", 18.55, 1, "ADM"),
    WildFakeArchive("ddim", "Images/Diffusion_based/DDIM.zip", 6.05, 1, "DDIM"),
    WildFakeArchive("ddpm", "Images/Diffusion_based/DDPM.zip", 8.14, 1, "DDPM"),
    WildFakeArchive("imagen", "Images/Diffusion_based/Imagen.zip", 17.07, 1, "Imagen"),
    WildFakeArchive("vqdm", "Images/Diffusion_based/VQDM.zip", 17.38, 1, "VQDM"),
    WildFakeArchive("gan", "Images/GAN_based.zip", 47.33, 1, "GAN_based"),
    WildFakeArchive("other", "Images/Other_based.zip", 13.34, 1, "Other_based"),
    WildFakeArchive("afhq", "Images/Real/afhq.zip", 0.4524, 0, "afhq"),
    WildFakeArchive("celebahq", "Images/Real/celebahq.zip", 0.35099, 0, "celebahq"),
    WildFakeArchive("church", "Images/Real/church.zip", 1.16, 0, "church"),
    WildFakeArchive("ffhq", "Images/Real/ffhq.zip", 0.81884, 0, "ffhq"),
    WildFakeArchive("imagenet", "Images/Real/imagenet.zip", 1.38, 0, "imagenet"),
    WildFakeArchive("laion5b", "Images/Real/laion5b.zip", 24.80, 0, "laion5b"),
)

FINAL_TEST_ARCHIVES = (
    WildFakeArchive(
        "dalle_advanced",
        "Images/Diffusion_based/DALLE.zip",
        25.59,
        1,
        "DALL-E_Advanced",
    ),
    WildFakeArchive(
        "coco_val2017",
        "Images/Real/coco.zip",
        2.35,
        0,
        "COCO_val2017",
    ),
)

# These archives are deliberately forbidden from the training archive inventory.
# They are downloaded into Dataset/WildFake/final_test and evaluated only after
# model selection and calibration are frozen.
EXCLUDED_REMOTE_PATHS = frozenset(
    archive.remote_path for archive in FINAL_TEST_ARCHIVES
)

ARCHIVES_BY_KEY = {archive.key: archive for archive in APPROVED_ARCHIVES}
FINAL_TEST_ARCHIVES_BY_KEY = {
    archive.key: archive for archive in FINAL_TEST_ARCHIVES
}


def archive_for_index(index: int) -> WildFakeArchive:
    if index < 0 or index >= len(APPROVED_ARCHIVES):
        raise IndexError(
            f"Archive index {index} is outside 0..{len(APPROVED_ARCHIVES) - 1}."
        )
    return APPROVED_ARCHIVES[index]


def archive_for_key(key: str) -> WildFakeArchive:
    try:
        return ARCHIVES_BY_KEY[key.lower()]
    except KeyError as exc:
        raise KeyError(
            f"Unknown or prohibited archive {key!r}; approved keys are "
            f"{sorted(ARCHIVES_BY_KEY)}"
        ) from exc


def final_test_archive_for_index(index: int) -> WildFakeArchive:
    if index < 0 or index >= len(FINAL_TEST_ARCHIVES):
        raise IndexError(
            f"Final-test archive index {index} is outside "
            f"0..{len(FINAL_TEST_ARCHIVES) - 1}."
        )
    return FINAL_TEST_ARCHIVES[index]


def final_test_archive_for_key(key: str) -> WildFakeArchive:
    try:
        return FINAL_TEST_ARCHIVES_BY_KEY[key.lower()]
    except KeyError as exc:
        raise KeyError(
            f"Unknown final-test archive {key!r}; allowed keys are "
            f"{sorted(FINAL_TEST_ARCHIVES_BY_KEY)}"
        ) from exc
