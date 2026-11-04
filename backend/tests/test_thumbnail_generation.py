from pathlib import Path
from typing import Sequence
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from app.utils import images


def _assert_jpeg(path: Path, expected_size: tuple[int, int]) -> None:
    assert path.stat().st_size > 0
    with Image.open(path) as result:
        assert result.format == "JPEG"
        result.load()
        assert result.size == expected_size


def _write_numeric(path: Path, mode: str, samples: Sequence[float]) -> None:
    dtype = {"I": np.int32, "F": np.float32}.get(mode, np.uint16)
    pixels = np.tile(np.asarray(samples, dtype=dtype), (48, 1))
    source = Image.fromarray(pixels)
    if mode == "I;16B":
        source = Image.fromarray(pixels.astype(">u2"))
    source.save(path, "PNG" if path.suffix == ".png" else "TIFF")
    with Image.open(path) as reopened:
        assert reopened.mode == mode
        reopened.load()


@pytest.mark.parametrize(
    "mode,extension,transparent",
    [
        ("RGB", ".png", False),
        ("L", ".png", False),
        ("RGBA", ".png", False),
        ("LA", ".png", False),
        ("P", ".png", False),
        ("P", ".png", True),
        ("1", ".png", False),
        ("CMYK", ".tiff", False),
    ],
)
def test_real_image_modes_generate_jpeg(
    tmp_path: Path, mode: str, extension: str, transparent: bool
) -> None:
    source_path = tmp_path / f"source{extension}"
    target = tmp_path / "thumbnail.jpg"
    source = Image.new(mode, (960, 480))
    if mode == "P":
        source.putpalette([80, 120, 160] * 256)
        if transparent:
            source.info["transparency"] = 0
    source.save(source_path)
    before = source_path.read_bytes()
    with Image.open(source_path) as reopened:
        assert reopened.mode == mode
        assert ("transparency" in reopened.info) == transparent

    assert images.image_util_generate_thumbnail(str(source_path), str(target)) is True

    _assert_jpeg(target, (600, 300))
    assert source_path.read_bytes() == before


@pytest.mark.parametrize("mode", ["1", "L", "RGB", "CMYK", "YCbCr", "RGBX"])
def test_supported_encoder_modes_keep_existing_output(
    tmp_path: Path, mode: str
) -> None:
    source = Image.new(mode, (960, 480))
    expected = tmp_path / "expected.jpg"
    target = tmp_path / "thumbnail.jpg"
    reference = source.copy()
    reference.thumbnail((600, 600))
    reference.save(expected, "JPEG")

    with patch.object(images.Image, "open", return_value=source):
        assert images.image_util_generate_thumbnail("in-memory", str(target)) is True

    _assert_jpeg(target, (600, 300))
    assert target.read_bytes() == expected.read_bytes()


@pytest.mark.parametrize("mode", ["RGBA", "LA", "P"])
def test_transparency_keeps_alpha_discard_policy(tmp_path: Path, mode: str) -> None:
    source = Image.new(mode, (32, 16))
    if mode == "RGBA":
        source.paste((80, 120, 160, 0), (0, 0, 32, 16))
    elif mode == "LA":
        source.paste((120, 0), (0, 0, 32, 16))
    else:
        source.putpalette([80, 120, 160] * 256)
        source.info["transparency"] = 0
    source_path = tmp_path / "source.png"
    target = tmp_path / "thumbnail.jpg"
    expected = tmp_path / "expected.jpg"
    source.save(source_path)
    source.convert("RGB").save(expected, "JPEG")

    assert images.image_util_generate_thumbnail(str(source_path), str(target)) is True

    assert target.read_bytes() == expected.read_bytes()


@pytest.mark.parametrize(
    "source_size,bounds,expected",
    [
        ((1200, 600), (600, 600), (600, 300)),
        ((600, 1200), (600, 600), (300, 600)),
        ((1200, 600), (240, 100), (200, 100)),
        ((32, 16), (600, 600), (32, 16)),
    ],
)
def test_thumbnail_bounds_preserve_aspect_and_do_not_enlarge(
    tmp_path: Path,
    source_size: tuple[int, int],
    bounds: tuple[int, int],
    expected: tuple[int, int],
) -> None:
    source_path = tmp_path / "source.png"
    target = tmp_path / "thumbnail.jpg"
    Image.new("LA", source_size, (120, 255)).save(source_path)

    assert images.image_util_generate_thumbnail(str(source_path), str(target), bounds)

    _assert_jpeg(target, expected)
    assert expected[0] <= bounds[0] and expected[1] <= bounds[1]
    assert expected[0] * source_size[1] == expected[1] * source_size[0]


@pytest.mark.parametrize(
    "mode,extension,minimum,maximum",
    [
        ("I", ".tiff", -2147483648.0, 2147483647.0),
        ("F", ".tiff", 0.0, 1.0),
        ("I;16", ".png", 0.0, 65535.0),
        ("I;16", ".tiff", 1000.0, 60000.0),
        ("I;16B", ".tiff", 1000.0, 60000.0),
    ],
)
def test_numeric_gradients_keep_contrast_and_pixel_order(
    tmp_path: Path, mode: str, extension: str, minimum: float, maximum: float
) -> None:
    source_path = tmp_path / f"source{extension}"
    target = tmp_path / "thumbnail.jpg"
    _write_numeric(source_path, mode, np.linspace(minimum, maximum, 1200).tolist())
    before = source_path.read_bytes()

    assert images.image_util_generate_thumbnail(str(source_path), str(target)) is True

    _assert_jpeg(target, (600, 24))
    with Image.open(target) as result:
        grayscale = np.asarray(result.convert("L"))
    samples = [int(grayscale[12, x]) for x in (60, 300, 540)]
    assert samples[0] < samples[1] < samples[2]
    assert samples[2] - samples[0] >= 190
    assert samples == pytest.approx([25, 127, 229], abs=6)
    assert source_path.read_bytes() == before


@pytest.mark.parametrize(
    "mode,value,expected",
    [
        ("I", -400.0, 0),
        ("I", 128.0, 128),
        ("I", 65535.0, 255),
        ("F", -1.0, 0),
        ("F", 0.5, 0),
        ("F", 128.5, 128),
        ("F", 300.0, 255),
        ("I;16", 128.0, 128),
        ("I;16", 65535.0, 255),
    ],
)
def test_constant_numeric_images_clamp_without_division(
    tmp_path: Path, mode: str, value: float, expected: int
) -> None:
    source_path = tmp_path / "source.tiff"
    target = tmp_path / "thumbnail.jpg"
    _write_numeric(source_path, mode, [value] * 64)

    assert images.image_util_generate_thumbnail(str(source_path), str(target)) is True

    _assert_jpeg(target, (64, 48))
    with Image.open(target) as result:
        pixels = np.asarray(result.convert("L"), dtype=np.int16)
    assert np.max(np.abs(pixels - expected)) <= 2


@pytest.mark.parametrize(
    "values,expected",
    [
        ([float("nan"), float("inf"), -float("inf")], [0, 0, 0]),
        ([float("nan"), 0.0, 0.5, 1.0, float("inf")], [0, 0, 127, 255, 0]),
        ([float("nan"), 128.0, 128.0, float("inf")], [0, 128, 128, 0]),
    ],
)
def test_nonfinite_float_pixels_are_black(
    tmp_path: Path, values: list[float], expected: list[int]
) -> None:
    source_path = tmp_path / "source.tiff"
    target = tmp_path / "thumbnail.jpg"
    samples = [value for value in values for _ in range(64)]
    _write_numeric(source_path, "F", samples)

    assert images.image_util_generate_thumbnail(str(source_path), str(target)) is True

    _assert_jpeg(target, (len(values) * 64, 48))
    with Image.open(target) as result:
        grayscale = np.asarray(result.convert("L"))
    actual = [int(grayscale[24, index * 64 + 32]) for index in range(len(values))]
    assert actual == pytest.approx(expected, abs=2)


@pytest.mark.parametrize("failure", ["missing", "corrupt", "write"])
def test_failures_return_false_and_log_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    source_path = tmp_path / "source.png"
    target = tmp_path / "thumbnail.jpg"
    if failure == "corrupt":
        source_path.write_bytes(b"not an image")
    elif failure == "write":
        Image.new("LA", (32, 16), (120, 255)).save(source_path)
        target = tmp_path / "missing-directory" / "thumbnail.jpg"
    error = Mock()
    monkeypatch.setattr(images.logger, "error", error)

    assert images.image_util_generate_thumbnail(str(source_path), str(target)) is False

    error.assert_called_once()
    message = error.call_args.args[0]
    assert f"Error generating thumbnail for {source_path}: " in message
    assert len(message.split(": ", 1)[1]) > 0
    assert not target.exists()


@pytest.mark.parametrize("mode", ["I;16L", "I;16N"])
def test_other_16_bit_variants_normalize_before_resize(
    tmp_path: Path, mode: str
) -> None:
    pixels = np.tile(np.linspace(0, 65535, 960, dtype=np.uint16), (48, 1))
    raw_pixels = pixels.astype("<u2") if mode == "I;16L" else pixels
    source = Image.frombytes(mode, (960, 48), raw_pixels.tobytes())
    target = tmp_path / "thumbnail.jpg"
    assert source.mode == mode

    with patch.object(images.Image, "open", return_value=source):
        assert images.image_util_generate_thumbnail("in-memory", str(target)) is True

    _assert_jpeg(target, (600, 30))
    with Image.open(target) as result:
        grayscale = np.asarray(result.convert("L"))
    samples = [int(grayscale[15, x]) for x in (60, 300, 540)]
    assert samples == pytest.approx([25, 127, 229], abs=6)
