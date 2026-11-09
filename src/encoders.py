"""Uniform interface to location encoders: encode(lonlat[N,2] in degrees) -> float32 [N,D]."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "external" / "TTE"))
sys.path.insert(0, str(ROOT / "src"))

SATCLIP_CKPT = {
    "satclip_l10": ROOT / "data/raw/satclip_ckpt/satclip-vit16-l10.ckpt",
    "satclip_l40": ROOT / "data/raw/satclip_ckpt/satclip-vit16-l40.ckpt",
}


def _batched(fn, lonlat, bs=8192):
    out = []
    for i in range(0, len(lonlat), bs):
        out.append(fn(lonlat[i : i + bs]))
    return np.concatenate(out, 0).astype(np.float32)


class TTEEncoder:
    name = "tte"

    def __init__(self):
        from tte import TTE

        self.model = TTE.from_pretrained("MVRL/TTE").eval()
        self.model.normalize = True  # matches eval_range.py default

    def __call__(self, lonlat):
        def f(b):
            latlon = torch.tensor(b[:, ::-1].copy(), dtype=torch.float32)
            with torch.no_grad():
                return self.model.encode(latlon).numpy()

        return _batched(f, lonlat)

    def sites_lonlat(self):
        s = self.model.location.sites.detach().numpy()
        lat = np.degrees(np.arcsin(np.clip(s[:, 2], -1, 1)))
        lon = np.degrees(np.arctan2(s[:, 1], s[:, 0]))
        return np.stack([lon, lat], 1), self.model.location.tau.detach().numpy()


class SatCLIPEncoder:
    def __init__(self, name):
        from vendor.satclip_le.location_encoder import LocationEncoder, get_neural_network, get_positional_encoding

        self.name = name
        ckpt = torch.load(SATCLIP_CKPT[name], map_location="cpu", weights_only=True)
        hp = ckpt["hyper_parameters"]
        posenc = get_positional_encoding(
            hp["le_type"],
            hp["legendre_polys"],
            hp["harmonics_calculation"],
            hp["min_radius"],
            hp["max_radius"],
            hp["frequency_num"],
        )
        nnet = get_neural_network(
            hp["pe_type"], posenc.embedding_dim, hp["embed_dim"], hp["capacity"], hp["num_hidden_layers"]
        )
        sd = {k[k.index("nnet") :]: v for k, v in ckpt["state_dict"].items() if "nnet" in k}
        self.model = LocationEncoder(posenc, nnet).double()
        self.model.load_state_dict(sd)
        self.model.eval()

    def __call__(self, lonlat):
        def f(b):
            with torch.no_grad():
                return self.model(torch.tensor(b, dtype=torch.float64)).float().numpy()

        return _batched(f, lonlat)


class SHBasis:
    """Fixed real spherical-harmonic basis, L=10 (100 features); no learning."""

    name = "sh_l10"

    def __init__(self, L=10):
        from vendor.satclip_le.positional_encoding import SphericalHarmonics

        self.pe = SphericalHarmonics(legendre_polys=L, harmonics_calculation="analytic")

    def __call__(self, lonlat):
        def f(b):
            with torch.no_grad():
                return self.pe(torch.tensor(b, dtype=torch.float64)).float().numpy()

        return _batched(f, lonlat)


class XYZ:
    name = "xyz"

    def __call__(self, lonlat):
        lon, lat = np.radians(lonlat[:, 0]), np.radians(lonlat[:, 1])
        return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], 1).astype(np.float32)


def get_encoder(name):
    if name == "tte":
        return TTEEncoder()
    if name.startswith("satclip"):
        return SatCLIPEncoder(name)
    if name == "sh_l10":
        return SHBasis()
    if name == "xyz":
        return XYZ()
    raise ValueError(name)


ENCODERS = ["tte", "satclip_l40", "satclip_l10", "sh_l10", "xyz"]
