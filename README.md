# radarwow-pipeline

**radar-wow stage 10: End-to-end pipeline.** Chains the stages per radar from a config; India / Delhi (IMD C-band, ISRO S-band) first.

Part of [radar-wow](https://github.com/manmeet3591/radar-wow), a set of repos that together build a radar QPE system that works for S, C and X band radars.
Every algorithm is one class implementing one paper, with the same interface: it takes an xradar sweep or volume with canonical moment names, declares the bands and moments it needs, and records provenance.

## Sub-components

| sub-component | algorithm | source | DOI | status |
|---|---|---|---|---|
| hourly totals from raw NEXRAD (chain of verify notebooks 01-05) | `hourly_totals` | QVP ML, VPR, blockage + hybrid scan, adaptive-alpha synthetic rate, advection totals |  | implemented + notebook |
| config-driven chain | `Pipeline.from_toml` | checks bands and moments before running |  | planned |
| India radar review | - | Bhat, Vivekanandan & Pradhan 2025, MAUSAM | [doi:10.54302/mausam.v76i1.6497](https://doi.org/10.54302/mausam.v76i1.6497) | to review |
| India QPE | - | Sharma et al. 2025, Phys. Chem. Earth | [doi:10.1016/j.pce.2025.104182](https://doi.org/10.1016/j.pce.2025.104182) | to review |
| India dual-pol QPE | - | Saini et al. 2026, Radio Science | [doi:10.1029/2025RS008484](https://doi.org/10.1029/2025RS008484) | to review |

## Install

```bash
pip install git+https://github.com/manmeet3591/radarwow-core git+https://github.com/manmeet3591/radarwow-pipeline
```

## Usage

```python
from radarwow.core import load_sample
from radarwow.pipeline import *  # see table above

vol = load_sample('arm_csapr2_c')        # or open_volume('my_file')
# out = SomeAlgorithm()(vol)            # band read from the file, or band='S'|'C'|'X'
```

## Status legend

*implemented + notebook*: code, tests, executed notebook. *planned*: interface fixed, code to write. *stub*: class exists, raises `NotImplementedError`. *to review*: paper to read before deciding.

## License

MIT. Data used in the notebooks comes from public archives (NEXRAD on AWS, ARM, IMD via open-radar-data) under their own terms.
