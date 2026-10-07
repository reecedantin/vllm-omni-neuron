# SPDX-License-Identifier: Apache-2.0
"""Host-side math vendored unchanged (apart from relative imports) from Black Forest Labs'
FLUX Action reference (https://github.com/black-forest-labs/flux-action, Apache-2.0):

* ``sampling.py``   <- ``flux_action/inference/sampling.py`` (Cosmos UniPC order 2, Euler, CFG)
* ``positional.py`` <- ``flux_action/models/positional.py`` (position-id packers)
* ``packing.py``    <- ``flux_action/processing/packing.py`` (camera canvas, token packing)
* ``normalization.py``, ``history.py`` <- ``flux_action/processing/`` (range normalization,
  observation history of the SO-101 profile)
* ``config.py``     <- ``flux_action/config.py`` (``PolicyConfig``, the checkpoint contract)

Keep these files byte-identical to upstream apart from the import lines so a diff against a
newer upstream stays readable.
"""
